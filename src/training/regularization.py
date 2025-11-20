"""
防过拟合策略实现

包含各种正则化技术和防过拟合方法
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 save_path: Optional[str] = None):
        """
        初始化早停机制
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
            mode: 'min'表示指标越小越好，'max'表示越大越好
            save_path: 最佳模型保存路径
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
        
        logger.info(f"早停机制初始化: patience={patience}, mode={mode}")
    
    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            model: 模型
            epoch: 当前epoch
            
        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            logger.info(f"Epoch {epoch}: 指标改善至 {score:.6f}")
        else:
            self.counter += 1
            logger.info(f"Epoch {epoch}: 指标未改善 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"早停触发！最佳epoch: {self.best_epoch}, 最佳指标: {self.best_score:.6f}")
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳模型"""
        if self.save_path:
            torch.save(model.state_dict(), self.save_path)
            logger.debug(f"最佳模型已保存: {self.save_path}")
    
    def load_best_model(self, model: nn.Module):
        """加载最佳模型"""
        if self.save_path and Path(self.save_path).exists():
            model.load_state_dict(torch.load(self.save_path))
            logger.info(f"已加载最佳模型: {self.save_path}")


class GradientClipper:
    """梯度裁剪"""
    
    def __init__(self,
                 max_norm: float = 1.0,
                 norm_type: float = 2.0,
                 clip_method: str = 'norm'):
        """
        初始化梯度裁剪
        
        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
            clip_method: 裁剪方法 ('norm', 'value')
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_method = clip_method
        
        logger.info(f"梯度裁剪初始化: max_norm={max_norm}, method={clip_method}")
    
    def clip(self, parameters) -> float:
        """
        裁剪梯度
        
        Args:
            parameters: 模型参数
            
        Returns:
            float: 裁剪前的梯度范数
        """
        if self.clip_method == 'norm':
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters, self.max_norm, self.norm_type
            )
        elif self.clip_method == 'value':
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
                    p.grad.data.clamp_(-self.max_norm, self.max_norm)
            total_norm = total_norm ** (1. / self.norm_type)
        else:
            raise ValueError(f"未知的裁剪方法: {self.clip_method}")
        
        return total_norm


class DropoutScheduler:
    """Dropout调度器"""
    
    def __init__(self,
                 model: nn.Module,
                 initial_dropout: float = 0.1,
                 final_dropout: float = 0.3,
                 warmup_epochs: int = 10):
        """
        初始化Dropout调度器
        
        Args:
            model: 模型
            initial_dropout: 初始dropout率
            final_dropout: 最终dropout率
            warmup_epochs: 预热epoch数
        """
        self.model = model
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_epochs = warmup_epochs
        
        logger.info(f"Dropout调度器初始化: {initial_dropout} -> {final_dropout}")
    
    def step(self, epoch: int):
        """
        更新dropout率
        
        Args:
            epoch: 当前epoch
        """
        if epoch < self.warmup_epochs:
            dropout_rate = self.initial_dropout + \
                          (self.final_dropout - self.initial_dropout) * \
                          (epoch / self.warmup_epochs)
        else:
            dropout_rate = self.final_dropout
        
        # 更新模型中的dropout层
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
        
        logger.debug(f"Epoch {epoch}: Dropout率更新为 {dropout_rate:.3f}")


class WeightDecayScheduler:
    """权重衰减调度器"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 initial_wd: float = 1e-4,
                 final_wd: float = 1e-2,
                 warmup_epochs: int = 10):
        """
        初始化权重衰减调度器
        
        Args:
            optimizer: 优化器
            initial_wd: 初始权重衰减
            final_wd: 最终权重衰减
            warmup_epochs: 预热epoch数
        """
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.warmup_epochs = warmup_epochs
        
        logger.info(f"权重衰减调度器初始化: {initial_wd} -> {final_wd}")
    
    def step(self, epoch: int):
        """
        更新权重衰减
        
        Args:
            epoch: 当前epoch
        """
        if epoch < self.warmup_epochs:
            weight_decay = self.initial_wd + \
                          (self.final_wd - self.initial_wd) * \
                          (epoch / self.warmup_epochs)
        else:
            weight_decay = self.final_wd
        
        # 更新优化器的权重衰减
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
        
        logger.debug(f"Epoch {epoch}: 权重衰减更新为 {weight_decay:.6f}")


class MixupAugmentation:
    """Mixup数据增强"""
    
    def __init__(self, alpha: float = 0.2):
        """
        初始化Mixup增强
        
        Args:
            alpha: Beta分布参数
        """
        self.alpha = alpha
        logger.info(f"Mixup增强初始化: alpha={alpha}")
    
    def __call__(self,
                 x: torch.Tensor,
                 y: torch.Tensor) -> tuple:
        """
        应用Mixup增强
        
        Args:
            x: 输入数据 [batch_size, ...]
            y: 标签 [batch_size, ...]
            
        Returns:
            tuple: (混合后的x, 混合后的y, lambda)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam


class LabelSmoothing(nn.Module):
    """标签平滑"""
    
    def __init__(self, smoothing: float = 0.1):
        """
        初始化标签平滑
        
        Args:
            smoothing: 平滑系数
        """
        super().__init__()
        self.smoothing = smoothing
        logger.info(f"标签平滑初始化: smoothing={smoothing}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算平滑后的交叉熵损失
        
        Args:
            pred: 预测值 [batch_size, num_classes]
            target: 目标值 [batch_size]
            
        Returns:
            torch.Tensor: 损失值
        """
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        
        return loss


class RegularizationManager:
    """正则化管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化正则化管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 早停
        if config.get('early_stopping', {}).get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=config['early_stopping'].get('patience', 10),
                min_delta=config['early_stopping'].get('min_delta', 0.0),
                mode=config['early_stopping'].get('mode', 'min'),
                save_path=config['early_stopping'].get('save_path')
            )
        else:
            self.early_stopping = None
        
        # 梯度裁剪
        if config.get('gradient_clipping', {}).get('enabled', True):
            self.gradient_clipper = GradientClipper(
                max_norm=config['gradient_clipping'].get('max_norm', 1.0),
                norm_type=config['gradient_clipping'].get('norm_type', 2.0),
                clip_method=config['gradient_clipping'].get('method', 'norm')
            )
        else:
            self.gradient_clipper = None
        
        # Mixup
        if config.get('mixup', {}).get('enabled', False):
            self.mixup = MixupAugmentation(
                alpha=config['mixup'].get('alpha', 0.2)
            )
        else:
            self.mixup = None
        
        # 标签平滑
        if config.get('label_smoothing', {}).get('enabled', False):
            self.label_smoothing = LabelSmoothing(
                smoothing=config['label_smoothing'].get('smoothing', 0.1)
            )
        else:
            self.label_smoothing = None
        
        logger.info("正则化管理器初始化完成")
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor):
        """应用Mixup"""
        if self.mixup:
            return self.mixup(x, y)
        return x, y, 1.0
    
    def clip_gradients(self, parameters) -> Optional[float]:
        """裁剪梯度"""
        if self.gradient_clipper:
            return self.gradient_clipper.clip(parameters)
        return None
    
    def check_early_stopping(self, score: float, model: nn.Module, epoch: int) -> bool:
        """检查早停"""
        if self.early_stopping:
            return self.early_stopping(score, model, epoch)
        return False
    
    def get_label_smoothing_loss(self) -> Optional[nn.Module]:
        """获取标签平滑损失"""
        return self.label_smoothing


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试早停
    print("测试早停机制...")
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    model = nn.Linear(10, 1)
    scores = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88, 0.89]
    
    for epoch, score in enumerate(scores):
        should_stop = early_stopping(score, model, epoch)
        if should_stop:
            print(f"在epoch {epoch}触发早停")
            break
    
    # 测试梯度裁剪
    print("\n测试梯度裁剪...")
    clipper = GradientClipper(max_norm=1.0)
    
    model = nn.Linear(10, 1)
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()
    
    grad_norm = clipper.clip(model.parameters())
    print(f"梯度范数: {grad_norm:.4f}")
    
    # 测试Mixup
    print("\n测试Mixup...")
    mixup = MixupAugmentation(alpha=0.2)
    
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    mixed_x, mixed_y, lam = mixup(x, y)
    print(f"Lambda: {lam:.4f}")
    print(f"混合后形状: {mixed_x.shape}, {mixed_y.shape}")