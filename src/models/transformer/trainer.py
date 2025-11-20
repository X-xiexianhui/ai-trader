"""
Transformer训练器实现

实现Transformer模型的训练流程，包括:
- 多任务学习
- 学习率调度
- 早停机制
- 模型保存
- 训练监控

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json
import time

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    Warmup + 余弦退火学习率调度器
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        """
        初始化调度器
        
        Args:
            optimizer: 优化器
            warmup_steps: Warmup步数
            total_steps: 总训练步数
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        logger.info(f"WarmupCosineScheduler initialized: "
                   f"warmup_steps={warmup_steps}, total_steps={total_steps}")
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        lr = self._get_lr()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # Warmup阶段：线性增长
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            # 余弦退火阶段
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
    
    def get_last_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        初始化早停
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改进量
            mode: 'min'表示越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop


class TransformerTrainer:
    """
    Transformer训练器
    
    实现完整的训练流程，包括多任务学习、学习率调度、早停等。
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        regression_weight: float = 1.0,
        classification_weight: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'models/transformer',
        patience: int = 10
    ):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: Warmup步数
            max_epochs: 最大训练轮数
            grad_clip: 梯度裁剪阈值
            regression_weight: 回归任务权重
            classification_weight: 分类任务权重
            device: 设备
            save_dir: 模型保存目录
            patience: 早停容忍轮数
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * max_epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # 早停
        self.early_stopping = EarlyStopping(patience=patience, mode='min')
        
        # 损失函数
        self.regression_criterion = nn.MSELoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        
        # 超参数
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_reg_loss': [],
            'train_cls_loss': [],
            'val_loss': [],
            'val_reg_loss': [],
            'val_cls_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"TransformerTrainer initialized on {device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def compute_loss(
        self,
        state_vector: torch.Tensor,
        reg_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        reg_target: torch.Tensor,
        cls_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            state_vector: 状态向量
            reg_pred: 回归预测
            cls_pred: 分类预测
            reg_target: 回归目标
            cls_target: 分类目标
            
        Returns:
            total_loss: 总损失
            reg_loss: 回归损失
            cls_loss: 分类损失
        """
        # 回归损失
        reg_loss = self.regression_criterion(reg_pred, reg_target)
        
        # 分类损失
        batch_size, seq_len, num_classes = cls_pred.shape
        cls_pred_flat = cls_pred.view(-1, num_classes)
        cls_target_flat = cls_target.view(-1)
        cls_loss = self.classification_criterion(cls_pred_flat, cls_target_flat)
        
        # 总损失
        total_loss = (
            self.regression_weight * reg_loss +
            self.classification_weight * cls_loss
        )
        
        return total_loss, reg_loss, cls_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # 移动数据到设备
            ts2vec_emb = batch['ts2vec_emb'].to(self.device)
            manual_features = batch['manual_features'].to(self.device)
            reg_target = batch['regression_target'].to(self.device)
            cls_target = batch['classification_target'].to(self.device)
            
            # 前向传播
            state_vector, reg_pred, cls_pred = self.model(
                ts2vec_emb, manual_features
            )
            
            # 计算损失
            loss, reg_loss, cls_loss = self.compute_loss(
                state_vector, reg_pred, cls_pred, reg_target, cls_target
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新统计
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr():.6f}"
            })
        
        # 计算平均指标
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_reg_loss': total_reg_loss / num_batches,
            'train_cls_loss': total_cls_loss / num_batches,
            'learning_rate': self.scheduler.get_last_lr()
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # 移动数据到设备
            ts2vec_emb = batch['ts2vec_emb'].to(self.device)
            manual_features = batch['manual_features'].to(self.device)
            reg_target = batch['regression_target'].to(self.device)
            cls_target = batch['classification_target'].to(self.device)
            
            # 前向传播
            state_vector, reg_pred, cls_pred = self.model(
                ts2vec_emb, manual_features
            )
            
            # 计算损失
            loss, reg_loss, cls_loss = self.compute_loss(
                state_vector, reg_pred, cls_pred, reg_target, cls_target
            )
            
            # 更新统计
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()
            num_batches += 1
        
        # 计算平均指标
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_reg_loss': total_reg_loss / num_batches,
            'val_cls_loss': total_cls_loss / num_batches
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存检查点
        
        Args:
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # 保存最新检查点
        checkpoint_path = self.save_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def train(self):
        """执行完整训练"""
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics}
            
            # 更新历史
            for key, value in metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # 打印指标
            log_str = f"Epoch {epoch + 1}/{self.max_epochs}"
            for key, value in metrics.items():
                log_str += f" | {key}: {value:.4f}"
            logger.info(log_str)
            
            # 保存检查点
            is_best = False
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                is_best = True
            
            self.save_checkpoint(is_best=is_best)
            
            # 早停检查
            if val_metrics and self.early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 训练完成
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")
        
        # 保存训练历史
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


# 示例用法
if __name__ == "__main__":
    from .model import TransformerModel
    from .dataset import TransformerDataModule
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Transformer训练器示例 ===")
    
    # 创建模拟数据
    n_train = 1000
    n_val = 200
    ts2vec_dim = 128
    feature_dim = 27
    
    train_ts2vec = np.random.randn(n_train, ts2vec_dim)
    train_features = np.random.randn(n_train, feature_dim)
    train_prices = np.cumsum(np.random.randn(n_train) * 0.01) + 100
    
    val_ts2vec = np.random.randn(n_val, ts2vec_dim)
    val_features = np.random.randn(n_val, feature_dim)
    val_prices = np.cumsum(np.random.randn(n_val) * 0.01) + 100
    
    # 创建数据模块
    data_module = TransformerDataModule(
        train_ts2vec=train_ts2vec,
        train_features=train_features,
        train_prices=train_prices,
        val_ts2vec=val_ts2vec,
        val_features=val_features,
        val_prices=val_prices,
        seq_len=64,
        batch_size=16
    )
    
    # 创建模型
    model = TransformerModel(
        ts2vec_dim=128,
        manual_dim=27,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # 创建训练器
    trainer = TransformerTrainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        learning_rate=1e-4,
        max_epochs=3,
        device='cpu'
    )
    
    # 训练
    print("\n开始训练...")
    trainer.train()
    
    print("\n示例完成!")