"""
TS2Vec训练模块

任务2.3.1-2.3.4实现:
1. TS2Vec训练循环
2. 学习率调度器(Warmup+CosineAnnealing)
3. 早停机制
4. 模型检查点保存
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    任务2.3.2: Warmup + Cosine Annealing学习率调度器
    """
    
    def __init__(self,
                 optimizer,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 min_lr: float = 1e-6):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup阶段的epoch数
            total_epochs: 总epoch数
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
        # 获取初始学习率
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Cosine调度器(从warmup结束后开始)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        self.current_epoch = 0
        
        logger.info(f"学习率调度器初始化: Warmup={warmup_epochs}, Total={total_epochs}")
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段: 线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine Annealing阶段
            self.cosine_scheduler.step()
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """
    任务2.3.3: 早停机制
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min'):
        """
        初始化早停控制器
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
            mode: 'min'表示越小越好,'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        logger.info(f"早停机制初始化: patience={patience}, min_delta={min_delta}")
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            epoch: 当前epoch
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # 判断是否改善
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"早停触发: 连续{self.patience}个epoch无改善")
        
        return self.early_stop


class ModelCheckpoint:
    """
    任务2.3.4: 模型检查点保存
    """
    
    def __init__(self,
                 save_dir: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True):
        """
        初始化检查点管理器
        
        Args:
            save_dir: 保存目录
            monitor: 监控的指标
            mode: 'min'或'max'
            save_best_only: 是否只保存最佳模型
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        
        self.best_score = None
        
        logger.info(f"检查点管理器初始化: {save_dir}")
    
    def save(self,
             model,
             optimizer,
             scheduler,
             epoch: int,
             score: float,
             metrics: Dict) -> bool:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            score: 监控指标值
            metrics: 其他指标
            
        Returns:
            是否保存了新的最佳模型
        """
        is_best = False
        
        # 判断是否是最佳模型
        if self.best_score is None:
            is_best = True
        elif self.mode == 'min':
            is_best = score < self.best_score
        else:
            is_best = score > self.best_score
        
        if is_best:
            self.best_score = score
        
        # 决定是否保存
        should_save = (not self.save_best_only) or is_best
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.cosine_scheduler.state_dict() if hasattr(scheduler, 'cosine_scheduler') else None,
                'score': score,
                'metrics': metrics
            }
            
            if is_best:
                filepath = self.save_dir / 'best_model.pt'
                torch.save(checkpoint, filepath)
                logger.info(f"保存最佳模型: epoch={epoch}, {self.monitor}={score:.4f}")
            else:
                filepath = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
                torch.save(checkpoint, filepath)
                logger.info(f"保存检查点: epoch={epoch}")
        
        return is_best


class TS2VecTrainer:
    """
    任务2.3.1: TS2Vec训练器
    
    完整的训练流程管理
    """
    
    def __init__(self,
                 model,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 0.001,
                 num_epochs: int = 100,
                 warmup_epochs: int = 5,
                 patience: int = 10,
                 save_dir: str = 'models/checkpoints/ts2vec',
                 device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            model: TS2Vec模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            num_epochs: 训练轮数
            warmup_epochs: Warmup轮数
            patience: 早停patience
            save_dir: 模型保存目录
            device: 计算设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs
        )
        
        # 早停
        self.early_stopping = EarlyStopping(patience=patience)
        
        # 检查点
        self.checkpoint = ModelCheckpoint(save_dir)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info("TS2Vec训练器初始化完成")
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for x_i, x_j in pbar:
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            
            # 前向传播
            loss = self.model(x_i, x_j, return_loss=True)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_i, x_j in self.val_loader:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)
                
                loss = self.model(x_i, x_j, return_loss=True)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self) -> Dict:
        """
        完整训练流程
        
        Returns:
            训练历史
        """
        logger.info(f"开始训练TS2Vec模型: {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # 日志
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            }
            self.checkpoint.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch + 1,
                val_loss,
                metrics
            )
            
            # 早停检查
            if self.early_stopping(val_loss, epoch + 1):
                logger.info(f"早停触发,在epoch {epoch+1}停止训练")
                break
        
        logger.info("训练完成!")
        
        return self.history
    
    def load_best_model(self):
        """加载最佳模型"""
        checkpoint_path = self.checkpoint.save_dir / 'best_model.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"已加载最佳模型: epoch={checkpoint['epoch']}")
        else:
            logger.warning("未找到最佳模型检查点")