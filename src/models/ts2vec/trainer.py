"""
TS2Vec训练器

实现TS2Vec模型的训练流程，包括:
- 训练循环
- 验证流程
- 早停机制
- 模型保存

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import logging
from pathlib import Path
import json
from tqdm import tqdm

from .model import TS2Vec
from .loss import TS2VecLoss
from .dataset import create_ts2vec_dataloader

logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    学习率调度器
    
    实现Warmup + CosineAnnealing学习率调度。
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        initial_lr: float = 1e-3
    ):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: Warmup轮数
            total_epochs: 总轮数
            min_lr: 最小学习率
            initial_lr: 初始学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.current_epoch = 0
        
        logger.info(f"LearningRateScheduler initialized: "
                   f"warmup_epochs={warmup_epochs}, "
                   f"total_epochs={total_epochs}")
    
    def step(self, epoch: int):
        """
        更新学习率
        
        Args:
            epoch: 当前轮数
        """
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # CosineAnnealing阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * \
                 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            lr = lr.item()
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']


class EarlyStopping:
    """
    早停机制
    
    监控验证损失，在性能不再提升时停止训练。
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
            min_delta: 最小改善量
            mode: 'min'或'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        logger.info(f"EarlyStopping initialized: patience={patience}")
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数（损失或指标）
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # 检查是否改善
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop


class TS2VecTrainer:
    """
    TS2Vec训练器
    
    管理完整的训练流程。
    """
    
    def __init__(
        self,
        model: TS2Vec,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'models/ts2vec'
    ):
        """
        初始化训练器
        
        Args:
            model: TS2Vec模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            weight_decay: 权重衰减
            temperature: 对比学习温度
            device: 设备
            save_dir: 模型保存目录
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
        
        # 损失函数
        self.criterion = TS2VecLoss(temperature=temperature)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"TS2VecTrainer initialized on device: {device}")
        logger.info(f"Model parameters: {model.count_parameters():,}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for view1, view2 in pbar:
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # 前向传播
            _, proj1 = self.model(view1, return_projection=True)
            _, proj2 = self.model(view2, return_projection=True)
            
            # 计算损失
            loss = self.criterion(proj1, proj2)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        验证
        
        Returns:
            平均验证损失
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for view1, view2 in self.val_loader:
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                # 前向传播
                _, proj1 = self.model(view1, return_projection=True)
                _, proj2 = self.model(view2, return_projection=True)
                
                # 计算损失
                loss = self.criterion(proj1, proj2)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        num_epochs: int = 100,
        warmup_epochs: int = 5,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ) -> Dict:
        """
        完整训练流程
        
        Args:
            num_epochs: 训练轮数
            warmup_epochs: Warmup轮数
            early_stopping_patience: 早停容忍轮数
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史
        """
        # 初始化学习率调度器
        scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs
        )
        
        # 初始化早停
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # 更新学习率
            lr = scheduler.step(epoch)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(lr)
            
            # 打印进度
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"LR: {lr:.6f}")
            
            # 保存最佳模型
            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"Best model saved with val_loss: {val_loss:.4f}")
            
            # 早停检查
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("Training completed")
        
        # 保存训练历史
        self.save_history()
        
        return self.history
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False
    ):
        """
        保存检查点
        
        Args:
            epoch: 当前轮数
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        加载检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Checkpoint loaded from: {path}")
        logger.info(f"Resumed from epoch: {checkpoint['epoch']}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved: {history_path}")


# 示例用法
if __name__ == "__main__":
    import numpy as np
    from .augmentation import TimeSeriesAugmentation
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== TS2Vec训练器示例 ===")
    
    # 创建模拟数据
    train_data = np.random.randn(5000, 27)
    val_data = np.random.randn(1000, 27)
    
    # 创建数据加载器
    augmenter = TimeSeriesAugmentation()
    
    train_loader = create_ts2vec_dataloader(
        data=train_data,
        window_size=256,
        stride=128,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        augmenter=augmenter
    )
    
    val_loader = create_ts2vec_dataloader(
        data=val_data,
        window_size=256,
        stride=128,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        augmenter=augmenter
    )
    
    # 创建模型
    model = TS2Vec(input_dim=27)
    
    # 创建训练器
    trainer = TS2VecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        device='cpu'  # 示例使用CPU
    )
    
    # 训练（仅2个epoch用于演示）
    print("\n开始训练...")
    history = trainer.train(
        num_epochs=2,
        warmup_epochs=1,
        early_stopping_patience=5
    )
    
    print("\n训练完成!")
    print(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")