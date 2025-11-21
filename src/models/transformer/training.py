"""
Transformer训练流程实现

任务3.4.1-3.4.3实现:
1. Transformer预训练循环
2. 梯度裁剪
3. Transformer模型保存
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
import time

from .model import TransformerStateModel
from .auxiliary_tasks import (
    TransformerWithAuxiliaryTasks,
    MultiTaskLoss,
    create_labels_from_returns,
    compute_auxiliary_metrics
)

logger = logging.getLogger(__name__)


class TransformerTrainer:
    """
    任务3.4.1: Transformer预训练循环
    
    实现完整的训练流程
    """
    
    def __init__(self,
                 model: TransformerWithAuxiliaryTasks,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: MultiTaskLoss,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_grad_norm: float = 0.5,
                 log_interval: int = 100):
        """
        初始化训练器
        
        Args:
            model: Transformer模型
            optimizer: 优化器
            loss_fn: 损失函数
            device: 设备
            max_grad_norm: 最大梯度范数(用于梯度裁剪)
            log_interval: 日志记录间隔
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'reg_loss': [],
            'cls_loss': [],
            'metrics': []
        }
        self.val_history = {
            'loss': [],
            'reg_loss': [],
            'cls_loss': [],
            'metrics': []
        }
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        logger.info(f"Transformer训练器初始化: device={device}")
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'direction_accuracy': 0.0,
            'classification_accuracy': 0.0
        }
        
        num_batches = len(train_loader)
        start_time = time.time()
        
        for batch_idx, (sequences, reg_targets) in enumerate(train_loader):
            # 移动到设备
            sequences = sequences.to(self.device)
            reg_targets = reg_targets.to(self.device)
            
            # 创建分类标签
            cls_targets = create_labels_from_returns(reg_targets.squeeze())
            
            # 前向传播
            _, reg_pred, cls_pred = self.model(sequences)
            
            # 计算损失
            loss, loss_dict = self.loss_fn(
                reg_pred, reg_targets,
                cls_pred, cls_targets
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 任务3.4.2: 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # 参数更新
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss_dict['total']
            total_reg_loss += loss_dict['regression']
            total_cls_loss += loss_dict['classification']
            
            # 计算指标
            metrics = compute_auxiliary_metrics(
                reg_pred, reg_targets,
                cls_pred, cls_targets
            )
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            # 日志记录
            if (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss_dict['total']:.4f} "
                    f"(Reg: {loss_dict['regression']:.4f}, "
                    f"Cls: {loss_dict['classification']:.4f}) "
                    f"Time: {elapsed:.2f}s"
                )
        
        # 平均指标
        avg_loss = total_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {
            'loss': avg_loss,
            'reg_loss': avg_reg_loss,
            'cls_loss': avg_cls_loss,
            **avg_metrics
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_metrics = {
            'mse': 0.0,
            'mae': 0.0,
            'direction_accuracy': 0.0,
            'classification_accuracy': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for sequences, reg_targets in val_loader:
                # 移动到设备
                sequences = sequences.to(self.device)
                reg_targets = reg_targets.to(self.device)
                
                # 创建分类标签
                cls_targets = create_labels_from_returns(reg_targets.squeeze())
                
                # 前向传播
                _, reg_pred, cls_pred = self.model(sequences)
                
                # 计算损失
                loss, loss_dict = self.loss_fn(
                    reg_pred, reg_targets,
                    cls_pred, cls_targets
                )
                
                # 累积损失
                total_loss += loss_dict['total']
                total_reg_loss += loss_dict['regression']
                total_cls_loss += loss_dict['classification']
                
                # 计算指标
                metrics = compute_auxiliary_metrics(
                    reg_pred, reg_targets,
                    cls_pred, cls_targets
                )
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
        
        # 平均指标
        avg_loss = total_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        return {
            'loss': avg_loss,
            'reg_loss': avg_reg_loss,
            'cls_loss': avg_cls_loss,
            **avg_metrics
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int,
             save_dir: str = 'models/checkpoints',
             early_stopping_patience: int = 10) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            early_stopping_patience: 早停耐心值
            
        Returns:
            训练历史
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        logger.info(f"开始训练: {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_history['loss'].append(train_metrics['loss'])
            self.train_history['reg_loss'].append(train_metrics['reg_loss'])
            self.train_history['cls_loss'].append(train_metrics['cls_loss'])
            self.train_history['metrics'].append({
                k: v for k, v in train_metrics.items()
                if k not in ['loss', 'reg_loss', 'cls_loss']
            })
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_history['loss'].append(val_metrics['loss'])
            self.val_history['reg_loss'].append(val_metrics['reg_loss'])
            self.val_history['cls_loss'].append(val_metrics['cls_loss'])
            self.val_history['metrics'].append({
                k: v for k, v in val_metrics.items()
                if k not in ['loss', 'reg_loss', 'cls_loss']
            })
            
            # 日志
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Dir Acc: {val_metrics['direction_accuracy']:.4f}, "
                f"Val Cls Acc: {val_metrics['classification_accuracy']:.4f}"
            )
            
            # 任务3.4.3: 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                checkpoint_path = save_path / 'best_transformer.pt'
                self.save_checkpoint(checkpoint_path, val_metrics)
                logger.info(f"保存最佳模型: {checkpoint_path}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停触发: {patience_counter} epochs无改善")
                break
            
            # 定期保存
            if epoch % 10 == 0:
                checkpoint_path = save_path / f'transformer_epoch_{epoch}.pt'
                self.save_checkpoint(checkpoint_path, val_metrics)
        
        logger.info("训练完成")
        
        return {
            'train': self.train_history,
            'val': self.val_history
        }
    
    def save_checkpoint(self,
                       filepath: str,
                       metrics: Optional[Dict] = None) -> None:
        """
        任务3.4.3: 保存模型检查点
        
        Args:
            filepath: 保存路径
            metrics: 当前指标
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        加载模型检查点
        
        Args:
            filepath: 检查点路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        logger.info(f"检查点已加载: {filepath}, epoch={self.current_epoch}")


def create_dataloaders(sequences: torch.Tensor,
                      targets: torch.Tensor,
                      train_ratio: float = 0.8,
                      batch_size: int = 64,
                      shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        sequences: 输入序列 [N, seq_len, input_dim]
        targets: 目标值 [N, 1]
        train_ratio: 训练集比例
        batch_size: 批次大小
        shuffle: 是否打乱
        
    Returns:
        训练加载器, 验证加载器
    """
    # 划分数据集
    n_samples = len(sequences)
    n_train = int(n_samples * train_ratio)
    
    train_sequences = sequences[:n_train]
    train_targets = targets[:n_train]
    val_sequences = sequences[n_train:]
    val_targets = targets[n_train:]
    
    # 创建数据集
    train_dataset = TensorDataset(train_sequences, train_targets)
    val_dataset = TensorDataset(val_sequences, val_targets)
    
    # 创建加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"数据加载器创建: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader