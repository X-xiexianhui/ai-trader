"""
LSTM+Attention神经网络训练脚本

功能：
1. 从processed加载训练数据和验证数据
2. 读取OHLCV数据并归一化
3. 训练LSTM+Attention神经网络
4. 保存最佳模型和训练日志
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm
import json
import pickle

from src.models.lstm_attention import LSTMAttentionModel, MultiTaskLoss, create_model
from src.models.data_loader import create_dataloaders
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    name='lstm_attention_training',
    log_level='INFO',
    log_dir='logs',
    log_file='lstm_attention_training.log'
)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善幅度
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            val_loss: 验证集损失
            
        Returns:
            是否应该早停
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        save_dir: Path,
        tensorboard_dir: Path
    ):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 模型保存目录
            tensorboard_dir: TensorBoard日志目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(tensorboard_dir)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_class_loss': [],
            'val_class_loss': [],
            'train_vol_loss': [],
            'val_vol_loss': [],
            'train_ret_loss': [],
            'val_ret_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_vol_loss = 0.0
        total_ret_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            sequences = batch['sequence'].to(self.device)
            market_states = batch['market_state'].to(self.device)
            volatilities = batch['volatility'].to(self.device)
            returns = batch['returns'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # 计算损失
            loss, loss_dict = self.criterion(
                outputs['class_logits'],
                market_states,
                outputs['volatility'],
                volatilities,
                outputs['returns'],
                returns
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss_dict['total']
            total_class_loss += loss_dict['classification']
            total_vol_loss += loss_dict['volatility']
            total_ret_loss += loss_dict['returns']
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'cls': f"{loss_dict['classification']:.4f}",
                'vol': f"{loss_dict['volatility']:.4f}",
                'ret': f"{loss_dict['returns']:.4f}"
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_vol_loss = total_vol_loss / len(self.train_loader)
        avg_ret_loss = total_ret_loss / len(self.train_loader)
        
        return {
            'total': avg_loss,
            'classification': avg_class_loss,
            'volatility': avg_vol_loss,
            'returns': avg_ret_loss
        }
    
    def validate_epoch(self, epoch: int) -> dict:
        """验证一个epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_class_loss = 0.0
        total_vol_loss = 0.0
        total_ret_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                # 将数据移到设备
                sequences = batch['sequence'].to(self.device)
                market_states = batch['market_state'].to(self.device)
                volatilities = batch['volatility'].to(self.device)
                returns = batch['returns'].to(self.device)
                
                # 前向传播
                outputs = self.model(sequences)
                
                # 计算损失
                loss, loss_dict = self.criterion(
                    outputs['class_logits'],
                    market_states,
                    outputs['volatility'],
                    volatilities,
                    outputs['returns'],
                    returns
                )
                
                # 累计损失
                total_loss += loss_dict['total']
                total_class_loss += loss_dict['classification']
                total_vol_loss += loss_dict['volatility']
                total_ret_loss += loss_dict['returns']
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'cls': f"{loss_dict['classification']:.4f}",
                    'vol': f"{loss_dict['volatility']:.4f}",
                    'ret': f"{loss_dict['returns']:.4f}"
                })
        
        # 计算平均损失
        avg_loss = total_loss / len(self.val_loader)
        avg_class_loss = total_class_loss / len(self.val_loader)
        avg_vol_loss = total_vol_loss / len(self.val_loader)
        avg_ret_loss = total_ret_loss / len(self.val_loader)
        
        return {
            'total': avg_loss,
            'classification': avg_class_loss,
            'volatility': avg_vol_loss,
            'returns': avg_ret_loss
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        # 保存最新检查点
        checkpoint_path = self.save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """训练模型"""
        logger.info("=" * 80)
        logger.info("开始训练LSTM+Attention模型")
        logger.info("=" * 80)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 80)
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_losses = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step(val_losses['total'])
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_class_loss'].append(train_losses['classification'])
            self.history['val_class_loss'].append(val_losses['classification'])
            self.history['train_vol_loss'].append(train_losses['volatility'])
            self.history['val_vol_loss'].append(val_losses['volatility'])
            self.history['train_ret_loss'].append(train_losses['returns'])
            self.history['val_ret_loss'].append(val_losses['returns'])
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/val', val_losses['total'], epoch)
            self.writer.add_scalar('Loss/train_classification', train_losses['classification'], epoch)
            self.writer.add_scalar('Loss/val_classification', val_losses['classification'], epoch)
            self.writer.add_scalar('Loss/train_volatility', train_losses['volatility'], epoch)
            self.writer.add_scalar('Loss/val_volatility', val_losses['volatility'], epoch)
            self.writer.add_scalar('Loss/train_returns', train_losses['returns'], epoch)
            self.writer.add_scalar('Loss/val_returns', val_losses['returns'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # 打印结果
            logger.info(f"训练损失: {train_losses['total']:.4f} "
                       f"(cls: {train_losses['classification']:.4f}, "
                       f"vol: {train_losses['volatility']:.4f}, "
                       f"ret: {train_losses['returns']:.4f})")
            logger.info(f"验证损失: {val_losses['total']:.4f} "
                       f"(cls: {val_losses['classification']:.4f}, "
                       f"vol: {val_losses['volatility']:.4f}, "
                       f"ret: {val_losses['returns']:.4f})")
            logger.info(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                logger.info(f"✓ 新的最佳验证损失: {self.best_val_loss:.4f}")
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 早停检查
            if early_stopping(val_losses['total']):
                logger.info(f"\n早停触发！最佳epoch: {self.best_epoch}, 最佳验证损失: {self.best_val_loss:.4f}")
                break
        
        # 保存训练历史
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"训练历史已保存: {history_path}")
        
        self.writer.close()
        
        logger.info("\n" + "=" * 80)
        logger.info("训练完成！")
        logger.info("=" * 80)
        logger.info(f"最佳epoch: {self.best_epoch}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("LSTM+Attention神经网络训练")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据路径
    data_dir = project_root / 'data' / 'processed'
    train_path = data_dir / 'MES_train.csv'
    val_path = data_dir / 'MES_val.csv'
    test_path = data_dir / 'MES_test.csv'
    
    # 检查数据文件
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        logger.error("数据文件不存在，请先运行 training/split_dataset.py")
        return
    
    # 超参数
    config = {
        'input_size': 5,  # OHLCV
        'hidden_size': 128,
        'num_layers': 2,
        'state_vector_size': 64,
        'num_classes': 4,
        'dropout': 0.2,
        'seq_len': 60,
        'future_periods': 5,
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'early_stopping_patience': 15,
        'loss_weights': {
            'alpha': 1.0,  # 分类
            'beta': 1.0,   # 波动率
            'gamma': 1.0   # 收益率
        }
    }
    
    logger.info("\n配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建数据加载器
    logger.info("\n创建数据加载器...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        str(train_path),
        str(val_path),
        str(test_path),
        seq_len=config['seq_len'],
        future_periods=config['future_periods'],
        batch_size=config['batch_size'],
        num_workers=4,
        scaler_type='standard'
    )
    
    # 保存归一化器
    scaler_dir = project_root / 'models' / 'lstm_attention'
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"归一化器已保存: {scaler_path}")
    
    # 创建模型
    logger.info("\n创建模型...")
    model = create_model(config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建损失函数
    criterion = MultiTaskLoss(
        alpha=config['loss_weights']['alpha'],
        beta=config['loss_weights']['beta'],
        gamma=config['loss_weights']['gamma']
    )
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 创建保存目录
    save_dir = project_root / 'models' / 'lstm_attention'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    tensorboard_dir = project_root / 'logs' / 'tensorboard' / datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置已保存: {config_path}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        tensorboard_dir=tensorboard_dir
    )
    
    # 开始训练
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    logger.info(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型保存在: {save_dir}")
    logger.info(f"TensorBoard日志: {tensorboard_dir}")
    logger.info("\n查看训练过程: tensorboard --logdir=" + str(tensorboard_dir.parent))


if __name__ == '__main__':
    main()