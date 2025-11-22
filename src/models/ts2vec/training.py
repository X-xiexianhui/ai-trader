"""
TS2Vec 训练模块（统一版本）

集成基础训练和高级优化功能:
1. 基础训练循环
2. 学习率调度器（Warmup + CosineAnnealing）
3. 早停机制
4. 模型检查点保存
5. 混合精度训练（AMP）
6. 梯度累积
7. 批次大小自动调优
8. torch.compile 编译优化
9. 优化的 DataLoader 配置
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器
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
            warmup_epochs: Warmup 阶段的 epoch 数
            total_epochs: 总 epoch 数
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
        # 获取初始学习率
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Cosine 调度器（从 warmup 结束后开始）
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
            # Warmup 阶段: 线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine Annealing 阶段
            self.cosine_scheduler.step()
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min'):
        """
        初始化早停控制器
        
        Args:
            patience: 容忍的 epoch 数
            min_delta: 最小改善幅度
            mode: 'min' 表示越小越好，'max' 表示越大越好
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
            epoch: 当前 epoch
            
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
                logger.info(f"早停触发: 连续 {self.patience} 个 epoch 无改善")
        
        return self.early_stop


class ModelCheckpoint:
    """
    模型检查点保存
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
            mode: 'min' 或 'max'
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
            epoch: 当前 epoch
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


class OptimizedDataLoader:
    """
    优化的数据加载器配置
    
    自动根据设备和数据集大小配置最优参数
    """
    
    @staticmethod
    def create_loader(dataset,
                     batch_size: int,
                     shuffle: bool = True,
                     device: str = 'cpu',
                     num_workers: Optional[int] = None,
                     prefetch_factor: int = 2) -> DataLoader:
        """
        创建优化的数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            device: 计算设备
            num_workers: 工作进程数（None 表示自动）
            prefetch_factor: 预取因子
            
        Returns:
            优化的 DataLoader
        """
        is_cuda = device.startswith('cuda')
        
        # 自动确定 num_workers
        if num_workers is None:
            if is_cuda:
                # GPU 训练时使用多进程加速数据加载
                import multiprocessing
                num_workers = min(4, multiprocessing.cpu_count())
            else:
                # CPU 训练时避免多进程开销
                num_workers = 0
        
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': is_cuda,  # 固定内存加速 CPU->GPU 传输
            'drop_last': shuffle,  # 训练时丢弃最后不完整的批次
        }
        
        # 多进程时的额外配置
        if num_workers > 0:
            loader_kwargs['persistent_workers'] = True  # 保持 worker 进程
            loader_kwargs['prefetch_factor'] = prefetch_factor  # 预取批次数
        
        loader = DataLoader(dataset, **loader_kwargs)
        
        logger.info(
            f"创建 DataLoader: batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={is_cuda}"
        )
        
        return loader


class BatchSizeOptimizer:
    """
    批次大小自动调优器
    
    根据 GPU 显存自动找到最优批次大小
    """
    
    @staticmethod
    def find_optimal_batch_size(model,
                                sample_input: torch.Tensor,
                                device: str = 'cuda',
                                min_batch_size: int = 8,
                                max_batch_size: int = 512,
                                target_memory_usage: float = 0.8) -> int:
        """
        二分搜索找到最优批次大小
        
        Args:
            model: 模型
            sample_input: 样本输入 [1, seq_len, input_dim]
            device: 设备
            min_batch_size: 最小批次大小
            max_batch_size: 最大批次大小
            target_memory_usage: 目标显存使用率
            
        Returns:
            最优批次大小
        """
        if not device.startswith('cuda'):
            logger.warning("非 CUDA 设备，返回默认批次大小 32")
            return 32
        
        model = model.to(device)
        model.train()
        
        # 获取总显存
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = total_memory * target_memory_usage
        
        logger.info(f"开始批次大小优化: 目标显存使用 {target_memory_usage*100:.0f}%")
        
        optimal_batch_size = min_batch_size
        left, right = min_batch_size, max_batch_size
        
        while left <= right:
            batch_size = (left + right) // 2
            
            try:
                # 清空缓存
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                # 创建批次数据
                batch_input = sample_input.repeat(batch_size, 1, 1).to(device)
                
                # 前向传播
                with autocast(enabled=True):
                    # 假设是对比学习，需要两个视图
                    loss = model(batch_input, batch_input, return_loss=True)
                
                # 反向传播
                loss.backward()
                
                # 检查显存使用
                peak_memory = torch.cuda.max_memory_allocated()
                
                if peak_memory < target_memory:
                    optimal_batch_size = batch_size
                    left = batch_size + 1
                    logger.info(f"  批次大小 {batch_size}: ✓ ({peak_memory/1024**3:.2f} GB)")
                else:
                    right = batch_size - 1
                    logger.info(f"  批次大小 {batch_size}: ✗ 显存不足")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    right = batch_size - 1
                    logger.info(f"  批次大小 {batch_size}: ✗ OOM")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        logger.info(f"最优批次大小: {optimal_batch_size}")
        
        # 清理
        torch.cuda.empty_cache()
        
        return optimal_batch_size


class TS2VecTrainer:
    """
    TS2Vec 训练器（统一版本）
    
    支持基础训练和所有高级优化功能
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
                 device: str = 'cpu',
                 # 优化参数（默认禁用以保持向后兼容）
                 use_amp: bool = False,
                 gradient_accumulation_steps: int = 1,
                 use_compile: bool = False,
                 optimizer_type: str = 'adam',
                 weight_decay: float = 0.0,
                 grad_clip_norm: Optional[float] = None):
        """
        初始化训练器
        
        Args:
            model: TS2Vec 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            num_epochs: 训练轮数
            warmup_epochs: Warmup 轮数
            patience: 早停 patience
            save_dir: 模型保存目录
            device: 计算设备
            use_amp: 是否使用混合精度训练（默认 False）
            gradient_accumulation_steps: 梯度累积步数（默认 1）
            use_compile: 是否使用 torch.compile（默认 False）
            optimizer_type: 优化器类型 'adam' 或 'adamw'（默认 'adam'）
            weight_decay: 权重衰减（默认 0.0）
            grad_clip_norm: 梯度裁剪范数（默认 None）
        """
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        
        # 模型移至设备
        self.model = model.to(device)
        
        # torch.compile 优化（PyTorch 2.0+）
        if use_compile and hasattr(torch, 'compile'):
            try:
                logger.info("使用 torch.compile 编译模型...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("✓ 模型编译成功")
            except Exception as e:
                logger.warning(f"torch.compile 失败: {e}，继续使用未编译模型")
        elif use_compile:
            logger.warning("torch.compile 需要 PyTorch 2.0+，跳过编译")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        if optimizer_type.lower() == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            logger.info(f"使用 AdamW 优化器 (weight_decay={weight_decay})")
        else:
            self.optimizer = Adam(model.parameters(), lr=learning_rate)
            logger.info("使用 Adam 优化器")
        
        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs
        )
        
        # 混合精度训练
        self.use_amp = use_amp and device.startswith('cuda')
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("✓ 启用混合精度训练（AMP）")
        else:
            self.scaler = None
            if use_amp and not device.startswith('cuda'):
                logger.warning("AMP 需要 CUDA，已禁用")
        
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
        
        # 统计信息
        self.total_steps = 0
        self.best_val_loss = float('inf')
        
        logger.info("TS2Vec 训练器初始化完成")
        logger.info(f"  设备: {device}")
        logger.info(f"  混合精度: {self.use_amp}")
        logger.info(f"  梯度累积步数: {gradient_accumulation_steps}")
        logger.info(f"  梯度裁剪: {grad_clip_norm}")
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # 统计信息
        stats = {
            'grad_norm': [],
            'loss_scale': []
        }
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (x_i, x_j) in enumerate(pbar):
            x_i = x_i.to(self.device)
            x_j = x_j.to(self.device)
            
            # 混合精度前向传播
            with autocast(enabled=self.use_amp):
                loss = self.model(x_i, x_j, return_loss=True)
                # 梯度累积时缩放损失
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.grad_clip_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )
                    stats['grad_norm'].append(grad_norm.item())
                
                # 优化器步进
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    stats['loss_scale'].append(self.scaler.get_scale())
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.total_steps += 1
            
            # 记录（注意要乘回累积步数）
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'step': self.total_steps
            })
        
        avg_loss = total_loss / num_batches
        
        # 计算统计信息
        if stats['grad_norm']:
            stats['avg_grad_norm'] = np.mean(stats['grad_norm'])
        if stats['loss_scale']:
            stats['avg_loss_scale'] = np.mean(stats['loss_scale'])
        
        return avg_loss, stats
    
    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_i, x_j in self.val_loader:
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)
                
                # 混合精度推理
                with autocast(enabled=self.use_amp):
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
        logger.info(f"开始训练 TS2Vec 模型: {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # 训练
            train_loss, train_stats = self.train_epoch()
            
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
            log_msg = (
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            if 'avg_grad_norm' in train_stats:
                log_msg += f", Grad Norm: {train_stats['avg_grad_norm']:.4f}"
            
            if 'avg_loss_scale' in train_stats:
                log_msg += f", Loss Scale: {train_stats['avg_loss_scale']:.0f}"
            
            logger.info(log_msg)
            
            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                **train_stats
            }
            
            is_best = self.checkpoint.save(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch + 1,
                val_loss,
                metrics
            )
            
            if is_best:
                self.best_val_loss = val_loss
            
            # 早停检查
            if self.early_stopping(val_loss, epoch + 1):
                logger.info(f"早停触发，在 epoch {epoch+1} 停止训练")
                break
        
        logger.info("训练完成!")
        if self.best_val_loss != float('inf'):
            logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        
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
    
    def get_memory_stats(self) -> Dict:
        """获取显存统计信息"""
        if not self.device.startswith('cuda'):
            return {}
        
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
        }