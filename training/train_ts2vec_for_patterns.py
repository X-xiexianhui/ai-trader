"""
针对形态识别优化的TS2Vec训练脚本
使用改进的数据增强和训练策略
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from pathlib import Path
import sys
from typing import Tuple

sys.path.append(str(Path(__file__).parent.parent))

from src.models.ts2vec.model import DilatedConvEncoder
from src.models.ts2vec.training import TS2VecTrainer
from src.models.ts2vec.data_preparation import TS2VecDataset, OptimizedDataLoader
from src.utils.logger import setup_logger


class PatternAwareAugmentation:
    """
    针对形态识别的数据增强
    保持形态特征的同时增加数据多样性
    """
    
    def __init__(self, 
                 jitter_strength: float = 0.01,
                 scaling_strength: float = 0.05,
                 rotation_strength: float = 0.1,
                 time_warp_strength: float = 0.05):
        """
        初始化增强参数
        
        Args:
            jitter_strength: 抖动强度（相对于价格范围）
            scaling_strength: 缩放强度
            rotation_strength: 旋转强度（改变趋势斜率）
            time_warp_strength: 时间扭曲强度
        """
        self.jitter_strength = jitter_strength
        self.scaling_strength = scaling_strength
        self.rotation_strength = rotation_strength
        self.time_warp_strength = time_warp_strength
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加轻微抖动，模拟市场噪声
        保持整体形态不变
        """
        noise = torch.randn_like(x) * self.jitter_strength * x.std()
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """
        垂直缩放，模拟不同价格水平的相同形态
        """
        scale = 1 + (torch.rand(1) - 0.5) * 2 * self.scaling_strength
        mean = x.mean(dim=1, keepdim=True)
        return mean + (x - mean) * scale
    
    def rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        旋转（改变趋势），模拟不同市场环境下的形态
        """
        batch_size, seq_len, n_features = x.shape
        
        # 创建线性趋势
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        slope = (torch.rand(batch_size, 1, n_features).to(x.device) - 0.5) * 2 * self.rotation_strength
        trend = slope * t * x.std(dim=1, keepdim=True)
        
        return x + trend
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        时间扭曲，模拟形态的时间压缩/拉伸
        保持形态特征但改变时间尺度
        """
        batch_size, seq_len, n_features = x.shape
        
        # 创建扭曲的时间索引
        orig_steps = torch.arange(seq_len, dtype=torch.float32)
        warp_steps = orig_steps + torch.randn(seq_len) * self.time_warp_strength * seq_len
        warp_steps = torch.clamp(warp_steps, 0, seq_len - 1)
        
        # 插值
        warped = []
        for i in range(batch_size):
            warped_sample = []
            for j in range(n_features):
                # 使用线性插值
                interp = torch.nn.functional.interpolate(
                    x[i, :, j].unsqueeze(0).unsqueeze(0),
                    size=seq_len,
                    mode='linear',
                    align_corners=True
                )
                warped_sample.append(interp.squeeze())
            warped.append(torch.stack(warped_sample, dim=-1))
        
        return torch.stack(warped)
    
    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        幅度扭曲，改变波动幅度但保持形态
        """
        batch_size, seq_len, n_features = x.shape
        
        # 创建平滑的幅度调制曲线
        knots = torch.rand(batch_size, 4, n_features).to(x.device) * 0.5 + 0.75  # 0.75-1.25
        t = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1).to(x.device)
        
        # 三次样条插值
        magnitude_curve = (
            knots[:, 0:1] * (1 - t)**3 +
            knots[:, 1:2] * 3 * (1 - t)**2 * t +
            knots[:, 2:3] * 3 * (1 - t) * t**2 +
            knots[:, 3:4] * t**3
        )
        
        mean = x.mean(dim=1, keepdim=True)
        return mean + (x - mean) * magnitude_curve
    
    def window_slice(self, x: torch.Tensor, slice_ratio: float = 0.9) -> torch.Tensor:
        """
        随机切片，关注形态的不同部分
        """
        batch_size, seq_len, n_features = x.shape
        slice_len = int(seq_len * slice_ratio)
        
        start_idx = torch.randint(0, seq_len - slice_len + 1, (batch_size,))
        
        sliced = []
        for i in range(batch_size):
            s = start_idx[i]
            sliced.append(x[i, s:s+slice_len])
        
        # 插值回原始长度
        sliced = torch.stack(sliced)
        return torch.nn.functional.interpolate(
            sliced.transpose(1, 2),
            size=seq_len,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)
    
    def __call__(self, x: torch.Tensor, augmentation_prob: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用随机增强组合
        
        Returns:
            两个增强后的视图
        """
        def apply_random_augmentations(x):
            # 每种增强有一定概率被应用
            if torch.rand(1) < augmentation_prob:
                x = self.jitter(x)
            if torch.rand(1) < augmentation_prob:
                x = self.scaling(x)
            if torch.rand(1) < augmentation_prob * 0.7:  # 旋转概率稍低
                x = self.rotation(x)
            if torch.rand(1) < augmentation_prob * 0.5:  # 时间扭曲概率更低
                x = self.time_warp(x)
            if torch.rand(1) < augmentation_prob:
                x = self.magnitude_warp(x)
            if torch.rand(1) < augmentation_prob * 0.6:
                x = self.window_slice(x)
            return x
        
        # 生成两个不同的增强视图
        x1 = apply_random_augmentations(x.clone())
        x2 = apply_random_augmentations(x.clone())
        
        return x1, x2


class PatternFocusedTrainer(TS2VecTrainer):
    """
    针对形态识别优化的训练器
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 使用改进的数据增强
        self.augmentation = PatternAwareAugmentation(
            jitter_strength=0.01,
            scaling_strength=0.05,
            rotation_strength=0.1,
            time_warp_strength=0.05
        )
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算对比学习损失
        使用改进的增强策略
        """
        # 应用数据增强
        x1, x2 = self.augmentation(x, augmentation_prob=0.8)
        
        # 获取embeddings
        z1 = self.model(x1)  # [batch, time, dim]
        z2 = self.model(x2)
        
        # 时间维度平均
        z1 = z1.mean(dim=1)  # [batch, dim]
        z2 = z2.mean(dim=1)
        
        # NT-Xent损失
        loss = self.nt_xent_loss(z1, z2)
        
        return loss
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失
        
        Args:
            z1, z2: 两个视图的embeddings [batch, dim]
            temperature: 温度参数
        """
        batch_size = z1.shape[0]
        
        # L2归一化
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # 拼接
        z = torch.cat([z1, z2], dim=0)  # [2*batch, dim]
        
        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / temperature  # [2*batch, 2*batch]
        
        # 创建标签：对角线上的对应位置为正样本
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])  # [2*batch]
        
        # 移除自身相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)
        
        # 计算损失
        loss = nn.functional.cross_entropy(sim_matrix, labels)
        
        return loss


def main():
    """主训练流程"""
    # 设置日志
    logger = setup_logger(
        name='ts2vec_pattern_training',
        log_file='logs/ts2vec_pattern_training.log',
        log_level='INFO'
    )
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 修改配置以更好地捕捉形态
    config['model']['depth'] = 12  # 增加深度以捕捉更复杂的形态
    config['model']['hidden_dims'] = 128  # 增加隐藏维度
    config['training']['epochs'] = 200  # 更多训练轮次
    config['training']['learning_rate'] = 0.0005  # 稍低的学习率
    config['training']['warmup_epochs'] = 20  # 更长的预热
    
    logger.info("训练配置:")
    logger.info(f"  模型深度: {config['model']['depth']}")
    logger.info(f"  隐藏维度: {config['model']['hidden_dims']}")
    logger.info(f"  训练轮次: {config['training']['epochs']}")
    logger.info(f"  学习率: {config['training']['learning_rate']}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载训练数据...")
    train_df = pd.read_csv('data/processed/MES_train.csv')
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df = train_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                       'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    logger.info("加载验证数据...")
    val_df = pd.read_csv('data/processed/MES_val.csv')
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df = val_df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                   'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    
    # 创建数据集
    train_dataset = TS2VecDataset(
        df=train_df,
        window_length=config['data']['window_length'],
        stride=config['data']['stride']
    )
    
    val_dataset = TS2VecDataset(
        df=val_df,
        window_length=config['data']['window_length'],
        stride=config['data']['stride']
    )
    
    # 创建数据加载器
    train_loader = OptimizedDataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = OptimizedDataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = DilatedConvEncoder(
        input_dims=config['model']['input_dims'],
        output_dims=config['model']['output_dims'],
        hidden_dims=config['model']['hidden_dims'],
        depth=config['model']['depth']
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = PatternFocusedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger
    )
    
    # 开始训练
    logger.info("\n" + "="*50)
    logger.info("开始训练...")
    logger.info("="*50 + "\n")
    
    trainer.train()
    
    logger.info("\n训练完成！")
    logger.info(f"最佳模型保存在: {trainer.checkpoint_dir}/best_model.pt")
    logger.info(f"最终模型保存在: {trainer.checkpoint_dir}/final_model.pt")
    
    # 训练后分析
    logger.info("\n建议下一步:")
    logger.info("1. 运行 python training/analyze_ts2vec_patterns.py 分析学到的形态")
    logger.info("2. 使用聚类查看模型自动发现的形态类别")
    logger.info("3. 与手动形态检测对比，验证模型学习效果")


if __name__ == '__main__':
    main()