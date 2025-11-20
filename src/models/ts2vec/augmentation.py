"""
TS2Vec数据增强模块

实现时序数据增强策略:
- 时间遮蔽 (Temporal Masking)
- 时间扭曲 (Time Warping)
- 幅度缩放 (Magnitude Scaling)
- 时间平移 (Time Shifting)

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TimeSeriesAugmentation:
    """
    时序数据增强器
    
    提供多种数据增强方法，用于对比学习。
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.15,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        shift_range: int = 10,
        warp_strength: float = 0.1
    ):
        """
        初始化数据增强器
        
        Args:
            mask_ratio: 遮蔽比例
            scale_range: 缩放范围 (min, max)
            shift_range: 平移范围 (±steps)
            warp_strength: 扭曲强度
        """
        self.mask_ratio = mask_ratio
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.warp_strength = warp_strength
        
        logger.info(f"TimeSeriesAugmentation initialized: "
                   f"mask_ratio={mask_ratio}, scale_range={scale_range}")
    
    def temporal_masking(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        时间遮蔽
        
        随机遮蔽时间序列中的部分时间步。
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            mask_ratio: 遮蔽比例，None则使用默认值
            
        Returns:
            遮蔽后的张量
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        batch_size, seq_len, features = x.shape
        
        # 创建遮蔽掩码
        mask = torch.rand(batch_size, seq_len, 1, device=x.device) > mask_ratio
        
        # 应用遮蔽（遮蔽位置设为0）
        x_masked = x * mask.float()
        
        return x_masked
    
    def magnitude_scaling(
        self,
        x: torch.Tensor,
        scale_range: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        幅度缩放
        
        对时间序列的幅度进行随机缩放。
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            scale_range: 缩放范围，None则使用默认值
            
        Returns:
            缩放后的张量
        """
        if scale_range is None:
            scale_range = self.scale_range
        
        batch_size = x.shape[0]
        
        # 为每个样本生成随机缩放因子
        scale = torch.FloatTensor(batch_size, 1, 1).uniform_(
            scale_range[0],
            scale_range[1]
        ).to(x.device)
        
        # 应用缩放
        x_scaled = x * scale
        
        return x_scaled
    
    def time_shifting(
        self,
        x: torch.Tensor,
        shift_range: Optional[int] = None
    ) -> torch.Tensor:
        """
        时间平移
        
        在时间维度上随机平移序列。
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            shift_range: 平移范围，None则使用默认值
            
        Returns:
            平移后的张量
        """
        if shift_range is None:
            shift_range = self.shift_range
        
        batch_size, seq_len, features = x.shape
        
        # 为每个样本生成随机平移量
        shifts = torch.randint(
            -shift_range,
            shift_range + 1,
            (batch_size,),
            device=x.device
        )
        
        # 应用平移
        x_shifted = torch.zeros_like(x)
        for i in range(batch_size):
            shift = shifts[i].item()
            if shift > 0:
                x_shifted[i, shift:] = x[i, :-shift]
            elif shift < 0:
                x_shifted[i, :shift] = x[i, -shift:]
            else:
                x_shifted[i] = x[i]
        
        return x_shifted
    
    def time_warping(
        self,
        x: torch.Tensor,
        warp_strength: Optional[float] = None
    ) -> torch.Tensor:
        """
        时间扭曲
        
        对时间轴进行非线性扭曲。
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            warp_strength: 扭曲强度，None则使用默认值
            
        Returns:
            扭曲后的张量
        """
        if warp_strength is None:
            warp_strength = self.warp_strength
        
        batch_size, seq_len, features = x.shape
        
        # 生成扭曲的时间索引
        # 使用平滑的随机扰动
        time_steps = torch.linspace(0, seq_len - 1, seq_len, device=x.device)
        
        x_warped = torch.zeros_like(x)
        
        for i in range(batch_size):
            # 生成随机扭曲
            warp = torch.randn(seq_len, device=x.device) * warp_strength * seq_len
            
            # 平滑扭曲（使用卷积）
            warp = torch.nn.functional.conv1d(
                warp.unsqueeze(0).unsqueeze(0),
                torch.ones(1, 1, 5, device=x.device) / 5,
                padding=2
            ).squeeze()
            
            # 应用扭曲
            warped_indices = (time_steps + warp).clamp(0, seq_len - 1)
            
            # 使用线性插值
            indices_floor = warped_indices.long()
            indices_ceil = (indices_floor + 1).clamp(max=seq_len - 1)
            weight = warped_indices - indices_floor.float()
            
            x_warped[i] = (
                x[i, indices_floor] * (1 - weight).unsqueeze(-1) +
                x[i, indices_ceil] * weight.unsqueeze(-1)
            )
        
        return x_warped
    
    def random_augment(
        self,
        x: torch.Tensor,
        num_augmentations: int = 2
    ) -> torch.Tensor:
        """
        随机应用多种增强
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            num_augmentations: 应用的增强数量
            
        Returns:
            增强后的张量
        """
        augmentations = [
            self.temporal_masking,
            self.magnitude_scaling,
            self.time_shifting,
            self.time_warping
        ]
        
        # 随机选择增强方法
        selected_augs = np.random.choice(
            len(augmentations),
            size=min(num_augmentations, len(augmentations)),
            replace=False
        )
        
        x_aug = x.clone()
        for aug_idx in selected_augs:
            x_aug = augmentations[aug_idx](x_aug)
        
        return x_aug
    
    def create_positive_pair(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        创建正样本对
        
        对同一输入应用两次不同的增强，创建正样本对。
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            
        Returns:
            (view1, view2) 两个增强视图
        """
        view1 = self.random_augment(x)
        view2 = self.random_augment(x)
        
        return view1, view2


class SubsampleAugmentation:
    """
    子采样增强
    
    通过子采样创建不同的视图。
    """
    
    def __init__(
        self,
        subsample_ratio: float = 0.5
    ):
        """
        初始化子采样增强
        
        Args:
            subsample_ratio: 子采样比例
        """
        self.subsample_ratio = subsample_ratio
    
    def subsample(
        self,
        x: torch.Tensor,
        ratio: Optional[float] = None
    ) -> torch.Tensor:
        """
        随机子采样
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            ratio: 子采样比例
            
        Returns:
            子采样后的张量
        """
        if ratio is None:
            ratio = self.subsample_ratio
        
        batch_size, seq_len, features = x.shape
        subsample_len = int(seq_len * ratio)
        
        # 随机选择起始位置
        start_indices = torch.randint(
            0,
            seq_len - subsample_len + 1,
            (batch_size,),
            device=x.device
        )
        
        # 提取子序列
        x_sub = torch.zeros(batch_size, subsample_len, features, device=x.device)
        for i in range(batch_size):
            start = start_indices[i].item()
            x_sub[i] = x[i, start:start + subsample_len]
        
        return x_sub


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建增强器
    augmenter = TimeSeriesAugmentation(
        mask_ratio=0.15,
        scale_range=(0.8, 1.2),
        shift_range=10,
        warp_strength=0.1
    )
    
    # 创建测试数据
    batch_size = 4
    seq_len = 256
    features = 27
    
    x = torch.randn(batch_size, seq_len, features)
    
    print("\n=== 时序数据增强示例 ===")
    print(f"原始数据形状: {x.shape}")
    print(f"原始数据统计: mean={x.mean():.4f}, std={x.std():.4f}")
    
    # 测试各种增强
    print("\n1. 时间遮蔽:")
    x_masked = augmenter.temporal_masking(x)
    print(f"   形状: {x_masked.shape}")
    print(f"   统计: mean={x_masked.mean():.4f}, std={x_masked.std():.4f}")
    print(f"   零值比例: {(x_masked == 0).float().mean():.2%}")
    
    print("\n2. 幅度缩放:")
    x_scaled = augmenter.magnitude_scaling(x)
    print(f"   形状: {x_scaled.shape}")
    print(f"   统计: mean={x_scaled.mean():.4f}, std={x_scaled.std():.4f}")
    
    print("\n3. 时间平移:")
    x_shifted = augmenter.time_shifting(x)
    print(f"   形状: {x_shifted.shape}")
    print(f"   统计: mean={x_shifted.mean():.4f}, std={x_shifted.std():.4f}")
    
    print("\n4. 时间扭曲:")
    x_warped = augmenter.time_warping(x)
    print(f"   形状: {x_warped.shape}")
    print(f"   统计: mean={x_warped.mean():.4f}, std={x_warped.std():.4f}")
    
    print("\n5. 随机组合增强:")
    x_aug = augmenter.random_augment(x, num_augmentations=2)
    print(f"   形状: {x_aug.shape}")
    print(f"   统计: mean={x_aug.mean():.4f}, std={x_aug.std():.4f}")
    
    print("\n6. 创建正样本对:")
    view1, view2 = augmenter.create_positive_pair(x)
    print(f"   View1形状: {view1.shape}")
    print(f"   View2形状: {view2.shape}")
    print(f"   View1统计: mean={view1.mean():.4f}, std={view1.std():.4f}")
    print(f"   View2统计: mean={view2.mean():.4f}, std={view2.std():.4f}")
    
    # 测试子采样
    print("\n7. 子采样:")
    subsampler = SubsampleAugmentation(subsample_ratio=0.5)
    x_sub = subsampler.subsample(x)
    print(f"   形状: {x_sub.shape}")
    print(f"   统计: mean={x_sub.mean():.4f}, std={x_sub.std():.4f}")