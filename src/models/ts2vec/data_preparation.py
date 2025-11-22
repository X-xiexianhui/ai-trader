"""
TS2Vec数据准备模块

任务2.1.1-2.1.5实现:
1. 滑动窗口数据生成器
2. 时间遮蔽数据增强
3. 时间扭曲数据增强
4. 幅度缩放数据增强
5. 对比学习样本对生成器
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
from scipy import interpolate
import logging

logger = logging.getLogger(__name__)


class SlidingWindowGenerator:
    """
    任务2.1.1: 滑动窗口数据生成器
    
    为TS2Vec训练准备数据,生成固定长度的时间序列窗口
    """
    
    def __init__(self, window_length: int = 256, stride: int = 1):
        """
        初始化滑动窗口生成器
        
        Args:
            window_length: 窗口长度
            stride: 滑动步长
        """
        self.window_length = window_length
        self.stride = stride
        
    def generate_windows(self,
                        data: np.ndarray,
                        normalize_per_window: bool = True) -> np.ndarray:
        """
        生成滑动窗口
        
        Args:
            data: 输入数据 [N, D]，N是时间步，D是特征维度
            normalize_per_window: 是否对每个窗口进行归一化
            
        Returns:
            窗口数组 [num_windows, window_length, D]
        """
        # 维度检查和转换
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError(f"数据必须是1D或2D数组，当前维度: {data.ndim}")
        
        if len(data) < self.window_length:
            raise ValueError(f"数据长度{len(data)}小于窗口长度{self.window_length}")
        
        # 计算窗口数量
        num_windows = (len(data) - self.window_length) // self.stride + 1
        
        windows = []
        for i in range(num_windows):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_length
            window = data[start_idx:end_idx].copy()
            
            # 窗口内归一化（z-score）
            if normalize_per_window:
                mean = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True)
                # 改进的除零保护：使用小值而不是1.0
                std = np.maximum(std, 1e-8)
                window = (window - mean) / std
            
            windows.append(window)
        
        windows = np.array(windows)
        logger.info(f"生成{num_windows}个窗口，形状: {windows.shape}")
        
        return windows
    
    def generate_from_dataframe(self,
                               df: pd.DataFrame,
                               columns: Optional[List[str]] = None) -> np.ndarray:
        """
        从DataFrame生成窗口
        
        Args:
            df: 输入DataFrame
            columns: 要使用的列名，None表示使用所有列
            
        Returns:
            窗口数组
        """
        if columns is None:
            data = df.values
        else:
            data = df[columns].values
        
        return self.generate_windows(data)


class TimeSeriesAugmentation:
    """
    时间序列数据增强
    
    任务2.1.2-2.1.4实现:
    - 时间遮蔽
    - 时间扭曲
    - 幅度缩放
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        初始化数据增强器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
    
    def temporal_masking(self,
                        x: np.ndarray,
                        masking_ratio: float = 0.2,
                        mask_value: float = 0.0) -> np.ndarray:
        """
        任务2.1.2: 时间遮蔽数据增强
        
        随机遮蔽时间步
        
        Args:
            x: 输入窗口 [T, D]
            masking_ratio: 遮蔽比例（0.1-0.3）
            mask_value: 遮蔽值（0或'mean'）
            
        Returns:
            增强后的窗口
        """
        x_aug = x.copy()
        T = len(x)
        
        # 验证masking_ratio
        if not 0 < masking_ratio < 1:
            raise ValueError(f"masking_ratio必须在(0,1)之间: {masking_ratio}")
        
        # 计算要遮蔽的时间步数量
        num_mask = int(T * masking_ratio)
        if num_mask == 0:
            return x_aug
        
        # 随机选择要遮蔽的时间步
        mask_indices = np.random.choice(T, size=num_mask, replace=False)
        
        # 改进的mask_value处理
        if isinstance(mask_value, str) and mask_value == 'mean':
            x_aug[mask_indices] = x.mean(axis=0)
        elif isinstance(mask_value, (int, float)):
            x_aug[mask_indices] = mask_value
        else:
            raise ValueError(f"mask_value必须是数值或'mean'，当前: {mask_value}")
        
        return x_aug
    
    def time_warping(self,
                    x: np.ndarray,
                    warp_ratio: float = 0.05) -> np.ndarray:
        """
        任务2.1.3: 时间扭曲数据增强
        
        轻微拉伸/压缩时间轴
        
        Args:
            x: 输入窗口 [T, D]
            warp_ratio: 扭曲比例（±5%）
            
        Returns:
            增强后的窗口
        """
        T, D = x.shape
        
        # 限制warp_ratio范围，防止过度扭曲
        warp_ratio = min(abs(warp_ratio), 0.2)
        warp_factor = 1.0 + np.random.uniform(-warp_ratio, warp_ratio)
        new_T = max(int(T * warp_factor), T // 2)  # 至少保留一半长度
        
        # 使用插值进行时间扭曲
        x_aug = np.zeros((T, D))
        
        for d in range(D):
            # 原始时间点
            old_indices = np.arange(T)
            # 新时间点
            new_indices = np.linspace(0, T-1, new_T)
            
            # 使用边界值填充而不是外推，防止产生不合理的值
            f = interpolate.interp1d(
                old_indices, x[:, d],
                kind='linear',
                bounds_error=False,
                fill_value=(x[0, d], x[-1, d])
            )
            warped = f(new_indices)
            
            # 重采样回原始长度
            resample_indices = np.linspace(0, new_T-1, T)
            f2 = interpolate.interp1d(
                np.arange(new_T), warped,
                kind='linear',
                bounds_error=False,
                fill_value=(warped[0], warped[-1])
            )
            x_aug[:, d] = f2(resample_indices)
        
        return x_aug
    
    def magnitude_scaling(self,
                         x: np.ndarray,
                         scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        任务2.1.4: 幅度缩放数据增强
        
        对所有特征统一缩放
        
        Args:
            x: 输入窗口 [T, D]
            scale_range: 缩放因子范围
            
        Returns:
            增强后的窗口
        """
        # 验证scale_range
        if scale_range[0] >= scale_range[1]:
            raise ValueError(f"scale_range[0]必须小于scale_range[1]: {scale_range}")
        if scale_range[0] <= 0:
            raise ValueError(f"scale_range必须为正值: {scale_range}")
        
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        x_aug = x * scale_factor
        
        return x_aug
    
    def random_augment(self,
                      x: np.ndarray,
                      aug_types: List[str] = ['masking', 'warping', 'scaling'],
                      **kwargs) -> np.ndarray:
        """
        随机应用一种或多种增强
        
        Args:
            x: 输入窗口
            aug_types: 要应用的增强类型列表
            **kwargs: 增强参数
            
        Returns:
            增强后的窗口
        """
        x_aug = x.copy()
        
        # 随机选择增强方法
        selected_aug = np.random.choice(aug_types)
        
        if selected_aug == 'masking':
            masking_ratio = kwargs.get('masking_ratio', 0.2)
            x_aug = self.temporal_masking(x_aug, masking_ratio)
        elif selected_aug == 'warping':
            warp_ratio = kwargs.get('warp_ratio', 0.05)
            x_aug = self.time_warping(x_aug, warp_ratio)
        elif selected_aug == 'scaling':
            scale_range = kwargs.get('scale_range', (0.9, 1.1))
            x_aug = self.magnitude_scaling(x_aug, scale_range)
        
        return x_aug


class ContrastivePairGenerator(Dataset):
    """
    任务2.1.5: 对比学习样本对生成器
    
    为TS2Vec生成正负样本对
    """
    
    def __init__(self,
                 windows: np.ndarray,
                 augmentation: TimeSeriesAugmentation,
                 aug_params: Optional[dict] = None):
        """
        初始化对比样本对生成器
        
        Args:
            windows: 窗口数组 [N, T, D]
            augmentation: 数据增强器
            aug_params: 增强参数
        """
        self.windows = windows
        self.augmentation = augmentation
        self.aug_params = aug_params or {}
        
        logger.info(f"对比样本对生成器初始化: {len(windows)}个窗口")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本对
        
        Returns:
            (anchor, positive): 正样本对，来自同一原始窗口的两次不同增强
        """
        # 获取原始窗口
        window = self.windows[idx]
        
        # 生成两个不同的增强视图
        anchor = self.augmentation.random_augment(window, **self.aug_params)
        positive = self.augmentation.random_augment(window, **self.aug_params)
        
        # 转换为tensor
        anchor = torch.FloatTensor(anchor)
        positive = torch.FloatTensor(positive)
        
        return anchor, positive
    
    def get_dataloader(self,
                      batch_size: int = 64,
                      shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
        """
        创建DataLoader
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            
        Returns:
            DataLoader对象
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )


class TS2VecDataset(Dataset):
    """
    TS2Vec完整数据集
    
    整合窗口生成、数据增强和样本对生成
    """
    
    def __init__(self,
                 data: np.ndarray,
                 window_length: int = 256,
                 stride: int = 1,
                 augmentation_params: Optional[dict] = None):
        """
        初始化TS2Vec数据集
        
        Args:
            data: 原始时间序列数据 [N, D]
            window_length: 窗口长度
            stride: 滑动步长
            augmentation_params: 数据增强参数
        """
        # 生成窗口
        window_gen = SlidingWindowGenerator(window_length, stride)
        self.windows = window_gen.generate_windows(data)
        
        # 初始化数据增强
        self.augmentation = TimeSeriesAugmentation()
        self.aug_params = augmentation_params or {
            'aug_types': ['masking', 'warping', 'scaling'],
            'masking_ratio': 0.2,
            'warp_ratio': 0.05,
            'scale_range': (0.9, 1.1)
        }
        
        logger.info(f"TS2Vec数据集初始化: {len(self.windows)}个窗口")
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取一个对比样本对"""
        window = self.windows[idx]
        
        # 生成两个增强视图
        anchor = self.augmentation.random_augment(window, **self.aug_params)
        positive = self.augmentation.random_augment(window, **self.aug_params)
        
        return torch.FloatTensor(anchor), torch.FloatTensor(positive)
    
    @classmethod
    def from_dataframe(cls,
                      df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      **kwargs) -> 'TS2VecDataset':
        """
        从DataFrame创建数据集
        
        Args:
            df: 输入DataFrame
            columns: 要使用的列
            **kwargs: 其他参数
            
        Returns:
            TS2VecDataset实例
        """
        if columns is None:
            data = df.values
        else:
            data = df[columns].values
        
        return cls(data, **kwargs)