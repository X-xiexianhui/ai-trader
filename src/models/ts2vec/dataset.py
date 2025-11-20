"""
TS2Vec数据集和数据加载器

实现滑动窗口数据生成器，用于TS2Vec训练。

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
import logging

from .augmentation import TimeSeriesAugmentation

logger = logging.getLogger(__name__)


class SlidingWindowDataset(Dataset):
    """
    滑动窗口数据集
    
    使用滑动窗口从时间序列中采样固定长度的子序列。
    """
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 256,
        stride: int = 1,
        augmenter: Optional[TimeSeriesAugmentation] = None
    ):
        """
        初始化滑动窗口数据集
        
        Args:
            data: 时间序列数据 [total_len, features]
            window_size: 窗口大小
            stride: 滑动步长
            augmenter: 数据增强器
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.augmenter = augmenter
        
        # 计算可以生成的窗口数量
        self.num_windows = (len(data) - window_size) // stride + 1
        
        logger.info(f"SlidingWindowDataset initialized: "
                   f"data_len={len(data)}, window_size={window_size}, "
                   f"stride={stride}, num_windows={self.num_windows}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_windows
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取一个窗口的数据
        
        Args:
            idx: 窗口索引
            
        Returns:
            窗口数据 [window_size, features]
        """
        # 计算窗口起始位置
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        # 提取窗口数据
        window = self.data[start_idx:end_idx]
        
        # 转换为tensor
        window = torch.FloatTensor(window)
        
        return window


class ContrastiveDataset(Dataset):
    """
    对比学习数据集
    
    为每个样本生成两个增强视图，用于对比学习。
    """
    
    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 256,
        stride: int = 1,
        augmenter: Optional[TimeSeriesAugmentation] = None
    ):
        """
        初始化对比学习数据集
        
        Args:
            data: 时间序列数据 [total_len, features]
            window_size: 窗口大小
            stride: 滑动步长
            augmenter: 数据增强器
        """
        self.base_dataset = SlidingWindowDataset(
            data=data,
            window_size=window_size,
            stride=stride
        )
        
        self.augmenter = augmenter
        if self.augmenter is None:
            self.augmenter = TimeSeriesAugmentation()
        
        logger.info(f"ContrastiveDataset initialized with {len(self.base_dataset)} windows")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本的两个增强视图
        
        Args:
            idx: 样本索引
            
        Returns:
            (view1, view2) 两个增强视图
        """
        # 获取原始窗口
        window = self.base_dataset[idx]
        
        # 添加batch维度用于增强
        window = window.unsqueeze(0)
        
        # 生成两个增强视图
        view1, view2 = self.augmenter.create_positive_pair(window)
        
        # 移除batch维度
        view1 = view1.squeeze(0)
        view2 = view2.squeeze(0)
        
        return view1, view2


class MultiScaleDataset(Dataset):
    """
    多尺度数据集
    
    同时提供多个时间尺度的窗口。
    """
    
    def __init__(
        self,
        data: np.ndarray,
        window_sizes: List[int] = [128, 256, 512],
        stride: int = 1
    ):
        """
        初始化多尺度数据集
        
        Args:
            data: 时间序列数据 [total_len, features]
            window_sizes: 多个窗口大小
            stride: 滑动步长
        """
        self.data = data
        self.window_sizes = window_sizes
        self.stride = stride
        
        # 使用最大窗口大小计算数量
        max_window = max(window_sizes)
        self.num_windows = (len(data) - max_window) // stride + 1
        
        logger.info(f"MultiScaleDataset initialized: "
                   f"window_sizes={window_sizes}, num_windows={self.num_windows}")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_windows
    
    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """
        获取多个尺度的窗口
        
        Args:
            idx: 窗口索引
            
        Returns:
            多个尺度的窗口列表
        """
        windows = []
        
        for window_size in self.window_sizes:
            start_idx = idx * self.stride
            end_idx = start_idx + window_size
            
            window = self.data[start_idx:end_idx]
            window = torch.FloatTensor(window)
            windows.append(window)
        
        return windows


def create_ts2vec_dataloader(
    data: np.ndarray,
    window_size: int = 256,
    stride: int = 1,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    augmenter: Optional[TimeSeriesAugmentation] = None
) -> DataLoader:
    """
    创建TS2Vec数据加载器
    
    Args:
        data: 时间序列数据 [total_len, features]
        window_size: 窗口大小
        stride: 滑动步长
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        augmenter: 数据增强器
        
    Returns:
        DataLoader
    """
    dataset = ContrastiveDataset(
        data=data,
        window_size=window_size,
        stride=stride,
        augmenter=augmenter
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"DataLoader created: "
               f"dataset_size={len(dataset)}, "
               f"batch_size={batch_size}, "
               f"num_batches={len(dataloader)}")
    
    return dataloader


def prepare_data_from_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从DataFrame准备训练、验证和测试数据
    
    Args:
        df: 包含特征的DataFrame
        feature_columns: 特征列名列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (train_data, val_data, test_data)
    """
    # 提取特征
    data = df[feature_columns].values
    
    # 计算分割点
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # 分割数据
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    logger.info(f"Data prepared: "
               f"train={len(train_data)}, "
               f"val={len(val_data)}, "
               f"test={len(test_data)}")
    
    return train_data, val_data, test_data


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建模拟数据
    total_len = 10000
    features = 27
    data = np.random.randn(total_len, features)
    
    print("\n=== 滑动窗口数据集示例 ===")
    print(f"原始数据形状: {data.shape}")
    
    # 测试滑动窗口数据集
    print("\n1. 滑动窗口数据集:")
    window_dataset = SlidingWindowDataset(
        data=data,
        window_size=256,
        stride=1
    )
    print(f"   数据集大小: {len(window_dataset)}")
    print(f"   第一个窗口形状: {window_dataset[0].shape}")
    
    # 测试对比学习数据集
    print("\n2. 对比学习数据集:")
    augmenter = TimeSeriesAugmentation()
    contrastive_dataset = ContrastiveDataset(
        data=data,
        window_size=256,
        stride=128,
        augmenter=augmenter
    )
    print(f"   数据集大小: {len(contrastive_dataset)}")
    view1, view2 = contrastive_dataset[0]
    print(f"   View1形状: {view1.shape}")
    print(f"   View2形状: {view2.shape}")
    
    # 测试数据加载器
    print("\n3. 数据加载器:")
    dataloader = create_ts2vec_dataloader(
        data=data,
        window_size=256,
        stride=128,
        batch_size=32,
        shuffle=True,
        num_workers=0  # 使用0避免多进程问题
    )
    
    # 获取一个批次
    batch_view1, batch_view2 = next(iter(dataloader))
    print(f"   批次View1形状: {batch_view1.shape}")
    print(f"   批次View2形状: {batch_view2.shape}")
    
    # 测试多尺度数据集
    print("\n4. 多尺度数据集:")
    multiscale_dataset = MultiScaleDataset(
        data=data,
        window_sizes=[128, 256, 512],
        stride=256
    )
    print(f"   数据集大小: {len(multiscale_dataset)}")
    windows = multiscale_dataset[0]
    print(f"   窗口数量: {len(windows)}")
    for i, window in enumerate(windows):
        print(f"   窗口{i}形状: {window.shape}")
    
    # 测试从DataFrame准备数据
    print("\n5. 从DataFrame准备数据:")
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(features)])
    feature_cols = df.columns.tolist()
    
    train_data, val_data, test_data = prepare_data_from_dataframe(
        df=df,
        feature_columns=feature_cols,
        train_ratio=0.7,
        val_ratio=0.15
    )
    print(f"   训练集形状: {train_data.shape}")
    print(f"   验证集形状: {val_data.shape}")
    print(f"   测试集形状: {test_data.shape}")