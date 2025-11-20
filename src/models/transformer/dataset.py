"""
Transformer数据集实现

实现用于Transformer训练的数据集，包括:
- 特征融合数据集
- 序列采样
- 辅助任务标签生成
- 数据加载器

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TransformerDataset(Dataset):
    """
    Transformer训练数据集
    
    融合TS2Vec embedding和手工特征，生成序列样本和辅助任务标签。
    """
    
    def __init__(
        self,
        ts2vec_embeddings: np.ndarray,
        manual_features: np.ndarray,
        prices: np.ndarray,
        seq_len: int = 64,
        pred_horizon: int = 1,
        stride: int = 1,
        return_threshold: float = 0.001
    ):
        """
        初始化数据集
        
        Args:
            ts2vec_embeddings: TS2Vec embeddings (n_samples, embedding_dim)
            manual_features: 手工特征 (n_samples, feature_dim)
            prices: 价格序列 (n_samples,)
            seq_len: 序列长度
            pred_horizon: 预测时间跨度
            stride: 滑动步长
            return_threshold: 收益率阈值（用于分类标签）
        """
        super().__init__()
        
        self.ts2vec_embeddings = torch.FloatTensor(ts2vec_embeddings)
        self.manual_features = torch.FloatTensor(manual_features)
        self.prices = torch.FloatTensor(prices)
        
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.stride = stride
        self.return_threshold = return_threshold
        
        # 计算有效样本数量
        self.n_samples = len(prices)
        self.valid_indices = self._get_valid_indices()
        
        logger.info(f"TransformerDataset initialized: "
                   f"n_samples={len(self.valid_indices)}, "
                   f"seq_len={seq_len}, pred_horizon={pred_horizon}")
    
    def _get_valid_indices(self) -> List[int]:
        """获取有效的起始索引"""
        valid_indices = []
        for i in range(0, self.n_samples - self.seq_len - self.pred_horizon + 1, self.stride):
            valid_indices.append(i)
        return valid_indices
    
    def _compute_return(self, start_idx: int, end_idx: int) -> float:
        """
        计算收益率
        
        Args:
            start_idx: 起始索引
            end_idx: 结束索引
            
        Returns:
            对数收益率
        """
        start_price = self.prices[start_idx]
        end_price = self.prices[end_idx]
        return torch.log(end_price / start_price).item()
    
    def _get_direction_label(self, return_value: float) -> int:
        """
        获取方向标签
        
        Args:
            return_value: 收益率
            
        Returns:
            标签: 0=下跌, 1=持平, 2=上涨
        """
        if return_value < -self.return_threshold:
            return 0  # 下跌
        elif return_value > self.return_threshold:
            return 2  # 上涨
        else:
            return 1  # 持平
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            样本字典，包含:
            - ts2vec_emb: TS2Vec embedding序列
            - manual_features: 手工特征序列
            - regression_target: 回归目标（未来收益率）
            - classification_target: 分类目标（价格方向）
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        pred_idx = end_idx + self.pred_horizon - 1
        
        # 提取序列
        ts2vec_emb = self.ts2vec_embeddings[start_idx:end_idx]
        manual_features = self.manual_features[start_idx:end_idx]
        
        # 计算辅助任务标签（对序列中每个时间步）
        regression_targets = []
        classification_targets = []
        
        for i in range(start_idx, end_idx):
            future_idx = min(i + self.pred_horizon, self.n_samples - 1)
            ret = self._compute_return(i, future_idx)
            direction = self._get_direction_label(ret)
            
            regression_targets.append(ret)
            classification_targets.append(direction)
        
        return {
            'ts2vec_emb': ts2vec_emb,
            'manual_features': manual_features,
            'regression_target': torch.FloatTensor(regression_targets).unsqueeze(-1),
            'classification_target': torch.LongTensor(classification_targets)
        }


class TransformerDataModule:
    """
    Transformer数据模块
    
    管理训练、验证和测试数据集。
    """
    
    def __init__(
        self,
        train_ts2vec: np.ndarray,
        train_features: np.ndarray,
        train_prices: np.ndarray,
        val_ts2vec: Optional[np.ndarray] = None,
        val_features: Optional[np.ndarray] = None,
        val_prices: Optional[np.ndarray] = None,
        test_ts2vec: Optional[np.ndarray] = None,
        test_features: Optional[np.ndarray] = None,
        test_prices: Optional[np.ndarray] = None,
        seq_len: int = 64,
        pred_horizon: int = 1,
        stride: int = 1,
        return_threshold: float = 0.001,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        初始化数据模块
        
        Args:
            train_ts2vec: 训练集TS2Vec embeddings
            train_features: 训练集手工特征
            train_prices: 训练集价格
            val_ts2vec: 验证集TS2Vec embeddings
            val_features: 验证集手工特征
            val_prices: 验证集价格
            test_ts2vec: 测试集TS2Vec embeddings
            test_features: 测试集手工特征
            test_prices: 测试集价格
            seq_len: 序列长度
            pred_horizon: 预测时间跨度
            stride: 滑动步长
            return_threshold: 收益率阈值
            batch_size: 批次大小
            num_workers: 数据加载线程数
        """
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.stride = stride
        self.return_threshold = return_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 创建训练集
        self.train_dataset = TransformerDataset(
            train_ts2vec, train_features, train_prices,
            seq_len, pred_horizon, stride, return_threshold
        )
        
        # 创建验证集
        self.val_dataset = None
        if val_ts2vec is not None:
            self.val_dataset = TransformerDataset(
                val_ts2vec, val_features, val_prices,
                seq_len, pred_horizon, stride, return_threshold
            )
        
        # 创建测试集
        self.test_dataset = None
        if test_ts2vec is not None:
            self.test_dataset = TransformerDataset(
                test_ts2vec, test_features, test_prices,
                seq_len, pred_horizon, stride, return_threshold
            )
        
        logger.info(f"TransformerDataModule initialized: "
                   f"train_size={len(self.train_dataset)}, "
                   f"val_size={len(self.val_dataset) if self.val_dataset else 0}, "
                   f"test_size={len(self.test_dataset) if self.test_dataset else 0}")
    
    def train_dataloader(self) -> DataLoader:
        """获取训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """获取验证数据加载器"""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """获取测试数据加载器"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """获取数据集统计信息"""
        stats = {
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset) if self.val_dataset else 0,
            'test_size': len(self.test_dataset) if self.test_dataset else 0,
            'seq_len': self.seq_len,
            'pred_horizon': self.pred_horizon,
            'batch_size': self.batch_size
        }
        
        # 计算标签分布（训练集）
        if len(self.train_dataset) > 0:
            sample = self.train_dataset[0]
            stats['ts2vec_dim'] = sample['ts2vec_emb'].shape[-1]
            stats['feature_dim'] = sample['manual_features'].shape[-1]
            
            # 统计分类标签分布
            all_labels = []
            for i in range(min(1000, len(self.train_dataset))):
                sample = self.train_dataset[i]
                all_labels.extend(sample['classification_target'].tolist())
            
            unique, counts = np.unique(all_labels, return_counts=True)
            stats['label_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
        
        return stats


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义collate函数
    
    Args:
        batch: 批次样本列表
        
    Returns:
        批次字典
    """
    return {
        'ts2vec_emb': torch.stack([item['ts2vec_emb'] for item in batch]),
        'manual_features': torch.stack([item['manual_features'] for item in batch]),
        'regression_target': torch.stack([item['regression_target'] for item in batch]),
        'classification_target': torch.stack([item['classification_target'] for item in batch])
    }


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Transformer数据集示例 ===")
    
    # 创建模拟数据
    n_samples = 1000
    ts2vec_dim = 128
    feature_dim = 27
    
    train_ts2vec = np.random.randn(n_samples, ts2vec_dim)
    train_features = np.random.randn(n_samples, feature_dim)
    train_prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    val_ts2vec = np.random.randn(200, ts2vec_dim)
    val_features = np.random.randn(200, feature_dim)
    val_prices = np.cumsum(np.random.randn(200) * 0.01) + 100
    
    # 创建数据模块
    data_module = TransformerDataModule(
        train_ts2vec=train_ts2vec,
        train_features=train_features,
        train_prices=train_prices,
        val_ts2vec=val_ts2vec,
        val_features=val_features,
        val_prices=val_prices,
        seq_len=64,
        pred_horizon=1,
        batch_size=32
    )
    
    # 获取统计信息
    stats = data_module.get_statistics()
    print("\n数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试数据加载
    train_loader = data_module.train_dataloader()
    print(f"\n训练批次数: {len(train_loader)}")
    
    # 获取一个批次
    batch = next(iter(train_loader))
    print("\n批次形状:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print("\n示例完成!")