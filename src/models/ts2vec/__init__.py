"""
TS2Vec模块

时间序列到向量的自监督表示学习模型。

主要组件:
- TS2Vec: 核心模型架构
- TimeSeriesAugmentation: 数据增强
- TS2VecLoss: 对比学习损失
- SlidingWindowDataset: 数据集
- TS2VecTrainer: 训练器
- TS2VecEvaluator: 评估器
- EmbeddingExtractor: Embedding提取器

Author: AI Trader Team
Date: 2025-11-20
"""

from .model import (
    DilatedCNNEncoder,
    ProjectionHead,
    TS2Vec
)

from .augmentation import TimeSeriesAugmentation

from .loss import (
    NTXentLoss,
    HierarchicalContrastiveLoss,
    TS2VecLoss
)

from .dataset import (
    SlidingWindowDataset,
    ContrastiveDataset,
    MultiScaleDataset,
    create_ts2vec_dataloader
)

from .trainer import (
    LearningRateScheduler,
    EarlyStopping,
    TS2VecTrainer
)

from .evaluator import (
    TS2VecEvaluator,
    EmbeddingExtractor
)

__all__ = [
    # 模型
    'DilatedCNNEncoder',
    'ProjectionHead',
    'TS2Vec',
    
    # 数据增强
    'TimeSeriesAugmentation',
    
    # 损失函数
    'NTXentLoss',
    'HierarchicalContrastiveLoss',
    'TS2VecLoss',
    
    # 数据集
    'SlidingWindowDataset',
    'ContrastiveDataset',
    'MultiScaleDataset',
    'create_ts2vec_dataloader',
    
    # 训练
    'LearningRateScheduler',
    'EarlyStopping',
    'TS2VecTrainer',
    
    # 评估
    'TS2VecEvaluator',
    'EmbeddingExtractor',
]

__version__ = '1.0.0'