"""
Transformer模块

用于时间序列状态表征学习的Transformer模型。

主要组件:
- TransformerModel: 完整的Transformer模型
- PositionalEncoding: 位置编码
- MultiHeadAttention: 多头自注意力
- FeatureFusion: 特征融合层
- AuxiliaryHeads: 辅助任务头
- TransformerDataset: 数据集
- TransformerDataModule: 数据模块
- TransformerTrainer: 训练器
- TransformerEvaluator: 评估器
- StateVectorExtractor: 状态向量提取器

Author: AI Trader Team
Date: 2025-11-20
"""

from .model import (
    TransformerModel,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    FeatureFusion,
    AuxiliaryHeads
)

from .dataset import (
    TransformerDataset,
    TransformerDataModule,
    collate_fn
)

from .trainer import (
    TransformerTrainer,
    WarmupCosineScheduler,
    EarlyStopping
)

from .evaluator import (
    TransformerEvaluator,
    StateVectorExtractor
)

__all__ = [
    # 模型组件
    'TransformerModel',
    'PositionalEncoding',
    'MultiHeadAttention',
    'FeedForward',
    'TransformerEncoderLayer',
    'FeatureFusion',
    'AuxiliaryHeads',
    
    # 数据组件
    'TransformerDataset',
    'TransformerDataModule',
    'collate_fn',
    
    # 训练组件
    'TransformerTrainer',
    'WarmupCosineScheduler',
    'EarlyStopping',
    
    # 评估组件
    'TransformerEvaluator',
    'StateVectorExtractor',
]

__version__ = '1.0.0'