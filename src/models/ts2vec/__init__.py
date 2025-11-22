"""
TS2Vec模块 - 时间序列对比学习编码器
"""

from .data_preparation import (
    SlidingWindowGenerator,
    TimeSeriesAugmentation,
    ContrastivePairGenerator,
    TS2VecDataset
)
from .model import DilatedConvEncoder, ProjectionHead, NTXentLoss, TS2VecModel
from .training import TS2VecTrainer, OptimizedDataLoader
from .evaluation import TS2VecEvaluator

__all__ = [
    'SlidingWindowGenerator',
    'TimeSeriesAugmentation',
    'ContrastivePairGenerator',
    'TS2VecDataset',
    'DilatedConvEncoder',
    'ProjectionHead',
    'NTXentLoss',
    'TS2VecModel',
    'TS2VecTrainer',
    'OptimizedDataLoader',
    'TS2VecEvaluator'
]