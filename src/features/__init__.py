"""
特征工程模块

提供完整的特征计算功能，包括:
- 价格与收益特征 (5维)
- 波动率特征 (5维)
- 技术指标特征 (4维)
- 成交量特征 (4维)
- K线形态特征 (7维)
- 时间特征 (2维)
- 特征归一化
- 特征工程管道

Author: AI Trader Team
Date: 2025-11-20
"""

from .price_features import PriceFeatureCalculator, calculate_price_features
from .volatility_features import VolatilityFeatureCalculator, calculate_volatility_features
from .technical_features import TechnicalFeatureCalculator, calculate_technical_features
from .volume_features import VolumeFeatureCalculator, calculate_volume_features
from .candlestick_features import CandlestickFeatureCalculator, calculate_candlestick_features
from .time_features import TimeFeatureCalculator, calculate_time_features
from .normalizer import FeatureNormalizer
from .pipeline import FeatureEngineeringPipeline

__all__ = [
    # 特征计算器
    'PriceFeatureCalculator',
    'VolatilityFeatureCalculator',
    'TechnicalFeatureCalculator',
    'VolumeFeatureCalculator',
    'CandlestickFeatureCalculator',
    'TimeFeatureCalculator',
    
    # 便捷函数
    'calculate_price_features',
    'calculate_volatility_features',
    'calculate_technical_features',
    'calculate_volume_features',
    'calculate_candlestick_features',
    'calculate_time_features',
    
    # 归一化和管道
    'FeatureNormalizer',
    'FeatureEngineeringPipeline'
]

__version__ = '1.0.0'