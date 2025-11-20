"""
特征工程管道模块

实现完整的特征工程管道:
- 整合所有特征计算模块
- 计算27维手工特征
- 支持配置化流程
- 生成特征报告

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from .price_features import PriceFeatureCalculator
from .volatility_features import VolatilityFeatureCalculator
from .technical_features import TechnicalFeatureCalculator
from .volume_features import VolumeFeatureCalculator
from .candlestick_features import CandlestickFeatureCalculator
from .time_features import TimeFeatureCalculator
from .normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    特征工程管道
    
    串联所有特征计算步骤，生成完整的27维特征集。
    """
    
    def __init__(
        self,
        normalize: bool = True,
        timestamp_col: str = 'timestamp'
    ):
        """
        初始化特征工程管道
        
        Args:
            normalize: 是否进行特征归一化
            timestamp_col: 时间戳列名
        """
        self.normalize = normalize
        self.timestamp_col = timestamp_col
        
        # 初始化各个特征计算器
        self.price_calculator = PriceFeatureCalculator()
        self.volatility_calculator = VolatilityFeatureCalculator()
        self.technical_calculator = TechnicalFeatureCalculator()
        self.volume_calculator = VolumeFeatureCalculator()
        self.candlestick_calculator = CandlestickFeatureCalculator()
        self.time_calculator = TimeFeatureCalculator()
        
        # 初始化归一化器
        self.normalizer = FeatureNormalizer() if normalize else None
        
        # 所有27维特征名称
        self.all_features = [
            # 价格与收益特征 (5维)
            'ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20',
            # 波动率特征 (5维)
            'ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol',
            # 技术指标特征 (4维)
            'EMA20', 'stoch', 'MACD', 'VWAP',
            # 成交量特征 (4维)
            'volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20',
            # K线形态特征 (7维)
            'pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm',
            'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG',
            # 时间特征 (2维)
            'sin_tod', 'cos_tod'
        ]
        
        logger.info("特征工程管道初始化完成")
    
    def calculate_features(
        self,
        data: pd.DataFrame,
        include_price: bool = True,
        include_volatility: bool = True,
        include_technical: bool = True,
        include_volume: bool = True,
        include_candlestick: bool = True,
        include_time: bool = True
    ) -> pd.DataFrame:
        """
        计算所有特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            include_price: 是否包含价格特征
            include_volatility: 是否包含波动率特征
            include_technical: 是否包含技术指标特征
            include_volume: 是否包含成交量特征
            include_candlestick: 是否包含K线形态特征
            include_time: 是否包含时间特征
            
        Returns:
            包含所有特征的DataFrame
            
        Raises:
            ValueError: 如果输入数据缺少必要的列
        """
        # 验证输入数据
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if include_time:
            required_cols.append(self.timestamp_col)
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要的列: {missing_cols}")
        
        logger.info(f"开始计算特征，数据长度: {len(data)}")
        
        result = data.copy()
        calculated_features = []
        
        # 1. 计算价格与收益特征 (5维)
        if include_price:
            logger.info("计算价格与收益特征...")
            result = self.price_calculator.calculate_all_features(result)
            calculated_features.extend(self.price_calculator.get_feature_names())
        
        # 2. 计算波动率特征 (5维)
        if include_volatility:
            logger.info("计算波动率特征...")
            result = self.volatility_calculator.calculate_all_features(result)
            calculated_features.extend(self.volatility_calculator.get_feature_names())
        
        # 3. 计算技术指标特征 (4维)
        if include_technical:
            logger.info("计算技术指标特征...")
            result = self.technical_calculator.calculate_all_features(result)
            calculated_features.extend(self.technical_calculator.get_feature_names())
        
        # 4. 计算成交量特征 (4维)
        if include_volume:
            logger.info("计算成交量特征...")
            result = self.volume_calculator.calculate_all_features(result)
            calculated_features.extend(self.volume_calculator.get_feature_names())
        
        # 5. 计算K线形态特征 (7维)
        if include_candlestick:
            logger.info("计算K线形态特征...")
            result = self.candlestick_calculator.calculate_all_features(result)
            calculated_features.extend(self.candlestick_calculator.get_feature_names())
        
        # 6. 计算时间特征 (2维)
        if include_time:
            logger.info("计算时间特征...")
            result = self.time_calculator.calculate_all_features(
                result,
                timestamp_col=self.timestamp_col
            )
            calculated_features.extend(self.time_calculator.get_feature_names())
        
        logger.info(f"特征计算完成，共 {len(calculated_features)} 个特征")
        
        return result
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        计算特征并拟合归一化器
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 传递给calculate_features的参数
            
        Returns:
            (特征DataFrame, 报告字典)
        """
        # 计算特征
        result = self.calculate_features(data, **kwargs)
        
        # 归一化
        if self.normalize and self.normalizer is not None:
            logger.info("拟合并应用特征归一化...")
            
            # 获取需要归一化的特征
            feature_cols = [col for col in self.all_features if col in result.columns]
            
            # 拟合并转换
            result = self.normalizer.fit_transform(result)
        
        # 生成报告
        report = self._generate_report(result)
        
        return result, report
    
    def transform(
        self,
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        计算特征并应用已拟合的归一化器
        
        Args:
            data: 包含OHLCV数据的DataFrame
            **kwargs: 传递给calculate_features的参数
            
        Returns:
            特征DataFrame
        """
        # 计算特征
        result = self.calculate_features(data, **kwargs)
        
        # 应用归一化
        if self.normalize and self.normalizer is not None:
            if not self.normalizer.is_fitted:
                raise ValueError("归一化器未拟合，请先调用 fit_transform()")
            
            logger.info("应用特征归一化...")
            result = self.normalizer.transform(result)
        
        return result
    
    def _generate_report(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        生成特征计算报告
        
        Args:
            data: 包含特征的DataFrame
            
        Returns:
            报告字典
        """
        report = {
            'total_records': len(data),
            'total_features': 27,
            'calculated_features': 0,
            'feature_stats': {},
            'missing_stats': {}
        }
        
        # 统计每个特征
        for feature in self.all_features:
            if feature in data.columns:
                report['calculated_features'] += 1
                
                valid_count = data[feature].notna().sum()
                missing_count = data[feature].isna().sum()
                
                report['feature_stats'][feature] = {
                    'valid_count': valid_count,
                    'missing_count': missing_count,
                    'valid_ratio': valid_count / len(data),
                    'mean': data[feature].mean(),
                    'std': data[feature].std(),
                    'min': data[feature].min(),
                    'max': data[feature].max()
                }
                
                report['missing_stats'][feature] = {
                    'count': missing_count,
                    'ratio': missing_count / len(data)
                }
        
        # 总体缺失统计
        feature_cols = [col for col in self.all_features if col in data.columns]
        if feature_cols:
            total_missing = data[feature_cols].isna().sum().sum()
            total_values = len(data) * len(feature_cols)
            report['overall_missing_ratio'] = total_missing / total_values
        
        return report
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名称
        
        Returns:
            特征名称列表
        """
        return self.all_features.copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        获取所有特征的描述
        
        Returns:
            特征名称到描述的字典
        """
        descriptions = {}
        
        descriptions.update(self.price_calculator.get_feature_descriptions())
        descriptions.update(self.volatility_calculator.get_feature_descriptions())
        descriptions.update(self.technical_calculator.get_feature_descriptions())
        descriptions.update(self.volume_calculator.get_feature_descriptions())
        descriptions.update(self.candlestick_calculator.get_feature_descriptions())
        descriptions.update(self.time_calculator.get_feature_descriptions())
        
        return descriptions
    
    def save_normalizer(
        self,
        filepath: str
    ) -> None:
        """
        保存归一化器
        
        Args:
            filepath: 保存路径
        """
        if self.normalizer is None:
            raise ValueError("管道未启用归一化")
        
        self.normalizer.save(filepath)
        logger.info(f"归一化器已保存到: {filepath}")
    
    def load_normalizer(
        self,
        filepath: str
    ) -> None:
        """
        加载归一化器
        
        Args:
            filepath: 加载路径
        """
        if self.normalizer is None:
            self.normalizer = FeatureNormalizer()
        
        self.normalizer.load(filepath)
        logger.info(f"归一化器已从 {filepath} 加载")
    
    def print_report(
        self,
        report: Dict
    ) -> None:
        """
        打印特征计算报告
        
        Args:
            report: 报告字典
        """
        print("\n" + "="*60)
        print("特征工程报告")
        print("="*60)
        
        print(f"\n总记录数: {report['total_records']}")
        print(f"总特征数: {report['total_features']}")
        print(f"已计算特征数: {report['calculated_features']}")
        print(f"总体缺失率: {report.get('overall_missing_ratio', 0):.2%}")
        
        print("\n特征缺失统计:")
        print("-" * 60)
        for feature, stats in report['missing_stats'].items():
            if stats['count'] > 0:
                print(f"{feature:25s}: {stats['count']:6d} ({stats['ratio']:.2%})")
        
        print("\n特征统计摘要:")
        print("-" * 60)
        print(f"{'特征名称':<25s} {'均值':>10s} {'标准差':>10s} {'最小值':>10s} {'最大值':>10s}")
        print("-" * 60)
        
        for feature, stats in report['feature_stats'].items():
            print(f"{feature:<25s} "
                  f"{stats['mean']:>10.4f} "
                  f"{stats['std']:>10.4f} "
                  f"{stats['min']:>10.4f} "
                  f"{stats['max']:>10.4f}")
        
        print("="*60)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
    
    # 生成模拟价格数据
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.01),
        'close': prices * (1 + np.random.randn(n_samples) * 0.005),
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # 确保OHLC一致性
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    print("\n=== 特征工程管道示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print("\n原始数据前5行:")
    print(sample_data.head())
    
    # 创建特征工程管道
    pipeline = FeatureEngineeringPipeline(normalize=True)
    
    # 计算特征
    features, report = pipeline.fit_transform(sample_data)
    
    print(f"\n特征数据形状: {features.shape}")
    
    # 打印报告
    pipeline.print_report(report)
    
    # 显示特征数据
    print("\n特征数据前5行:")
    feature_cols = pipeline.get_feature_names()
    available_features = [col for col in feature_cols if col in features.columns]
    print(features[available_features].head())
    
    # 保存归一化器
    save_path = "scalers/feature_normalizer.pkl"
    pipeline.save_normalizer(save_path)
    print(f"\n归一化器已保存到: {save_path}")
    
    # 测试transform（使用已拟合的归一化器）
    new_data = sample_data.iloc[50:60].copy()
    transformed_data = pipeline.transform(new_data)
    print(f"\n新数据转换后形状: {transformed_data.shape}")