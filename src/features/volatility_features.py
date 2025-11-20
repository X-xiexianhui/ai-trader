"""
波动率特征计算模块

实现5维波动率特征:
1. ATR14_norm: 归一化的14期平均真实波幅
2. vol_20: 20期滚动标准差
3. range_20_norm: 归一化的20期价格区间
4. BB_width_norm: 归一化的布林带宽度
5. parkinson_vol: Parkinson波动率估计

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VolatilityFeatureCalculator:
    """
    波动率特征计算器
    
    计算各种波动率指标，用于衡量市场风险和价格变动幅度。
    """
    
    def __init__(self):
        """初始化波动率特征计算器"""
        self.feature_names = [
            'ATR14_norm',
            'vol_20',
            'range_20_norm',
            'BB_width_norm',
            'parkinson_vol'
        ]
        logger.info("波动率特征计算器初始化完成")
    
    def calculate_atr(
        self,
        data: pd.DataFrame,
        window: int = 14,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        计算平均真实波幅 (Average True Range)
        
        ATR是衡量价格波动性的指标，考虑了跳空缺口。
        
        Args:
            data: 包含OHLC数据的DataFrame
            window: ATR计算窗口
            normalize: 是否归一化（除以收盘价）
            
        Returns:
            包含ATR特征的DataFrame
        """
        result = data.copy()
        
        # 计算真实波幅 (True Range)
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift(1))
        low_close = np.abs(result['low'] - result['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 计算ATR（使用指数移动平均）
        atr = true_range.ewm(span=window, adjust=False).mean()
        
        # 归一化
        if normalize:
            col_name = f'ATR{window}_norm'
            result[col_name] = atr / result['close']
        else:
            col_name = f'ATR{window}'
            result[col_name] = atr
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_rolling_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算滚动标准差波动率
        
        使用对数收益率的标准差衡量波动率。
        
        Args:
            data: 包含收盘价的DataFrame
            window: 滚动窗口大小
            
        Returns:
            包含波动率特征的DataFrame
        """
        result = data.copy()
        
        # 计算对数收益率
        log_returns = np.log(result['close'] / result['close'].shift(1))
        
        # 计算滚动标准差
        col_name = f'vol_{window}'
        result[col_name] = log_returns.rolling(
            window=window,
            min_periods=window
        ).std()
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_price_range(
        self,
        data: pd.DataFrame,
        window: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        计算价格区间
        
        衡量一段时间内的最高价和最低价之间的差距。
        
        Args:
            data: 包含OHLC数据的DataFrame
            window: 计算窗口
            normalize: 是否归一化
            
        Returns:
            包含价格区间特征的DataFrame
        """
        result = data.copy()
        
        # 计算滚动最高价和最低价
        rolling_high = result['high'].rolling(window=window, min_periods=window).max()
        rolling_low = result['low'].rolling(window=window, min_periods=window).min()
        
        # 计算价格区间
        price_range = rolling_high - rolling_low
        
        # 归一化
        if normalize:
            col_name = f'range_{window}_norm'
            result[col_name] = price_range / result['close']
        else:
            col_name = f'range_{window}'
            result[col_name] = price_range
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_bollinger_bands_width(
        self,
        data: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        计算布林带宽度
        
        布林带宽度反映了价格的波动范围。
        
        Args:
            data: 包含收盘价的DataFrame
            window: 移动平均窗口
            num_std: 标准差倍数
            normalize: 是否归一化
            
        Returns:
            包含布林带宽度特征的DataFrame
        """
        result = data.copy()
        
        # 计算移动平均和标准差
        ma = result['close'].rolling(window=window, min_periods=window).mean()
        std = result['close'].rolling(window=window, min_periods=window).std()
        
        # 计算布林带上下轨
        upper_band = ma + (num_std * std)
        lower_band = ma - (num_std * std)
        
        # 计算布林带宽度
        bb_width = upper_band - lower_band
        
        # 归一化
        if normalize:
            col_name = 'BB_width_norm'
            result[col_name] = bb_width / ma
        else:
            col_name = 'BB_width'
            result[col_name] = bb_width
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_parkinson_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算Parkinson波动率
        
        Parkinson波动率使用高低价信息，比仅使用收盘价更准确。
        公式: sqrt(1/(4*ln(2)) * mean((ln(high/low))^2))
        
        Args:
            data: 包含高低价的DataFrame
            window: 计算窗口
            
        Returns:
            包含Parkinson波动率特征的DataFrame
        """
        result = data.copy()
        
        # 计算 ln(high/low)^2
        hl_ratio = np.log(result['high'] / result['low'])
        hl_ratio_squared = hl_ratio ** 2
        
        # 计算滚动平均
        mean_hl_squared = hl_ratio_squared.rolling(
            window=window,
            min_periods=window
        ).mean()
        
        # 计算Parkinson波动率
        # 使用系数 1/(4*ln(2)) ≈ 0.361
        col_name = 'parkinson_vol'
        result[col_name] = np.sqrt(0.361 * mean_hl_squared)
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有波动率特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有波动率特征的DataFrame
            
        Raises:
            ValueError: 如果输入数据缺少必要的列
        """
        # 验证输入数据
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要的列: {missing_cols}")
        
        if len(data) < 20:
            logger.warning(f"数据长度 {len(data)} 小于最小窗口 20，某些特征可能无法计算")
        
        logger.info(f"开始计算波动率特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 计算ATR (ATR14_norm)
        result = self.calculate_atr(result, window=14, normalize=True)
        
        # 2. 计算滚动波动率 (vol_20)
        result = self.calculate_rolling_volatility(result, window=20)
        
        # 3. 计算价格区间 (range_20_norm)
        result = self.calculate_price_range(result, window=20, normalize=True)
        
        # 4. 计算布林带宽度 (BB_width_norm)
        result = self.calculate_bollinger_bands_width(
            result,
            window=20,
            num_std=2.0,
            normalize=True
        )
        
        # 5. 计算Parkinson波动率 (parkinson_vol)
        result = self.calculate_parkinson_volatility(result, window=20)
        
        # 统计特征计算结果
        feature_stats = {}
        for feature in self.feature_names:
            if feature in result.columns:
                valid_count = result[feature].notna().sum()
                feature_stats[feature] = {
                    'valid_count': valid_count,
                    'valid_ratio': valid_count / len(result),
                    'mean': result[feature].mean(),
                    'std': result[feature].std()
                }
        
        logger.info(f"波动率特征计算完成，共 {len(self.feature_names)} 个特征")
        
        return result
    
    def get_feature_names(self) -> list:
        """
        获取所有特征名称
        
        Returns:
            特征名称列表
        """
        return self.feature_names.copy()
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        获取特征描述
        
        Returns:
            特征名称到描述的字典
        """
        descriptions = {
            'ATR14_norm': '归一化的14期平均真实波幅 - 衡量价格波动幅度',
            'vol_20': '20期滚动标准差 - 衡量收益率波动性',
            'range_20_norm': '归一化的20期价格区间 - 衡量价格波动范围',
            'BB_width_norm': '归一化的布林带宽度 - 衡量价格波动带宽',
            'parkinson_vol': 'Parkinson波动率 - 基于高低价的波动率估计'
        }
        return descriptions


def calculate_volatility_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有波动率特征
    
    Args:
        data: 包含OHLCV数据的DataFrame
        
    Returns:
        包含波动率特征的DataFrame
    """
    calculator = VolatilityFeatureCalculator()
    return calculator.calculate_all_features(data)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    
    # 生成模拟价格数据（随机游走 + 波动率）
    returns = np.random.randn(100) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    volatility = np.abs(np.random.randn(100)) * 0.005
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * volatility),
        'high': prices * (1 + np.abs(np.random.randn(100)) * volatility * 2),
        'low': prices * (1 - np.abs(np.random.randn(100)) * volatility * 2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 确保OHLC一致性
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    # 计算波动率特征
    calculator = VolatilityFeatureCalculator()
    result = calculator.calculate_all_features(sample_data)
    
    # 显示结果
    print("\n=== 波动率特征计算示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print(f"结果数据形状: {result.shape}")
    
    print("\n特征列表:")
    for i, feature in enumerate(calculator.get_feature_names(), 1):
        desc = calculator.get_feature_descriptions()[feature]
        print(f"{i}. {feature}: {desc}")
    
    print("\n前5行特征值:")
    feature_cols = calculator.get_feature_names()
    print(result[feature_cols].head())
    
    print("\n特征统计:")
    print(result[feature_cols].describe())
    
    print("\n缺失值统计:")
    print(result[feature_cols].isna().sum())