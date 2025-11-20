"""
K线形态特征计算模块

实现7维K线形态特征:
1. pos_in_range_20: 收盘价在20期区间中的相对位置
2. dist_to_HH20_norm: 距离20期最高点的归一化距离
3. dist_to_LL20_norm: 距离20期最低点的归一化距离
4. body_ratio: K线实体占整体的比例
5. upper_shadow_ratio: 上影线占整体的比例
6. lower_shadow_ratio: 下影线占整体的比例
7. FVG: 公允价值缺口 (Fair Value Gap)

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CandlestickFeatureCalculator:
    """
    K线形态特征计算器
    
    计算基于K线形态的技术特征，用于分析价格结构和市场心理。
    """
    
    def __init__(self):
        """初始化K线形态特征计算器"""
        self.feature_names = [
            'pos_in_range_20',
            'dist_to_HH20_norm',
            'dist_to_LL20_norm',
            'body_ratio',
            'upper_shadow_ratio',
            'lower_shadow_ratio',
            'FVG'
        ]
        logger.info("K线形态特征计算器初始化完成")
    
    def calculate_position_in_range(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算收盘价在区间中的相对位置
        
        相对位置 = (Close - Low_n) / (High_n - Low_n)
        值域 [0, 1]，0表示在最低点，1表示在最高点
        
        Args:
            data: 包含OHLC数据的DataFrame
            window: 计算窗口
            
        Returns:
            包含相对位置特征的DataFrame
        """
        result = data.copy()
        
        # 计算滚动最高价和最低价
        rolling_high = result['high'].rolling(window=window, min_periods=window).max()
        rolling_low = result['low'].rolling(window=window, min_periods=window).min()
        
        # 计算相对位置
        col_name = f'pos_in_range_{window}'
        result[col_name] = (result['close'] - rolling_low) / (rolling_high - rolling_low)
        
        # 处理除零情况
        result[col_name] = result[col_name].replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_distance_to_extremes(
        self,
        data: pd.DataFrame,
        window: int = 20,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        计算距离最高点和最低点的距离
        
        Args:
            data: 包含OHLC数据的DataFrame
            window: 计算窗口
            normalize: 是否归一化（除以收盘价）
            
        Returns:
            包含距离特征的DataFrame
        """
        result = data.copy()
        
        # 计算滚动最高价和最低价
        rolling_high = result['high'].rolling(window=window, min_periods=window).max()
        rolling_low = result['low'].rolling(window=window, min_periods=window).min()
        
        # 计算距离
        dist_to_high = rolling_high - result['close']
        dist_to_low = result['close'] - rolling_low
        
        # 归一化
        if normalize:
            result[f'dist_to_HH{window}_norm'] = dist_to_high / result['close']
            result[f'dist_to_LL{window}_norm'] = dist_to_low / result['close']
        else:
            result[f'dist_to_HH{window}'] = dist_to_high
            result[f'dist_to_LL{window}'] = dist_to_low
        
        logger.debug(f"计算距离特征: HH和LL")
        
        return result
    
    def calculate_candle_ratios(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算K线各部分的比例
        
        - body_ratio: 实体占整体的比例
        - upper_shadow_ratio: 上影线占整体的比例
        - lower_shadow_ratio: 下影线占整体的比例
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            包含K线比例特征的DataFrame
        """
        result = data.copy()
        
        # 计算K线各部分长度
        total_range = result['high'] - result['low']
        body = np.abs(result['close'] - result['open'])
        upper_shadow = result['high'] - np.maximum(result['open'], result['close'])
        lower_shadow = np.minimum(result['open'], result['close']) - result['low']
        
        # 计算比例
        # 避免除零，当total_range为0时，所有比例都设为0
        result['body_ratio'] = np.where(
            total_range > 0,
            body / total_range,
            0
        )
        
        result['upper_shadow_ratio'] = np.where(
            total_range > 0,
            upper_shadow / total_range,
            0
        )
        
        result['lower_shadow_ratio'] = np.where(
            total_range > 0,
            lower_shadow / total_range,
            0
        )
        
        logger.debug("计算K线比例特征: body, upper_shadow, lower_shadow")
        
        return result
    
    def calculate_fvg(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算公允价值缺口 (Fair Value Gap)
        
        FVG是指三根K线之间出现的价格缺口，表示市场失衡。
        
        看涨FVG: 当前K线的低点 > 两根K线前的高点
        看跌FVG: 当前K线的高点 < 两根K线前的低点
        
        返回值:
        - 正值: 看涨FVG的大小
        - 负值: 看跌FVG的大小
        - 0: 无FVG
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            包含FVG特征的DataFrame
        """
        result = data.copy()
        
        # 获取前两根K线的高低点
        high_2 = result['high'].shift(2)
        low_2 = result['low'].shift(2)
        
        # 计算看涨FVG（当前低点 > 两根前的高点）
        bullish_fvg = np.where(
            result['low'] > high_2,
            result['low'] - high_2,
            0
        )
        
        # 计算看跌FVG（当前高点 < 两根前的低点）
        bearish_fvg = np.where(
            result['high'] < low_2,
            result['high'] - low_2,  # 这将是负值
            0
        )
        
        # 合并FVG（看涨为正，看跌为负）
        result['FVG'] = bullish_fvg + bearish_fvg
        
        # 归一化（除以收盘价）
        result['FVG'] = result['FVG'] / result['close']
        
        logger.debug(f"计算 FVG: {result['FVG'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有K线形态特征
        
        Args:
            data: 包含OHLC数据的DataFrame
            
        Returns:
            包含所有K线形态特征的DataFrame
            
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
        
        logger.info(f"开始计算K线形态特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 计算相对位置 (pos_in_range_20)
        result = self.calculate_position_in_range(result, window=20)
        
        # 2. 计算距离最高最低点 (dist_to_HH20_norm, dist_to_LL20_norm)
        result = self.calculate_distance_to_extremes(result, window=20, normalize=True)
        
        # 3. 计算K线比例 (body_ratio, upper_shadow_ratio, lower_shadow_ratio)
        result = self.calculate_candle_ratios(result)
        
        # 4. 计算FVG (FVG)
        result = self.calculate_fvg(result)
        
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
        
        logger.info(f"K线形态特征计算完成，共 {len(self.feature_names)} 个特征")
        
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
            'pos_in_range_20': '收盘价在20期区间中的相对位置 - 衡量价格在区间中的位置',
            'dist_to_HH20_norm': '距离20期最高点的归一化距离 - 衡量距离阻力位的距离',
            'dist_to_LL20_norm': '距离20期最低点的归一化距离 - 衡量距离支撑位的距离',
            'body_ratio': 'K线实体占整体的比例 - 衡量多空力量对比',
            'upper_shadow_ratio': '上影线占整体的比例 - 衡量上方压力',
            'lower_shadow_ratio': '下影线占整体的比例 - 衡量下方支撑',
            'FVG': '公允价值缺口 - 识别市场失衡区域'
        }
        return descriptions


def calculate_candlestick_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有K线形态特征
    
    Args:
        data: 包含OHLC数据的DataFrame
        
    Returns:
        包含K线形态特征的DataFrame
    """
    calculator = CandlestickFeatureCalculator()
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
    
    # 生成模拟价格数据
    returns = np.random.randn(100) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLC数据
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        'close': prices * (1 + np.random.randn(100) * 0.005),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 确保OHLC一致性
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    # 计算K线形态特征
    calculator = CandlestickFeatureCalculator()
    result = calculator.calculate_all_features(sample_data)
    
    # 显示结果
    print("\n=== K线形态特征计算示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print(f"结果数据形状: {result.shape}")
    
    print("\n特征列表:")
    for i, feature in enumerate(calculator.get_feature_names(), 1):
        desc = calculator.get_feature_descriptions()[feature]
        print(f"{i}. {feature}: {desc}")
    
    print("\n前10行特征值:")
    feature_cols = calculator.get_feature_names()
    print(result[feature_cols].head(25))
    
    print("\n特征统计:")
    print(result[feature_cols].describe())
    
    print("\n缺失值统计:")
    print(result[feature_cols].isna().sum())
    
    # 显示一些有FVG的样本
    fvg_samples = result[result['FVG'] != 0][['timestamp', 'open', 'high', 'low', 'close', 'FVG']].head(10)
    if len(fvg_samples) > 0:
        print("\n检测到的FVG样本:")
        print(fvg_samples)