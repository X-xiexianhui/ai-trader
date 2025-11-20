"""
技术指标特征计算模块

实现4维技术指标特征:
1. EMA20: 20期指数移动平均
2. stoch: 随机指标 (Stochastic Oscillator)
3. MACD: MACD指标
4. VWAP: 成交量加权平均价

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalFeatureCalculator:
    """
    技术指标特征计算器
    
    计算常用的技术分析指标，用于捕捉市场趋势和动量。
    """
    
    def __init__(self):
        """初始化技术指标特征计算器"""
        self.feature_names = [
            'EMA20',
            'stoch',
            'MACD',
            'VWAP'
        ]
        logger.info("技术指标特征计算器初始化完成")
    
    def calculate_ema(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算指数移动平均 (Exponential Moving Average)
        
        EMA对近期价格赋予更高权重，反应更灵敏。
        
        Args:
            data: 包含收盘价的DataFrame
            window: EMA周期
            
        Returns:
            包含EMA特征的DataFrame
        """
        result = data.copy()
        
        col_name = f'EMA{window}'
        result[col_name] = result['close'].ewm(
            span=window,
            adjust=False
        ).mean()
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_window: int = 14,
        d_window: int = 3
    ) -> pd.DataFrame:
        """
        计算随机指标 (Stochastic Oscillator)
        
        随机指标衡量收盘价在一定周期内的相对位置，用于判断超买超卖。
        %K = (Close - Low_n) / (High_n - Low_n) * 100
        %D = %K的移动平均
        
        Args:
            data: 包含OHLC数据的DataFrame
            k_window: %K计算周期
            d_window: %D平滑周期
            
        Returns:
            包含随机指标特征的DataFrame
        """
        result = data.copy()
        
        # 计算n期最高价和最低价
        low_min = result['low'].rolling(window=k_window, min_periods=k_window).min()
        high_max = result['high'].rolling(window=k_window, min_periods=k_window).max()
        
        # 计算%K
        stoch_k = 100 * (result['close'] - low_min) / (high_max - low_min)
        
        # 计算%D (对%K进行平滑)
        stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()
        
        # 使用%K作为主要特征（也可以使用%D或两者的组合）
        result['stoch'] = stoch_k
        
        # 处理除零情况
        result['stoch'] = result['stoch'].replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"计算 stoch: {result['stoch'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_macd(
        self,
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        计算MACD指标 (Moving Average Convergence Divergence)
        
        MACD是趋势跟踪动量指标，显示两条移动平均线之间的关系。
        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        
        Args:
            data: 包含收盘价的DataFrame
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
            
        Returns:
            包含MACD特征的DataFrame
        """
        result = data.copy()
        
        # 计算快速和慢速EMA
        ema_fast = result['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        macd_histogram = macd_line - signal_line
        
        # 使用MACD柱状图作为主要特征（也可以使用MACD线）
        result['MACD'] = macd_histogram
        
        logger.debug(f"计算 MACD: {result['MACD'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_vwap(
        self,
        data: pd.DataFrame,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        计算成交量加权平均价 (Volume Weighted Average Price)
        
        VWAP是价格和成交量的加权平均，常用于判断价格是否合理。
        VWAP = Σ(Price * Volume) / Σ(Volume)
        
        Args:
            data: 包含价格和成交量的DataFrame
            window: 计算窗口，None表示累积计算
            
        Returns:
            包含VWAP特征的DataFrame
        """
        result = data.copy()
        
        # 使用典型价格 (Typical Price) = (High + Low + Close) / 3
        typical_price = (result['high'] + result['low'] + result['close']) / 3
        
        # 计算价格*成交量
        pv = typical_price * result['volume']
        
        if window is None:
            # 累积VWAP（从开始到当前）
            result['VWAP'] = pv.cumsum() / result['volume'].cumsum()
        else:
            # 滚动VWAP
            pv_sum = pv.rolling(window=window, min_periods=window).sum()
            volume_sum = result['volume'].rolling(window=window, min_periods=window).sum()
            result['VWAP'] = pv_sum / volume_sum
        
        # 处理除零情况
        result['VWAP'] = result['VWAP'].replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"计算 VWAP: {result['VWAP'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有技术指标特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有技术指标特征的DataFrame
            
        Raises:
            ValueError: 如果输入数据缺少必要的列
        """
        # 验证输入数据
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要的列: {missing_cols}")
        
        if len(data) < 26:
            logger.warning(f"数据长度 {len(data)} 小于MACD所需的最小窗口 26，某些特征可能无法计算")
        
        logger.info(f"开始计算技术指标特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 计算EMA (EMA20)
        result = self.calculate_ema(result, window=20)
        
        # 2. 计算随机指标 (stoch)
        result = self.calculate_stochastic(result, k_window=14, d_window=3)
        
        # 3. 计算MACD (MACD)
        result = self.calculate_macd(
            result,
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        # 4. 计算VWAP (VWAP)
        # 使用20期滚动VWAP而不是累积VWAP，更适合特征工程
        result = self.calculate_vwap(result, window=20)
        
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
        
        logger.info(f"技术指标特征计算完成，共 {len(self.feature_names)} 个特征")
        
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
            'EMA20': '20期指数移动平均 - 衡量价格趋势',
            'stoch': '随机指标 - 衡量超买超卖状态',
            'MACD': 'MACD柱状图 - 衡量趋势强度和方向',
            'VWAP': '成交量加权平均价 - 衡量价格合理性'
        }
        return descriptions


def calculate_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有技术指标特征
    
    Args:
        data: 包含OHLCV数据的DataFrame
        
    Returns:
        包含技术指标特征的DataFrame
    """
    calculator = TechnicalFeatureCalculator()
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
    
    # 生成模拟价格数据（带趋势的随机游走）
    trend = np.linspace(0, 0.1, 100)
    returns = np.random.randn(100) * 0.01 + trend
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 确保OHLC一致性
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    # 计算技术指标特征
    calculator = TechnicalFeatureCalculator()
    result = calculator.calculate_all_features(sample_data)
    
    # 显示结果
    print("\n=== 技术指标特征计算示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print(f"结果数据形状: {result.shape}")
    
    print("\n特征列表:")
    for i, feature in enumerate(calculator.get_feature_names(), 1):
        desc = calculator.get_feature_descriptions()[feature]
        print(f"{i}. {feature}: {desc}")
    
    print("\n前5行特征值:")
    feature_cols = calculator.get_feature_names()
    print(result[feature_cols].head(30))  # 显示更多行以看到非NaN值
    
    print("\n特征统计:")
    print(result[feature_cols].describe())
    
    print("\n缺失值统计:")
    print(result[feature_cols].isna().sum())