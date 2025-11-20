"""
价格与收益特征计算模块

实现5维价格与收益特征:
1. ret_1: 1期对数收益率
2. ret_5: 5期对数收益率
3. ret_20: 20期对数收益率
4. price_slope_20: 20期价格线性回归斜率
5. C_div_MA20: 收盘价/20期均线比值

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PriceFeatureCalculator:
    """
    价格与收益特征计算器
    
    计算基于价格的技术特征，包括收益率、趋势和相对位置等。
    """
    
    def __init__(self):
        """初始化价格特征计算器"""
        self.feature_names = [
            'ret_1',
            'ret_5', 
            'ret_20',
            'price_slope_20',
            'C_div_MA20'
        ]
        logger.info("价格特征计算器初始化完成")
    
    def calculate_returns(
        self,
        data: pd.DataFrame,
        periods: list = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        计算对数收益率
        
        Args:
            data: 包含OHLCV数据的DataFrame
            periods: 收益率周期列表
            
        Returns:
            包含收益率特征的DataFrame
        """
        result = data.copy()
        
        for period in periods:
            # 计算对数收益率: log(price_t / price_{t-period})
            col_name = f'ret_{period}'
            result[col_name] = np.log(result['close'] / result['close'].shift(period))
            
            # 处理无穷大和NaN值
            result[col_name] = result[col_name].replace([np.inf, -np.inf], np.nan)
            
            logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_price_slope(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算价格线性回归斜率
        
        使用滚动窗口对价格进行线性回归，提取斜率作为趋势指标。
        
        Args:
            data: 包含OHLCV数据的DataFrame
            window: 回归窗口大小
            
        Returns:
            包含价格斜率特征的DataFrame
        """
        result = data.copy()
        
        def linear_regression_slope(y):
            """计算线性回归斜率"""
            if len(y) < 2 or y.isna().any():
                return np.nan
            
            x = np.arange(len(y))
            # 使用最小二乘法计算斜率
            x_mean = x.mean()
            y_mean = y.mean()
            
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = ((x - x_mean) ** 2).sum()
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            return slope
        
        # 使用滚动窗口计算斜率
        col_name = f'price_slope_{window}'
        result[col_name] = result['close'].rolling(
            window=window,
            min_periods=window
        ).apply(linear_regression_slope, raw=False)
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_price_ma_ratio(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算收盘价与移动平均线的比值
        
        Args:
            data: 包含OHLCV数据的DataFrame
            window: 移动平均窗口大小
            
        Returns:
            包含价格/均线比值特征的DataFrame
        """
        result = data.copy()
        
        # 计算移动平均
        ma_col = f'MA{window}'
        result[ma_col] = result['close'].rolling(
            window=window,
            min_periods=window
        ).mean()
        
        # 计算比值
        ratio_col = f'C_div_MA{window}'
        result[ratio_col] = result['close'] / result[ma_col]
        
        # 删除中间计算列
        result = result.drop(columns=[ma_col])
        
        logger.debug(f"计算 {ratio_col}: {result[ratio_col].notna().sum()} 个有效值")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有价格与收益特征
        
        Args:
            data: 包含OHLCV数据的DataFrame，必须包含 'close' 列
            
        Returns:
            包含所有价格特征的DataFrame
            
        Raises:
            ValueError: 如果输入数据缺少必要的列
        """
        # 验证输入数据
        if 'close' not in data.columns:
            raise ValueError("输入数据必须包含 'close' 列")
        
        if len(data) < 20:
            logger.warning(f"数据长度 {len(data)} 小于最小窗口 20，某些特征可能无法计算")
        
        logger.info(f"开始计算价格特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 计算收益率特征 (ret_1, ret_5, ret_20)
        result = self.calculate_returns(result, periods=[1, 5, 20])
        
        # 2. 计算价格斜率 (price_slope_20)
        result = self.calculate_price_slope(result, window=20)
        
        # 3. 计算价格/均线比值 (C_div_MA20)
        result = self.calculate_price_ma_ratio(result, window=20)
        
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
        
        logger.info(f"价格特征计算完成，共 {len(self.feature_names)} 个特征")
        
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
            'ret_1': '1期对数收益率 - 衡量短期价格变化',
            'ret_5': '5期对数收益率 - 衡量中短期价格变化',
            'ret_20': '20期对数收益率 - 衡量中期价格变化',
            'price_slope_20': '20期价格线性回归斜率 - 衡量价格趋势强度',
            'C_div_MA20': '收盘价/20期均线比值 - 衡量价格相对位置'
        }
        return descriptions


def calculate_price_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有价格特征
    
    Args:
        data: 包含OHLCV数据的DataFrame
        
    Returns:
        包含价格特征的DataFrame
    """
    calculator = PriceFeatureCalculator()
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
    
    # 生成模拟价格数据（随机游走）
    returns = np.random.randn(100) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # 计算价格特征
    calculator = PriceFeatureCalculator()
    result = calculator.calculate_all_features(sample_data)
    
    # 显示结果
    print("\n=== 价格特征计算示例 ===")
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