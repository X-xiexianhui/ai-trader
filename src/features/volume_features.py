"""
成交量特征计算模块

实现4维成交量特征:
1. volume: 原始成交量
2. volume_zscore: 成交量Z-score标准化
3. volume_change_1: 1期成交量变化率
4. OBV_slope_20: 20期OBV斜率

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeFeatureCalculator:
    """
    成交量特征计算器
    
    计算基于成交量的技术特征，用于分析市场参与度和资金流向。
    """
    
    def __init__(self):
        """初始化成交量特征计算器"""
        self.feature_names = [
            'volume',
            'volume_zscore',
            'volume_change_1',
            'OBV_slope_20'
        ]
        logger.info("成交量特征计算器初始化完成")
    
    def calculate_volume_zscore(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算成交量Z-score标准化
        
        Z-score = (volume - mean) / std
        用于识别异常成交量。
        
        Args:
            data: 包含成交量的DataFrame
            window: 计算均值和标准差的窗口
            
        Returns:
            包含成交量Z-score特征的DataFrame
        """
        result = data.copy()
        
        # 计算滚动均值和标准差
        volume_mean = result['volume'].rolling(
            window=window,
            min_periods=window
        ).mean()
        
        volume_std = result['volume'].rolling(
            window=window,
            min_periods=window
        ).std()
        
        # 计算Z-score
        result['volume_zscore'] = (result['volume'] - volume_mean) / volume_std
        
        # 处理除零和无穷大
        result['volume_zscore'] = result['volume_zscore'].replace(
            [np.inf, -np.inf], np.nan
        )
        
        logger.debug(f"计算 volume_zscore: {result['volume_zscore'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_volume_change(
        self,
        data: pd.DataFrame,
        period: int = 1
    ) -> pd.DataFrame:
        """
        计算成交量变化率
        
        变化率 = (volume_t - volume_{t-period}) / volume_{t-period}
        
        Args:
            data: 包含成交量的DataFrame
            period: 变化率周期
            
        Returns:
            包含成交量变化率特征的DataFrame
        """
        result = data.copy()
        
        col_name = f'volume_change_{period}'
        
        # 计算变化率
        result[col_name] = result['volume'].pct_change(periods=period)
        
        # 处理无穷大和NaN
        result[col_name] = result[col_name].replace([np.inf, -np.inf], np.nan)
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_obv(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算能量潮指标 (On-Balance Volume)
        
        OBV是累积成交量指标，根据价格变化方向累加或累减成交量。
        - 如果收盘价上涨，OBV += volume
        - 如果收盘价下跌，OBV -= volume
        - 如果收盘价不变，OBV不变
        
        Args:
            data: 包含价格和成交量的DataFrame
            
        Returns:
            包含OBV的DataFrame
        """
        result = data.copy()
        
        # 计算价格变化方向
        price_change = result['close'].diff()
        
        # 根据价格方向调整成交量符号
        signed_volume = np.where(
            price_change > 0,
            result['volume'],
            np.where(
                price_change < 0,
                -result['volume'],
                0
            )
        )
        
        # 累积计算OBV
        result['OBV'] = signed_volume.cumsum()
        
        logger.debug(f"计算 OBV: {result['OBV'].notna().sum()} 个有效值")
        
        return result
    
    def calculate_obv_slope(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        计算OBV斜率
        
        使用线性回归计算OBV的趋势斜率。
        
        Args:
            data: 包含OBV的DataFrame
            window: 回归窗口大小
            
        Returns:
            包含OBV斜率特征的DataFrame
        """
        # 先计算OBV（如果还没有）
        if 'OBV' not in data.columns:
            result = self.calculate_obv(data)
        else:
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
        
        # 使用滚动窗口计算OBV斜率
        col_name = f'OBV_slope_{window}'
        result[col_name] = result['OBV'].rolling(
            window=window,
            min_periods=window
        ).apply(linear_regression_slope, raw=False)
        
        # 删除中间计算的OBV列（如果原始数据中没有）
        if 'OBV' not in data.columns:
            result = result.drop(columns=['OBV'])
        
        logger.debug(f"计算 {col_name}: {result[col_name].notna().sum()} 个有效值")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有成交量特征
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有成交量特征的DataFrame
            
        Raises:
            ValueError: 如果输入数据缺少必要的列
        """
        # 验证输入数据
        required_cols = ['close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要的列: {missing_cols}")
        
        if len(data) < 20:
            logger.warning(f"数据长度 {len(data)} 小于最小窗口 20，某些特征可能无法计算")
        
        logger.info(f"开始计算成交量特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 原始成交量已经在数据中，无需额外计算
        # 但我们确保列名正确
        if 'volume' not in result.columns:
            raise ValueError("数据中必须包含 'volume' 列")
        
        # 2. 计算成交量Z-score (volume_zscore)
        result = self.calculate_volume_zscore(result, window=20)
        
        # 3. 计算成交量变化率 (volume_change_1)
        result = self.calculate_volume_change(result, period=1)
        
        # 4. 计算OBV斜率 (OBV_slope_20)
        result = self.calculate_obv_slope(result, window=20)
        
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
        
        logger.info(f"成交量特征计算完成，共 {len(self.feature_names)} 个特征")
        
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
            'volume': '原始成交量 - 衡量市场参与度',
            'volume_zscore': '成交量Z-score - 识别异常成交量',
            'volume_change_1': '1期成交量变化率 - 衡量成交量变化',
            'OBV_slope_20': '20期OBV斜率 - 衡量资金流向趋势'
        }
        return descriptions


def calculate_volume_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有成交量特征
    
    Args:
        data: 包含OHLCV数据的DataFrame
        
    Returns:
        包含成交量特征的DataFrame
    """
    calculator = VolumeFeatureCalculator()
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
    
    # 生成成交量数据（与价格变化相关）
    base_volume = 5000
    volume_noise = np.random.randint(-1000, 1000, 100)
    price_impact = np.abs(returns) * 100000  # 价格变化越大，成交量越大
    volumes = base_volume + volume_noise + price_impact
    volumes = np.maximum(volumes, 100)  # 确保成交量为正
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.002),
        'close': prices,
        'volume': volumes.astype(int)
    })
    
    # 计算成交量特征
    calculator = VolumeFeatureCalculator()
    result = calculator.calculate_all_features(sample_data)
    
    # 显示结果
    print("\n=== 成交量特征计算示例 ===")
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