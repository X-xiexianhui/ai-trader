"""
时间周期特征计算模块

实现2维时间特征:
1. sin_tod: 时间的正弦编码 (Time of Day)
2. cos_tod: 时间的余弦编码 (Time of Day)

使用正弦和余弦编码可以保持时间的周期性特征。

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TimeFeatureCalculator:
    """
    时间周期特征计算器
    
    使用三角函数编码时间特征，保持时间的周期性和连续性。
    """
    
    def __init__(self):
        """初始化时间特征计算器"""
        self.feature_names = [
            'sin_tod',
            'cos_tod'
        ]
        logger.info("时间特征计算器初始化完成")
    
    def calculate_time_of_day_encoding(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        计算时间的正弦和余弦编码
        
        将一天中的时间（0-24小时）编码为正弦和余弦值，
        这样可以保持时间的周期性（23:59和00:00是相邻的）。
        
        公式:
        - sin_tod = sin(2π * hour / 24)
        - cos_tod = cos(2π * hour / 24)
        
        Args:
            data: 包含时间戳的DataFrame
            timestamp_col: 时间戳列名
            
        Returns:
            包含时间编码特征的DataFrame
            
        Raises:
            ValueError: 如果时间戳列不存在或格式不正确
        """
        result = data.copy()
        
        # 验证时间戳列
        if timestamp_col not in result.columns:
            raise ValueError(f"数据中不存在时间戳列: {timestamp_col}")
        
        # 确保时间戳列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            try:
                result[timestamp_col] = pd.to_datetime(result[timestamp_col])
            except Exception as e:
                raise ValueError(f"无法将 {timestamp_col} 转换为datetime类型: {e}")
        
        # 提取小时和分钟
        hours = result[timestamp_col].dt.hour
        minutes = result[timestamp_col].dt.minute
        
        # 将时间转换为小数形式（0-24）
        time_decimal = hours + minutes / 60.0
        
        # 计算正弦和余弦编码
        # 使用2π来表示一天的周期
        result['sin_tod'] = np.sin(2 * np.pi * time_decimal / 24.0)
        result['cos_tod'] = np.cos(2 * np.pi * time_decimal / 24.0)
        
        logger.debug(f"计算时间编码特征: sin_tod, cos_tod")
        
        return result
    
    def calculate_day_of_week_encoding(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        计算星期的正弦和余弦编码（可选功能）
        
        将一周中的天数（0-6）编码为正弦和余弦值。
        
        Args:
            data: 包含时间戳的DataFrame
            timestamp_col: 时间戳列名
            
        Returns:
            包含星期编码特征的DataFrame
        """
        result = data.copy()
        
        # 确保时间戳列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            result[timestamp_col] = pd.to_datetime(result[timestamp_col])
        
        # 提取星期几（0=Monday, 6=Sunday）
        day_of_week = result[timestamp_col].dt.dayofweek
        
        # 计算正弦和余弦编码
        result['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7.0)
        result['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7.0)
        
        logger.debug(f"计算星期编码特征: sin_dow, cos_dow")
        
        return result
    
    def calculate_month_encoding(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        计算月份的正弦和余弦编码（可选功能）
        
        将一年中的月份（1-12）编码为正弦和余弦值。
        
        Args:
            data: 包含时间戳的DataFrame
            timestamp_col: 时间戳列名
            
        Returns:
            包含月份编码特征的DataFrame
        """
        result = data.copy()
        
        # 确保时间戳列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
            result[timestamp_col] = pd.to_datetime(result[timestamp_col])
        
        # 提取月份（1-12）
        month = result[timestamp_col].dt.month
        
        # 计算正弦和余弦编码
        result['sin_month'] = np.sin(2 * np.pi * month / 12.0)
        result['cos_month'] = np.cos(2 * np.pi * month / 12.0)
        
        logger.debug(f"计算月份编码特征: sin_month, cos_month")
        
        return result
    
    def calculate_all_features(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        include_dow: bool = False,
        include_month: bool = False
    ) -> pd.DataFrame:
        """
        计算所有时间特征
        
        Args:
            data: 包含时间戳的DataFrame
            timestamp_col: 时间戳列名
            include_dow: 是否包含星期编码
            include_month: 是否包含月份编码
            
        Returns:
            包含所有时间特征的DataFrame
            
        Raises:
            ValueError: 如果时间戳列不存在
        """
        if timestamp_col not in data.columns:
            raise ValueError(f"数据中不存在时间戳列: {timestamp_col}")
        
        logger.info(f"开始计算时间特征，数据长度: {len(data)}")
        
        result = data.copy()
        
        # 1. 计算时间编码 (sin_tod, cos_tod) - 必需
        result = self.calculate_time_of_day_encoding(result, timestamp_col)
        
        # 2. 计算星期编码（可选）
        if include_dow:
            result = self.calculate_day_of_week_encoding(result, timestamp_col)
        
        # 3. 计算月份编码（可选）
        if include_month:
            result = self.calculate_month_encoding(result, timestamp_col)
        
        # 统计特征计算结果
        feature_stats = {}
        for feature in self.feature_names:
            if feature in result.columns:
                valid_count = result[feature].notna().sum()
                feature_stats[feature] = {
                    'valid_count': valid_count,
                    'valid_ratio': valid_count / len(result),
                    'mean': result[feature].mean(),
                    'std': result[feature].std(),
                    'min': result[feature].min(),
                    'max': result[feature].max()
                }
        
        logger.info(f"时间特征计算完成，共 {len(self.feature_names)} 个特征")
        
        return result
    
    def get_feature_names(self, include_dow: bool = False, include_month: bool = False) -> list:
        """
        获取所有特征名称
        
        Args:
            include_dow: 是否包含星期特征
            include_month: 是否包含月份特征
            
        Returns:
            特征名称列表
        """
        names = self.feature_names.copy()
        
        if include_dow:
            names.extend(['sin_dow', 'cos_dow'])
        
        if include_month:
            names.extend(['sin_month', 'cos_month'])
        
        return names
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        获取特征描述
        
        Returns:
            特征名称到描述的字典
        """
        descriptions = {
            'sin_tod': '时间的正弦编码 - 保持时间周期性（一天）',
            'cos_tod': '时间的余弦编码 - 保持时间周期性（一天）',
            'sin_dow': '星期的正弦编码 - 保持星期周期性（一周）',
            'cos_dow': '星期的余弦编码 - 保持星期周期性（一周）',
            'sin_month': '月份的正弦编码 - 保持月份周期性（一年）',
            'cos_month': '月份的余弦编码 - 保持月份周期性（一年）'
        }
        return descriptions


def calculate_time_features(
    data: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    便捷函数：计算时间特征
    
    Args:
        data: 包含时间戳的DataFrame
        timestamp_col: 时间戳列名
        
    Returns:
        包含时间特征的DataFrame
    """
    calculator = TimeFeatureCalculator()
    return calculator.calculate_all_features(data, timestamp_col)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建示例数据 - 覆盖一整天的不同时间
    dates = pd.date_range('2023-01-01 00:00', periods=288, freq='5min')  # 一天288个5分钟
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'close': 100 + np.random.randn(288) * 2
    })
    
    # 计算时间特征
    calculator = TimeFeatureCalculator()
    result = calculator.calculate_all_features(
        sample_data,
        timestamp_col='timestamp',
        include_dow=True,
        include_month=True
    )
    
    # 显示结果
    print("\n=== 时间特征计算示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print(f"结果数据形状: {result.shape}")
    
    print("\n特征列表:")
    feature_names = calculator.get_feature_names(include_dow=True, include_month=True)
    descriptions = calculator.get_feature_descriptions()
    for i, feature in enumerate(feature_names, 1):
        desc = descriptions.get(feature, "无描述")
        print(f"{i}. {feature}: {desc}")
    
    print("\n不同时间点的特征值:")
    # 显示一天中不同时间的编码
    sample_times = [0, 6, 12, 18, 23]  # 0点, 6点, 12点, 18点, 23点
    for hour in sample_times:
        idx = hour * 12  # 每小时12个5分钟
        if idx < len(result):
            row = result.iloc[idx]
            print(f"\n时间 {row['timestamp'].strftime('%H:%M')}:")
            print(f"  sin_tod: {row['sin_tod']:.4f}")
            print(f"  cos_tod: {row['cos_tod']:.4f}")
    
    print("\n特征统计:")
    feature_cols = calculator.get_feature_names()
    print(result[feature_cols].describe())
    
    print("\n缺失值统计:")
    print(result[feature_cols].isna().sum())
    
    # 验证周期性
    print("\n验证周期性（0点和24点应该相同）:")
    print(f"0点: sin={result.iloc[0]['sin_tod']:.4f}, cos={result.iloc[0]['cos_tod']:.4f}")
    print(f"23:55: sin={result.iloc[-1]['sin_tod']:.4f}, cos={result.iloc[-1]['cos_tod']:.4f}")
    print(f"差异: sin_diff={abs(result.iloc[0]['sin_tod'] - result.iloc[-1]['sin_tod']):.6f}")