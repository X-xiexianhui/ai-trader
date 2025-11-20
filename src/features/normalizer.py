"""
特征归一化模块

实现特征归一化功能:
- StandardScaler归一化 (12维特征)
- RobustScaler归一化 (13维特征)
- 支持scaler保存和加载
- 避免look-ahead bias

Author: AI Trader Team
Date: 2025-11-20
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    特征归一化器
    
    根据特征类型使用不同的归一化方法:
    - StandardScaler: 适用于正态分布的特征
    - RobustScaler: 适用于有异常值的特征
    """
    
    def __init__(self):
        """初始化特征归一化器"""
        self.scalers = {}
        self.feature_groups = {
            'standard': [],  # 使用StandardScaler的特征
            'robust': []     # 使用RobustScaler的特征
        }
        self.is_fitted = False
        logger.info("特征归一化器初始化完成")
    
    def _standard_scaler_fit(
        self,
        data: pd.Series
    ) -> Dict[str, float]:
        """
        拟合StandardScaler
        
        StandardScaler: (x - mean) / std
        
        Args:
            data: 特征数据
            
        Returns:
            包含mean和std的字典
        """
        mean = data.mean()
        std = data.std()
        
        # 避免除零
        if std == 0 or np.isnan(std):
            std = 1.0
        
        return {'mean': mean, 'std': std}
    
    def _standard_scaler_transform(
        self,
        data: pd.Series,
        params: Dict[str, float]
    ) -> pd.Series:
        """
        应用StandardScaler变换
        
        Args:
            data: 特征数据
            params: scaler参数
            
        Returns:
            归一化后的数据
        """
        return (data - params['mean']) / params['std']
    
    def _robust_scaler_fit(
        self,
        data: pd.Series
    ) -> Dict[str, float]:
        """
        拟合RobustScaler
        
        RobustScaler: (x - median) / IQR
        IQR = Q3 - Q1 (四分位距)
        
        Args:
            data: 特征数据
            
        Returns:
            包含median和iqr的字典
        """
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # 避免除零
        if iqr == 0 or np.isnan(iqr):
            iqr = 1.0
        
        return {'median': median, 'iqr': iqr, 'q1': q1, 'q3': q3}
    
    def _robust_scaler_transform(
        self,
        data: pd.Series,
        params: Dict[str, float]
    ) -> pd.Series:
        """
        应用RobustScaler变换
        
        Args:
            data: 特征数据
            params: scaler参数
            
        Returns:
            归一化后的数据
        """
        return (data - params['median']) / params['iqr']
    
    def fit(
        self,
        data: pd.DataFrame,
        standard_features: Optional[List[str]] = None,
        robust_features: Optional[List[str]] = None
    ) -> 'FeatureNormalizer':
        """
        拟合归一化器
        
        Args:
            data: 包含特征的DataFrame
            standard_features: 使用StandardScaler的特征列表
            robust_features: 使用RobustScaler的特征列表
            
        Returns:
            self
        """
        logger.info(f"开始拟合归一化器，数据长度: {len(data)}")
        
        # 默认特征分组（根据设计文档）
        if standard_features is None:
            standard_features = [
                'ret_1', 'ret_5', 'ret_20',
                'vol_20', 'MACD',
                'volume_zscore', 'volume_change_1',
                'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
                'sin_tod', 'cos_tod'
            ]
        
        if robust_features is None:
            robust_features = [
                'price_slope_20', 'C_div_MA20',
                'ATR14_norm', 'range_20_norm', 'BB_width_norm', 'parkinson_vol',
                'EMA20', 'stoch', 'VWAP',
                'volume', 'OBV_slope_20',
                'pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm', 'FVG'
            ]
        
        self.feature_groups['standard'] = standard_features
        self.feature_groups['robust'] = robust_features
        
        # 拟合StandardScaler特征
        for feature in standard_features:
            if feature in data.columns:
                # 只使用非NaN值拟合
                valid_data = data[feature].dropna()
                if len(valid_data) > 0:
                    self.scalers[feature] = {
                        'type': 'standard',
                        'params': self._standard_scaler_fit(valid_data)
                    }
                    logger.debug(f"拟合 StandardScaler: {feature}")
                else:
                    logger.warning(f"特征 {feature} 没有有效数据，跳过")
            else:
                logger.warning(f"特征 {feature} 不在数据中，跳过")
        
        # 拟合RobustScaler特征
        for feature in robust_features:
            if feature in data.columns:
                valid_data = data[feature].dropna()
                if len(valid_data) > 0:
                    self.scalers[feature] = {
                        'type': 'robust',
                        'params': self._robust_scaler_fit(valid_data)
                    }
                    logger.debug(f"拟合 RobustScaler: {feature}")
                else:
                    logger.warning(f"特征 {feature} 没有有效数据，跳过")
            else:
                logger.warning(f"特征 {feature} 不在数据中，跳过")
        
        self.is_fitted = True
        logger.info(f"归一化器拟合完成，共 {len(self.scalers)} 个特征")
        
        return self
    
    def transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        应用归一化变换
        
        Args:
            data: 包含特征的DataFrame
            
        Returns:
            归一化后的DataFrame
            
        Raises:
            ValueError: 如果归一化器未拟合
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，请先调用 fit() 方法")
        
        result = data.copy()
        
        # 应用归一化
        for feature, scaler_info in self.scalers.items():
            if feature in result.columns:
                scaler_type = scaler_info['type']
                params = scaler_info['params']
                
                if scaler_type == 'standard':
                    result[feature] = self._standard_scaler_transform(
                        result[feature],
                        params
                    )
                elif scaler_type == 'robust':
                    result[feature] = self._robust_scaler_transform(
                        result[feature],
                        params
                    )
                
                logger.debug(f"归一化特征: {feature} ({scaler_type})")
        
        logger.info(f"特征归一化完成，共处理 {len(self.scalers)} 个特征")
        
        return result
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        standard_features: Optional[List[str]] = None,
        robust_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        拟合并应用归一化
        
        Args:
            data: 包含特征的DataFrame
            standard_features: 使用StandardScaler的特征列表
            robust_features: 使用RobustScaler的特征列表
            
        Returns:
            归一化后的DataFrame
        """
        self.fit(data, standard_features, robust_features)
        return self.transform(data)
    
    def inverse_transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        反归一化
        
        Args:
            data: 归一化后的DataFrame
            
        Returns:
            原始尺度的DataFrame
            
        Raises:
            ValueError: 如果归一化器未拟合
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，无法进行反归一化")
        
        result = data.copy()
        
        for feature, scaler_info in self.scalers.items():
            if feature in result.columns:
                scaler_type = scaler_info['type']
                params = scaler_info['params']
                
                if scaler_type == 'standard':
                    # 反StandardScaler: x_original = x_scaled * std + mean
                    result[feature] = result[feature] * params['std'] + params['mean']
                elif scaler_type == 'robust':
                    # 反RobustScaler: x_original = x_scaled * iqr + median
                    result[feature] = result[feature] * params['iqr'] + params['median']
        
        logger.info(f"反归一化完成，共处理 {len(self.scalers)} 个特征")
        
        return result
    
    def save(
        self,
        filepath: str
    ) -> None:
        """
        保存归一化器参数
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合，无法保存")
        
        save_data = {
            'scalers': self.scalers,
            'feature_groups': self.feature_groups,
            'is_fitted': self.is_fitted
        }
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"归一化器已保存到: {filepath}")
    
    def load(
        self,
        filepath: str
    ) -> 'FeatureNormalizer':
        """
        加载归一化器参数
        
        Args:
            filepath: 加载路径
            
        Returns:
            self
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.scalers = save_data['scalers']
        self.feature_groups = save_data['feature_groups']
        self.is_fitted = save_data['is_fitted']
        
        logger.info(f"归一化器已从 {filepath} 加载，共 {len(self.scalers)} 个特征")
        
        return self
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        获取特征归一化统计信息
        
        Returns:
            包含特征统计的DataFrame
        """
        if not self.is_fitted:
            raise ValueError("归一化器未拟合")
        
        stats = []
        for feature, scaler_info in self.scalers.items():
            scaler_type = scaler_info['type']
            params = scaler_info['params']
            
            stat = {'feature': feature, 'scaler_type': scaler_type}
            stat.update(params)
            stats.append(stat)
        
        return pd.DataFrame(stats)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        # StandardScaler特征（正态分布）
        'ret_1': np.random.randn(n_samples) * 0.01,
        'vol_20': np.abs(np.random.randn(n_samples)) * 0.02,
        'sin_tod': np.sin(np.linspace(0, 4*np.pi, n_samples)),
        
        # RobustScaler特征（有异常值）
        'price_slope_20': np.random.randn(n_samples) * 0.5,
        'ATR14_norm': np.abs(np.random.randn(n_samples)) * 0.03,
        'volume': np.random.lognormal(10, 1, n_samples)  # 对数正态分布，有异常值
    })
    
    # 添加一些异常值
    sample_data.loc[np.random.choice(n_samples, 10), 'volume'] *= 10
    
    print("\n=== 特征归一化示例 ===")
    print(f"\n原始数据形状: {sample_data.shape}")
    print("\n原始数据统计:")
    print(sample_data.describe())
    
    # 创建归一化器
    normalizer = FeatureNormalizer()
    
    # 拟合并转换
    normalized_data = normalizer.fit_transform(
        sample_data,
        standard_features=['ret_1', 'vol_20', 'sin_tod'],
        robust_features=['price_slope_20', 'ATR14_norm', 'volume']
    )
    
    print("\n归一化后数据统计:")
    print(normalized_data.describe())
    
    # 显示归一化参数
    print("\n归一化参数:")
    print(normalizer.get_feature_stats())
    
    # 测试反归一化
    denormalized_data = normalizer.inverse_transform(normalized_data)
    
    print("\n反归一化后数据统计:")
    print(denormalized_data.describe())
    
    # 验证反归一化的准确性
    print("\n反归一化误差:")
    for col in sample_data.columns:
        error = np.abs(sample_data[col] - denormalized_data[col]).mean()
        print(f"{col}: {error:.10f}")
    
    # 测试保存和加载
    save_path = "scalers/test_normalizer.pkl"
    normalizer.save(save_path)
    
    # 创建新的归一化器并加载
    new_normalizer = FeatureNormalizer()
    new_normalizer.load(save_path)
    
    # 使用加载的归一化器转换数据
    normalized_data_2 = new_normalizer.transform(sample_data)
    
    print("\n加载后的归一化器结果一致性检查:")
    for col in sample_data.columns:
        diff = np.abs(normalized_data[col] - normalized_data_2[col]).max()
        print(f"{col}: 最大差异 = {diff:.10f}")