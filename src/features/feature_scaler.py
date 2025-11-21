"""
特征归一化模块 - StandardScaler和RobustScaler实现
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class StandardScaler:
    """
    标准化归一化器 - 用于收益率、价格斜率等特征
    
    任务1.3.1实现：
    - 仅使用训练集计算均值和标准差
    - 应用z-score标准化：z = (x - μ) / σ
    - 支持保存和加载
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            feature_names: Optional[list] = None) -> 'StandardScaler':
        """
        使用训练集计算均值和标准差
        
        Args:
            X: 训练集特征
            feature_names: 特征名称列表
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = feature_names
        
        # 计算均值和标准差
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        
        # 处理常数特征（标准差为0）
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        
        self.is_fitted_ = True
        logger.info(f"StandardScaler已拟合: {X.shape[1]}个特征")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        应用标准化转换
        
        Args:
            X: 待转换的特征
            
        Returns:
            标准化后的特征
        """
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合，请先调用fit()方法")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        columns = X.columns if is_dataframe else None
        
        if is_dataframe:
            X_array = X.values
        else:
            X_array = X
        
        # 应用z-score标准化
        X_scaled = (X_array - self.mean_) / self.std_
        
        # 返回相同类型
        if is_dataframe:
            return pd.DataFrame(X_scaled, index=index, columns=columns)
        else:
            return X_scaled
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                     feature_names: Optional[list] = None) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换"""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """反向转换"""
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        columns = X.columns if is_dataframe else None
        
        if is_dataframe:
            X_array = X.values
        else:
            X_array = X
        
        # 反向转换
        X_original = X_array * self.std_ + self.mean_
        
        if is_dataframe:
            return pd.DataFrame(X_original, index=index, columns=columns)
        else:
            return X_original
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        任务1.3.3: 保存scaler到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合，无法保存")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        scaler_data = {
            'mean': self.mean_,
            'std': self.std_,
            'feature_names': self.feature_names_,
            'scaler_type': 'StandardScaler'
        }
        
        joblib.dump(scaler_data, filepath)
        logger.info(f"StandardScaler已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'StandardScaler':
        """
        任务1.3.3: 从文件加载scaler
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的scaler对象
        """
        scaler_data = joblib.load(filepath)
        
        if scaler_data.get('scaler_type') != 'StandardScaler':
            raise ValueError(f"文件不是StandardScaler类型")
        
        scaler = cls()
        scaler.mean_ = scaler_data['mean']
        scaler.std_ = scaler_data['std']
        scaler.feature_names_ = scaler_data['feature_names']
        scaler.is_fitted_ = True
        
        logger.info(f"StandardScaler已从{filepath}加载")
        
        return scaler


class RobustScaler:
    """
    鲁棒归一化器 - 用于波动率、技术指标等对异常值敏感的特征
    
    任务1.3.2实现：
    - 仅使用训练集计算中位数和IQR
    - 应用鲁棒标准化：z = (x - median) / IQR
    - 支持保存和加载
    """
    
    def __init__(self, quantile_range: tuple = (25.0, 75.0)):
        """
        初始化RobustScaler
        
        Args:
            quantile_range: 用于计算IQR的分位数范围
        """
        self.quantile_range = quantile_range
        self.median_ = None
        self.iqr_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            feature_names: Optional[list] = None) -> 'RobustScaler':
        """
        使用训练集计算中位数和IQR
        
        Args:
            X: 训练集特征
            feature_names: 特征名称列表
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = feature_names
        
        # 计算中位数
        self.median_ = np.nanmedian(X, axis=0)
        
        # 计算IQR
        q1 = np.nanpercentile(X, self.quantile_range[0], axis=0)
        q3 = np.nanpercentile(X, self.quantile_range[1], axis=0)
        self.iqr_ = q3 - q1
        
        # 处理IQR为0的情况
        self.iqr_ = np.where(self.iqr_ == 0, 1.0, self.iqr_)
        
        self.is_fitted_ = True
        logger.info(f"RobustScaler已拟合: {X.shape[1]}个特征")
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        应用鲁棒标准化转换
        
        Args:
            X: 待转换的特征
            
        Returns:
            标准化后的特征
        """
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合，请先调用fit()方法")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        columns = X.columns if is_dataframe else None
        
        if is_dataframe:
            X_array = X.values
        else:
            X_array = X
        
        # 应用鲁棒标准化
        X_scaled = (X_array - self.median_) / self.iqr_
        
        # 返回相同类型
        if is_dataframe:
            return pd.DataFrame(X_scaled, index=index, columns=columns)
        else:
            return X_scaled
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                     feature_names: Optional[list] = None) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换"""
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """反向转换"""
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合")
        
        is_dataframe = isinstance(X, pd.DataFrame)
        index = X.index if is_dataframe else None
        columns = X.columns if is_dataframe else None
        
        if is_dataframe:
            X_array = X.values
        else:
            X_array = X
        
        # 反向转换
        X_original = X_array * self.iqr_ + self.median_
        
        if is_dataframe:
            return pd.DataFrame(X_original, index=index, columns=columns)
        else:
            return X_original
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        任务1.3.3: 保存scaler到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted_:
            raise ValueError("Scaler未拟合，无法保存")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        scaler_data = {
            'median': self.median_,
            'iqr': self.iqr_,
            'quantile_range': self.quantile_range,
            'feature_names': self.feature_names_,
            'scaler_type': 'RobustScaler'
        }
        
        joblib.dump(scaler_data, filepath)
        logger.info(f"RobustScaler已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'RobustScaler':
        """
        任务1.3.3: 从文件加载scaler
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的scaler对象
        """
        scaler_data = joblib.load(filepath)
        
        if scaler_data.get('scaler_type') != 'RobustScaler':
            raise ValueError(f"文件不是RobustScaler类型")
        
        scaler = cls(quantile_range=scaler_data['quantile_range'])
        scaler.median_ = scaler_data['median']
        scaler.iqr_ = scaler_data['iqr']
        scaler.feature_names_ = scaler_data['feature_names']
        scaler.is_fitted_ = True
        
        logger.info(f"RobustScaler已从{filepath}加载")
        
        return scaler


class FeatureScaler:
    """
    特征归一化管理器 - 自动为不同特征组选择合适的scaler
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_groups = {}
        self.is_fitted_ = False
        
    def fit(self, X: pd.DataFrame, feature_groups: dict) -> 'FeatureScaler':
        """
        为不同特征组拟合不同的scaler
        
        Args:
            X: 特征DataFrame
            feature_groups: 特征分组字典
            
        Returns:
            self
        """
        self.feature_groups = feature_groups
        
        # 为不同特征组选择scaler
        # 价格和收益特征使用StandardScaler
        if 'price_return' in feature_groups:
            features = feature_groups['price_return']
            self.scalers['price_return'] = StandardScaler()
            self.scalers['price_return'].fit(X[features])
        
        # 波动率特征使用RobustScaler
        if 'volatility' in feature_groups:
            features = feature_groups['volatility']
            self.scalers['volatility'] = RobustScaler()
            self.scalers['volatility'].fit(X[features])
        
        # 技术指标使用RobustScaler
        if 'technical' in feature_groups:
            features = feature_groups['technical']
            self.scalers['technical'] = RobustScaler()
            self.scalers['technical'].fit(X[features])
        
        # 成交量特征使用RobustScaler
        if 'volume' in feature_groups:
            features = feature_groups['volume']
            self.scalers['volume'] = RobustScaler()
            self.scalers['volume'].fit(X[features])
        
        # K线形态特征使用StandardScaler
        if 'candlestick' in feature_groups:
            features = feature_groups['candlestick']
            self.scalers['candlestick'] = StandardScaler()
            self.scalers['candlestick'].fit(X[features])
        
        # 时间特征不需要归一化（已经在[-1,1]范围内）
        
        self.is_fitted_ = True
        logger.info(f"FeatureScaler已拟合: {len(self.scalers)}个scaler")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用归一化转换"""
        if not self.is_fitted_:
            raise ValueError("FeatureScaler未拟合")
        
        X_scaled = X.copy()
        
        for group_name, scaler in self.scalers.items():
            features = self.feature_groups[group_name]
            X_scaled[features] = scaler.transform(X[features])
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, feature_groups: dict) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(X, feature_groups)
        return self.transform(X)
    
    def save(self, dirpath: Union[str, Path]) -> None:
        """保存所有scaler"""
        if not self.is_fitted_:
            raise ValueError("FeatureScaler未拟合")
        
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # 保存每个scaler
        for group_name, scaler in self.scalers.items():
            filepath = dirpath / f"{group_name}_scaler.pkl"
            scaler.save(filepath)
        
        # 保存特征分组信息
        joblib.dump(self.feature_groups, dirpath / "feature_groups.pkl")
        
        logger.info(f"FeatureScaler已保存到: {dirpath}")
    
    @classmethod
    def load(cls, dirpath: Union[str, Path]) -> 'FeatureScaler':
        """加载所有scaler"""
        dirpath = Path(dirpath)
        
        # 加载特征分组
        feature_groups = joblib.load(dirpath / "feature_groups.pkl")
        
        # 创建FeatureScaler实例
        feature_scaler = cls()
        feature_scaler.feature_groups = feature_groups
        
        # 加载每个scaler
        for group_name in feature_groups.keys():
            if group_name == 'time':  # 时间特征不需要scaler
                continue
            
            filepath = dirpath / f"{group_name}_scaler.pkl"
            if filepath.exists():
                scaler_data = joblib.load(filepath)
                scaler_type = scaler_data.get('scaler_type')
                
                if scaler_type == 'StandardScaler':
                    feature_scaler.scalers[group_name] = StandardScaler.load(filepath)
                elif scaler_type == 'RobustScaler':
                    feature_scaler.scalers[group_name] = RobustScaler.load(filepath)
        
        feature_scaler.is_fitted_ = True
        logger.info(f"FeatureScaler已从{dirpath}加载")
        
        return feature_scaler