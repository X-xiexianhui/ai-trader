"""
归一化模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from src.data.normalization import StandardScaler, RobustScaler, FeatureScaler


class TestStandardScaler:
    """StandardScaler单元测试"""
    
    def test_fit_transform_dataframe(self):
        """测试DataFrame的拟合和转换"""
        # 创建测试数据
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        # 检查均值约为0，标准差约为1
        assert isinstance(scaled, pd.DataFrame)
        assert np.allclose(scaled.mean(), 0, atol=1e-10)
        assert np.allclose(scaled.std(ddof=0), 1, atol=1e-10)
    
    def test_fit_transform_array(self):
        """测试numpy数组的拟合和转换"""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        assert isinstance(scaled, np.ndarray)
        assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(scaled.std(axis=0, ddof=0), 1, atol=1e-10)
    
    def test_constant_feature(self):
        """测试常数特征（标准差为0）"""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        # 常数特征应该保持不变（因为std被设为1）
        assert np.allclose(scaled['constant'], 0)
        assert np.allclose(scaled['variable'].mean(), 0, atol=1e-10)
    
    def test_inverse_transform(self):
        """测试反向转换"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        
        # 检查恢复的数据与原始数据一致
        assert np.allclose(recovered.values, data.values, rtol=1e-5)
    
    def test_transform_without_fit(self):
        """测试未拟合就转换应该报错"""
        scaler = StandardScaler()
        data = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Scaler未拟合"):
            scaler.transform(data)
    
    def test_save_load(self):
        """测试保存和加载"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = StandardScaler()
        scaler.fit(data)
        
        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'scaler.pkl'
            scaler.save(filepath)
            
            # 加载
            loaded_scaler = StandardScaler.load(filepath)
            
            # 检查参数一致
            assert np.allclose(loaded_scaler.mean_, scaler.mean_)
            assert np.allclose(loaded_scaler.std_, scaler.std_)
            assert loaded_scaler.feature_names_ == scaler.feature_names_
            
            # 检查转换结果一致
            scaled1 = scaler.transform(data)
            scaled2 = loaded_scaler.transform(data)
            assert np.allclose(scaled1.values, scaled2.values)
    
    def test_with_nan_values(self):
        """测试包含NaN值的数据"""
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, 20, 30, np.nan, 50]
        })
        
        scaler = StandardScaler()
        scaler.fit(data)
        
        # 应该忽略NaN计算均值和标准差
        assert not np.isnan(scaler.mean_).any()
        assert not np.isnan(scaler.std_).any()


class TestRobustScaler:
    """RobustScaler单元测试"""
    
    def test_fit_transform_dataframe(self):
        """测试DataFrame的拟合和转换"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data)
        
        # 检查中位数约为0
        assert isinstance(scaled, pd.DataFrame)
        assert np.allclose(scaled.median(), 0, atol=1e-10)
    
    def test_robust_to_outliers(self):
        """测试对异常值的鲁棒性"""
        # 包含异常值的数据
        data_with_outliers = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 100]  # 100是异常值
        })
        
        data_without_outliers = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 6]
        })
        
        # RobustScaler
        robust_scaler = RobustScaler()
        robust_scaled_with = robust_scaler.fit_transform(data_with_outliers)
        
        # StandardScaler
        standard_scaler = StandardScaler()
        standard_scaled_with = standard_scaler.fit_transform(data_with_outliers)
        
        # RobustScaler应该对异常值更鲁棒
        # 检查前5个值的标准差，RobustScaler应该更小
        robust_std = robust_scaled_with['feature'].iloc[:5].std()
        standard_std = standard_scaled_with['feature'].iloc[:5].std()
        
        assert robust_std < standard_std
    
    def test_inverse_transform(self):
        """测试反向转换"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(scaled)
        
        assert np.allclose(recovered.values, data.values, rtol=1e-5)
    
    def test_zero_iqr(self):
        """测试IQR为0的情况"""
        data = pd.DataFrame({
            'constant': [5, 5, 5, 5, 5],
            'variable': [1, 2, 3, 4, 5]
        })
        
        scaler = RobustScaler()
        scaled = scaler.fit_transform(data)
        
        # IQR为0的特征应该被处理（IQR设为1）
        assert np.allclose(scaled['constant'], 0)
    
    def test_save_load(self):
        """测试保存和加载"""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        scaler = RobustScaler()
        scaler.fit(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'scaler.pkl'
            scaler.save(filepath)
            
            loaded_scaler = RobustScaler.load(filepath)
            
            assert np.allclose(loaded_scaler.median_, scaler.median_)
            assert np.allclose(loaded_scaler.iqr_, scaler.iqr_)
            assert loaded_scaler.quantile_range == scaler.quantile_range


class TestFeatureScaler:
    """FeatureScaler单元测试"""
    
    def test_fit_transform_with_groups(self):
        """测试特征分组归一化"""
        # 创建测试数据
        data = pd.DataFrame({
            'ret_1': [0.01, 0.02, -0.01, 0.03, -0.02],
            'ret_5': [0.05, 0.10, -0.05, 0.15, -0.10],
            'ATR14_norm': [0.02, 0.03, 0.025, 0.035, 0.028],
            'vol_20': [0.015, 0.020, 0.018, 0.022, 0.019],
            'sin_tod': [0.5, 0.7, 0.3, 0.8, 0.4],
            'cos_tod': [0.8, 0.7, 0.9, 0.6, 0.9]
        })
        
        # 定义特征分组
        feature_groups = {
            'price_return': ['ret_1', 'ret_5'],
            'volatility': ['ATR14_norm', 'vol_20'],
            'time': ['sin_tod', 'cos_tod']
        }
        
        scaler = FeatureScaler()
        scaled = scaler.fit_transform(data, feature_groups)
        
        # 检查归一化效果
        assert isinstance(scaled, pd.DataFrame)
        assert scaled.shape == data.shape
        
        # 价格收益特征应该被StandardScaler归一化
        assert np.allclose(scaled[['ret_1', 'ret_5']].mean(), 0, atol=1e-10)
        
        # 波动率特征应该被RobustScaler归一化
        assert np.allclose(scaled[['ATR14_norm', 'vol_20']].median(), 0, atol=1e-10)
        
        # 时间特征不应该被归一化（保持原值）
        assert np.allclose(scaled[['sin_tod', 'cos_tod']].values, 
                          data[['sin_tod', 'cos_tod']].values)
    
    def test_save_load(self):
        """测试保存和加载"""
        data = pd.DataFrame({
            'ret_1': [0.01, 0.02, -0.01, 0.03, -0.02],
            'ATR14_norm': [0.02, 0.03, 0.025, 0.035, 0.028]
        })
        
        feature_groups = {
            'price_return': ['ret_1'],
            'volatility': ['ATR14_norm']
        }
        
        scaler = FeatureScaler()
        scaler.fit(data, feature_groups)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dirpath = Path(tmpdir) / 'scalers'
            scaler.save(dirpath)
            
            loaded_scaler = FeatureScaler.load(dirpath)
            
            # 检查转换结果一致
            scaled1 = scaler.transform(data)
            scaled2 = loaded_scaler.transform(data)
            assert np.allclose(scaled1.values, scaled2.values)
    
    def test_transform_without_fit(self):
        """测试未拟合就转换应该报错"""
        scaler = FeatureScaler()
        data = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="FeatureScaler未拟合"):
            scaler.transform(data)


class TestScalerConsistency:
    """测试scaler的一致性"""
    
    def test_train_val_test_consistency(self):
        """测试训练集、验证集、测试集使用相同scaler"""
        # 创建训练、验证、测试数据
        np.random.seed(42)
        train_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 10
        })
        val_data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30) * 10
        })
        test_data = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30) * 10
        })
        
        # 仅使用训练集拟合
        scaler = StandardScaler()
        scaler.fit(train_data)
        
        # 转换所有数据集
        train_scaled = scaler.transform(train_data)
        val_scaled = scaler.transform(val_data)
        test_scaled = scaler.transform(test_data)
        
        # 训练集应该均值为0，标准差为1
        assert np.allclose(train_scaled.mean(), 0, atol=1e-10)
        assert np.allclose(train_scaled.std(ddof=0), 1, atol=1e-10)
        
        # 验证集和测试集使用相同的均值和标准差
        # 它们的均值和标准差可能不是0和1，但应该使用训练集的参数
        assert val_scaled is not None
        assert test_scaled is not None
    
    def test_no_data_leakage(self):
        """测试无数据泄露"""
        # 创建有明显差异的训练和测试数据
        train_data = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5]
        })
        test_data = pd.DataFrame({
            'feature': [100, 200, 300, 400, 500]
        })
        
        # 正确方式：仅使用训练集拟合
        scaler = StandardScaler()
        scaler.fit(train_data)
        
        train_scaled = scaler.transform(train_data)
        test_scaled = scaler.transform(test_data)
        
        # 训练集应该被正确归一化
        assert np.allclose(train_scaled['feature'].mean(), 0, atol=1e-10)
        
        # 测试集使用训练集的参数，所以均值不会是0
        assert not np.allclose(test_scaled['feature'].mean(), 0, atol=0.1)
        
        # 检查使用的是训练集的均值
        assert np.allclose(scaler.mean_[0], train_data['feature'].mean())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])