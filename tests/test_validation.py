"""
特征验证模块单元测试
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.evaluation import FeatureValidator


class TestFeatureValidator:
    """FeatureValidator单元测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 200
        
        # 创建特征
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),  # 随机特征
            'feature2': np.random.randn(n_samples) * 2,  # 更大方差
            'feature3': np.random.randn(n_samples) * 0.5,  # 更小方差
            'feature4': np.random.randn(n_samples),  # 与target相关
            'feature5': np.random.randn(n_samples),  # 与feature1高度相关
        })
        
        # feature5与feature1高度相关
        X['feature5'] = X['feature1'] * 0.9 + np.random.randn(n_samples) * 0.1
        
        # 创建目标变量（与feature4相关）
        y = pd.Series(X['feature4'] * 2 + np.random.randn(n_samples) * 0.5)
        
        return X, y
    
    def test_single_feature_information(self, sample_data):
        """测试单特征信息量测试"""
        X, y = sample_data
        
        validator = FeatureValidator()
        results = validator.test_single_feature_information(X, y, top_n=5)
        
        # 检查返回结果
        assert isinstance(results, pd.DataFrame)
        assert len(results) <= 5
        assert 'feature' in results.columns
        assert 'r2_score' in results.columns
        assert 'mutual_info' in results.columns
        assert 'combined_score' in results.columns
        
        # feature4应该有最高的信息量（因为与y相关）
        top_feature = results.iloc[0]['feature']
        assert top_feature == 'feature4'
        
        # R²和互信息应该都是正数
        assert (results['r2_score'] >= 0).all()
        assert (results['mutual_info'] >= 0).all()
    
    def test_permutation_importance(self, sample_data):
        """测试置换重要性测试"""
        X, y = sample_data
        
        # 训练一个简单模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        validator = FeatureValidator()
        results = validator.test_permutation_importance(
            model, X, y, n_repeats=20, random_state=42
        )
        
        # 检查返回结果
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(X.columns)
        assert 'feature' in results.columns
        assert 'importance' in results.columns
        assert 'std' in results.columns
        assert 'p_value' in results.columns
        assert 'is_significant' in results.columns
        
        # feature4应该是最重要的
        top_feature = results.iloc[0]['feature']
        assert top_feature == 'feature4'
        
        # 应该有显著特征
        assert results['is_significant'].sum() > 0
    
    def test_feature_correlation(self, sample_data):
        """测试特征相关性检测"""
        X, y = sample_data
        
        validator = FeatureValidator()
        corr_matrix, high_corr_pairs = validator.test_feature_correlation(
            X, threshold=0.85, plot=False
        )
        
        # 检查相关矩阵
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (len(X.columns), len(X.columns))
        
        # 对角线应该是1
        assert np.allclose(np.diag(corr_matrix.values), 1.0)
        
        # 应该检测到feature1和feature5高度相关
        assert len(high_corr_pairs) > 0
        
        # 检查高相关对的格式
        for feat1, feat2, corr in high_corr_pairs:
            assert isinstance(feat1, str)
            assert isinstance(feat2, str)
            assert 0 <= corr <= 1
            assert corr > 0.85
    
    def test_vif_multicollinearity(self, sample_data):
        """测试VIF多重共线性检测"""
        X, y = sample_data
        
        validator = FeatureValidator()
        vif_df = validator.test_vif_multicollinearity(X, threshold=10.0)
        
        # 检查返回结果
        assert isinstance(vif_df, pd.DataFrame)
        assert len(vif_df) == len(X.columns)
        assert 'feature' in vif_df.columns
        assert 'VIF' in vif_df.columns
        assert 'has_multicollinearity' in vif_df.columns
        
        # VIF应该都是正数
        assert (vif_df['VIF'].dropna() > 0).all()
    
    def test_generate_validation_report(self, sample_data):
        """测试生成验证报告"""
        X, y = sample_data
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        validator = FeatureValidator()
        
        # 运行所有验证
        validator.test_single_feature_information(X, y)
        validator.test_permutation_importance(model, X, y, n_repeats=10)
        validator.test_feature_correlation(X, plot=False)
        validator.test_vif_multicollinearity(X)
        
        # 生成报告
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            report_path = f.name
        
        results = validator.generate_validation_report(report_path)
        
        # 检查报告内容
        assert isinstance(results, dict)
        assert 'single_feature_info' in results
        assert 'permutation_importance' in results
        assert 'correlation_matrix' in results
        assert 'high_corr_pairs' in results
        assert 'vif' in results
        
        # 检查报告文件存在
        import os
        assert os.path.exists(report_path)
        
        # 清理
        os.remove(report_path)
    
    def test_suggest_feature_removal(self, sample_data):
        """测试特征移除建议"""
        X, y = sample_data
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        validator = FeatureValidator()
        
        # 运行验证
        validator.test_permutation_importance(model, X, y, n_repeats=10)
        validator.test_feature_correlation(X, threshold=0.85, plot=False)
        validator.test_vif_multicollinearity(X, threshold=10.0)
        
        # 获取移除建议
        features_to_remove = validator.suggest_feature_removal(
            importance_threshold=0.001,
            corr_threshold=0.85,
            vif_threshold=10.0
        )
        
        # 应该建议移除一些特征
        assert isinstance(features_to_remove, list)
        
        # feature1和feature5高度相关，应该移除其中一个
        assert 'feature1' in features_to_remove or 'feature5' in features_to_remove


class TestFeatureValidationWithRealData:
    """使用更真实的数据测试特征验证"""
    
    @pytest.fixture
    def realistic_features(self):
        """创建更真实的金融特征数据"""
        np.random.seed(42)
        n_samples = 500
        
        # 模拟价格数据
        price = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
        
        X = pd.DataFrame({
            # 收益率特征
            'ret_1': np.diff(np.log(price), prepend=np.log(price[0])),
            'ret_5': pd.Series(np.log(price)).diff(5).fillna(0),
            
            # 波动率特征
            'vol_20': pd.Series(np.log(price)).diff().rolling(20).std().fillna(0),
            'atr_norm': np.random.uniform(0.01, 0.05, n_samples),
            
            # 技术指标
            'ma_ratio': price / pd.Series(price).rolling(20).mean().fillna(price),
            'rsi': np.random.uniform(20, 80, n_samples),
            
            # 成交量特征
            'volume': np.random.lognormal(10, 1, n_samples),
            'volume_change': np.random.randn(n_samples) * 0.1,
        })
        
        # 目标：未来收益
        y = pd.Series(X['ret_1']).shift(-5).fillna(0)
        
        return X, y
    
    def test_realistic_validation_workflow(self, realistic_features):
        """测试完整的验证工作流"""
        X, y = realistic_features
        
        # 训练模型
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        validator = FeatureValidator()
        
        # 1. 单特征信息量
        info_results = validator.test_single_feature_information(X, y, top_n=5)
        assert len(info_results) > 0
        
        # 2. 置换重要性
        perm_results = validator.test_permutation_importance(
            model, X, y, n_repeats=20
        )
        assert len(perm_results) == len(X.columns)
        
        # 3. 相关性检测
        corr_matrix, high_corr_pairs = validator.test_feature_correlation(
            X, threshold=0.8, plot=False
        )
        assert corr_matrix.shape[0] == len(X.columns)
        
        # 4. VIF检测
        vif_df = validator.test_vif_multicollinearity(X, threshold=10.0)
        assert len(vif_df) == len(X.columns)
        
        # 5. 生成报告
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            report_path = f.name
        
        validator.generate_validation_report(report_path)
        
        # 6. 获取移除建议
        features_to_remove = validator.suggest_feature_removal()
        
        # 清理
        import os
        os.remove(report_path)
        
        # 验证工作流完成
        assert True


class TestEdgeCases:
    """测试边界情况"""
    
    def test_with_constant_feature(self):
        """测试包含常数特征"""
        X = pd.DataFrame({
            'constant': [5] * 100,
            'variable': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        validator = FeatureValidator()
        
        # 应该能处理常数特征
        results = validator.test_single_feature_information(X, y)
        assert len(results) > 0
    
    def test_with_missing_values(self):
        """测试包含缺失值"""
        X = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5] * 20,
            'feature2': [10, np.nan, 30, 40, 50] * 20
        })
        y = pd.Series(np.random.randn(100))
        
        validator = FeatureValidator()
        
        # 应该能处理缺失值
        results = validator.test_single_feature_information(X, y)
        assert len(results) > 0
    
    def test_with_perfect_correlation(self):
        """测试完全相关的特征"""
        X = pd.DataFrame({
            'feature1': np.arange(100),
            'feature2': np.arange(100) * 2,  # 完全线性相关
            'feature3': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        validator = FeatureValidator()
        
        # 应该检测到完全相关
        corr_matrix, high_corr_pairs = validator.test_feature_correlation(
            X, threshold=0.9, plot=False
        )
        
        assert len(high_corr_pairs) > 0
        # feature1和feature2应该完全相关
        assert any(
            (feat1 == 'feature1' and feat2 == 'feature2') or
            (feat1 == 'feature2' and feat2 == 'feature1')
            for feat1, feat2, _ in high_corr_pairs
        )
    
    def test_with_small_dataset(self):
        """测试小数据集"""
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        y = pd.Series([1, 2, 3, 4, 5])
        
        validator = FeatureValidator()
        
        # 应该能处理小数据集
        results = validator.test_single_feature_information(X, y)
        assert len(results) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])