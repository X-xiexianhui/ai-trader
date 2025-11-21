"""
特征计算模块单元测试

测试所有27维手工特征的计算正确性、边界处理和数值稳定性
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import FeatureCalculator


class TestFeatureCalculator:
    """特征计算器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试用的样本数据"""
        np.random.seed(42)
        n = 100
        
        # 创建时间索引
        start_date = datetime(2023, 1, 1, 9, 30)
        dates = [start_date + timedelta(minutes=5*i) for i in range(n)]
        
        # 生成模拟OHLC数据
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_price = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 10000, n)
        
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=pd.DatetimeIndex(dates))
        
        return df
    
    @pytest.fixture
    def calculator(self):
        """创建特征计算器实例"""
        return FeatureCalculator()
    
    def test_initialization(self, calculator):
        """测试初始化"""
        assert calculator.feature_names == []
        assert isinstance(calculator, FeatureCalculator)
    
    def test_price_return_features(self, calculator, sample_data):
        """测试价格与收益特征计算（任务1.2.1）"""
        df = calculator.calculate_price_return_features(sample_data)
        
        # 检查特征是否存在
        expected_features = ['ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20']
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查特征名称是否记录
        assert all(f in calculator.feature_names for f in expected_features)
        
        # 检查数值范围合理性
        assert df['ret_1'].abs().max() < 1.0, "ret_1超出合理范围"
        assert df['C_div_MA20'].min() > 0, "C_div_MA20应该为正值"
        
        # 检查边界处理：前20根K线某些特征应该是NaN
        assert pd.isna(df['ret_20'].iloc[0:20]).any(), "前20根K线的ret_20应该有NaN"
        assert pd.isna(df['price_slope_20'].iloc[0:19]).all(), "前19根K线的price_slope_20应该是NaN"
        
        # 检查无inf值
        assert not np.isinf(df[expected_features]).any().any(), "存在inf值"
    
    def test_volatility_features(self, calculator, sample_data):
        """测试波动率特征计算（任务1.2.2）"""
        df = calculator.calculate_volatility_features(sample_data)
        
        # 检查特征是否存在
        expected_features = ['ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol']
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查数值范围合理性
        assert (df['ATR14_norm'].dropna() >= 0).all(), "ATR14_norm应该非负"
        assert (df['vol_20'].dropna() >= 0).all(), "vol_20应该非负"
        assert (df['parkinson_vol'].dropna() >= 0).all(), "parkinson_vol应该非负"
        
        # 检查边界处理
        assert pd.isna(df['vol_20'].iloc[0:19]).all(), "前19根K线的vol_20应该是NaN"
        
        # 检查无inf值
        assert not np.isinf(df[expected_features]).any().any(), "存在inf值"
    
    def test_technical_indicators(self, calculator, sample_data):
        """测试技术指标特征计算（任务1.2.3）"""
        df = calculator.calculate_technical_indicators(sample_data)
        
        # 检查特征是否存在
        expected_features = ['EMA20', 'stoch', 'MACD', 'VWAP']
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查数值范围合理性
        assert (df['stoch'].dropna() >= 0).all() and (df['stoch'].dropna() <= 100).all(), \
            "stoch应该在[0,100]范围内"
        assert (df['EMA20'].dropna() > 0).all(), "EMA20应该为正值"
        
        # 检查无inf值
        assert not np.isinf(df[expected_features]).any().any(), "存在inf值"
    
    def test_volume_features(self, calculator, sample_data):
        """测试成交量特征计算（任务1.2.4）"""
        df = calculator.calculate_volume_features(sample_data)
        
        # 检查特征是否存在
        expected_features = ['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20']
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查数值范围合理性
        assert (df['volume'] >= 0).all(), "volume应该非负"
        assert df['volume_zscore'].dropna().std() > 0, "volume_zscore应该有变化"
        
        # 检查边界处理
        assert pd.isna(df['volume_zscore'].iloc[0:19]).all(), "前19根K线的volume_zscore应该是NaN"
        
        # 检查无inf值
        assert not np.isinf(df[expected_features]).any().any(), "存在inf值"
    
    def test_candlestick_features(self, calculator, sample_data):
        """测试K线形态特征计算（任务1.2.5）"""
        df = calculator.calculate_candlestick_features(sample_data)
        
        # 检查特征是否存在
        expected_features = [
            'pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm',
            'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'
        ]
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查比例特征在[0,1]范围内
        ratio_features = ['body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio']
        for feature in ratio_features:
            valid_values = df[feature].dropna()
            assert (valid_values >= 0).all() and (valid_values <= 1).all(), \
                f"{feature}应该在[0,1]范围内"
        
        # 检查pos_in_range_20在[0,1]范围内
        valid_pos = df['pos_in_range_20'].dropna()
        assert (valid_pos >= 0).all() and (valid_pos <= 1).all(), \
            "pos_in_range_20应该在[0,1]范围内"
        
        # 检查无inf值
        assert not np.isinf(df[expected_features]).any().any(), "存在inf值"
    
    def test_time_features(self, calculator, sample_data):
        """测试时间周期特征计算（任务1.2.6）"""
        df = calculator.calculate_time_features(sample_data)
        
        # 检查特征是否存在
        expected_features = ['sin_tod', 'cos_tod']
        for feature in expected_features:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查数值范围在[-1,1]
        assert (df['sin_tod'] >= -1).all() and (df['sin_tod'] <= 1).all(), \
            "sin_tod应该在[-1,1]范围内"
        assert (df['cos_tod'] >= -1).all() and (df['cos_tod'] <= 1).all(), \
            "cos_tod应该在[-1,1]范围内"
        
        # 检查sin^2 + cos^2 = 1
        sum_squares = df['sin_tod']**2 + df['cos_tod']**2
        assert np.allclose(sum_squares, 1.0, atol=1e-10), "sin^2 + cos^2应该等于1"
    
    def test_fvg_calculation(self, calculator):
        """测试FVG公允价值缺口计算（任务1.2.7）"""
        # 创建包含明显FVG的测试数据
        dates = pd.date_range('2023-01-01', periods=10, freq='5T')
        
        # 多头FVG示例：第1根K线高点10，第3根K线低点12（有缺口）
        df = pd.DataFrame({
            'Open': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'High': [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5],
            'Low': [9.5, 10.5, 12.0, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5],
            'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Volume': [1000] * 10
        }, index=dates)
        
        fvg = calculator._calculate_fvg(df)
        
        # 检查前两根K线是0
        assert fvg.iloc[0] == 0, "第1根K线FVG应该是0"
        assert fvg.iloc[1] == 0, "第2根K线FVG应该是0"
        
        # 检查第3根K线检测到多头FVG（正值）
        assert fvg.iloc[2] > 0, "第3根K线应该检测到多头FVG"
        
        # 检查无inf值
        assert not np.isinf(fvg).any(), "FVG存在inf值"
    
    def test_calculate_all_features(self, calculator, sample_data):
        """测试计算所有特征"""
        df = calculator.calculate_all_features(sample_data)
        
        # 检查所有27个特征都存在
        assert len(calculator.feature_names) == 27, f"应该有27个特征，实际有{len(calculator.feature_names)}个"
        
        # 检查所有特征列都在DataFrame中
        for feature in calculator.feature_names:
            assert feature in df.columns, f"缺少特征: {feature}"
        
        # 检查数据行数（dropna后应该减少）
        assert len(df) < len(sample_data), "dropna后数据行数应该减少"
        assert len(df) > 0, "不应该删除所有数据"
        
        # 检查无NaN值（因为已经dropna）
        assert not df[calculator.feature_names].isna().any().any(), "不应该有NaN值"
        
        # 检查无inf值
        assert not np.isinf(df[calculator.feature_names]).any().any(), "不应该有inf值"
    
    def test_feature_groups(self, calculator, sample_data):
        """测试特征分组"""
        calculator.calculate_all_features(sample_data)
        groups = calculator.get_feature_groups()
        
        # 检查所有组都存在
        expected_groups = ['price_return', 'volatility', 'technical', 'volume', 'candlestick', 'time']
        for group in expected_groups:
            assert group in groups, f"缺少特征组: {group}"
        
        # 检查每组的特征数量
        assert len(groups['price_return']) == 5, "价格与收益特征应该有5个"
        assert len(groups['volatility']) == 5, "波动率特征应该有5个"
        assert len(groups['technical']) == 4, "技术指标特征应该有4个"
        assert len(groups['volume']) == 4, "成交量特征应该有4个"
        assert len(groups['candlestick']) == 7, "K线形态特征应该有7个"
        assert len(groups['time']) == 2, "时间周期特征应该有2个"
        
        # 检查总数
        total_features = sum(len(features) for features in groups.values())
        assert total_features == 27, f"总特征数应该是27，实际是{total_features}"
    
    def test_edge_cases(self, calculator):
        """测试边界情况"""
        # 测试最小数据量
        dates = pd.date_range('2023-01-01', periods=25, freq='5T')
        df_small = pd.DataFrame({
            'Open': np.random.randn(25) + 100,
            'High': np.random.randn(25) + 101,
            'Low': np.random.randn(25) + 99,
            'Close': np.random.randn(25) + 100,
            'Volume': np.random.randint(1000, 10000, 25)
        }, index=dates)
        
        result = calculator.calculate_all_features(df_small)
        assert len(result) > 0, "小数据集应该能计算特征"
        
    def test_zero_volume_handling(self, calculator, sample_data):
        """测试成交量为0的处理"""
        df = sample_data.copy()
        df.loc[df.index[10:15], 'Volume'] = 0
        
        result = calculator.calculate_volume_features(df)
        
        # 检查不会产生inf
        assert not np.isinf(result['volume_zscore']).any(), "volume_zscore不应该有inf"
        assert not np.isinf(result['volume_change_1']).any(), "volume_change_1不应该有inf"
    
    def test_equal_high_low_handling(self, calculator):
        """测试high=low的处理"""
        dates = pd.date_range('2023-01-01', periods=50, freq='5T')
        df = pd.DataFrame({
            'Open': [100] * 50,
            'High': [100] * 50,  # high = low
            'Low': [100] * 50,
            'Close': [100] * 50,
            'Volume': [1000] * 50
        }, index=dates)
        
        result = calculator.calculate_candlestick_features(df)
        
        # 检查不会产生inf
        assert not np.isinf(result['body_ratio']).any(), "body_ratio不应该有inf"
        assert not np.isinf(result['upper_shadow_ratio']).any(), "upper_shadow_ratio不应该有inf"
        assert not np.isinf(result['lower_shadow_ratio']).any(), "lower_shadow_ratio不应该有inf"
    
    def test_no_future_information(self, calculator, sample_data):
        """测试不引入未来信息"""
        # 计算特征
        df = calculator.calculate_price_return_features(sample_data)
        
        # 检查ret_1只使用历史数据
        for i in range(1, len(df)):
            if not pd.isna(df['ret_1'].iloc[i]):
                expected = np.log(df['Close'].iloc[i] / df['Close'].iloc[i-1])
                actual = df['ret_1'].iloc[i]
                assert np.isclose(expected, actual, rtol=1e-5), \
                    f"ret_1在索引{i}处使用了未来信息"


class TestPerformance:
    """性能测试类"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        import time
        
        # 创建10万条数据
        n = 100000
        dates = pd.date_range('2020-01-01', periods=n, freq='5T')
        df = pd.DataFrame({
            'Open': np.random.randn(n) + 100,
            'High': np.random.randn(n) + 101,
            'Low': np.random.randn(n) + 99,
            'Close': np.random.randn(n) + 100,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        calculator = FeatureCalculator()
        
        start_time = time.time()
        result = calculator.calculate_all_features(df)
        elapsed_time = time.time() - start_time
        
        # 性能要求：10万条数据<5秒
        assert elapsed_time < 5.0, f"性能不达标: {elapsed_time:.2f}秒 > 5秒"
        assert len(result) > 0, "应该返回有效数据"
        
        print(f"\n性能测试通过: 处理{n}条数据耗时{elapsed_time:.2f}秒")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])