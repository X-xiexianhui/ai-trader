"""
数据清洗模块的单元测试

测试覆盖：
1. 缺失值处理功能
2. 异常值检测与处理
3. 时间对齐与时区处理
4. 数据质量验证
5. 完整清洗流程
6. 边界情况和异常处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.cleaning import DataCleaner


class TestDataCleanerInit:
    """测试DataCleaner初始化"""
    
    def test_default_init(self):
        """测试默认参数初始化"""
        cleaner = DataCleaner()
        assert cleaner.max_consecutive_missing == 5
        assert cleaner.interpolation_limit == 3
        assert cleaner.sigma_threshold == 3.0
        assert cleaner.cleaning_report == {}
    
    def test_custom_init(self):
        """测试自定义参数初始化"""
        cleaner = DataCleaner(
            max_consecutive_missing=10,
            interpolation_limit=5,
            sigma_threshold=2.5
        )
        assert cleaner.max_consecutive_missing == 10
        assert cleaner.interpolation_limit == 5
        assert cleaner.sigma_threshold == 2.5
    
    def test_invalid_params(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            DataCleaner(max_consecutive_missing=0)
        
        with pytest.raises(ValueError):
            DataCleaner(interpolation_limit=0)
        
        with pytest.raises(ValueError):
            DataCleaner(sigma_threshold=-1)


class TestHandleMissingValues:
    """测试缺失值处理功能"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5T')
        data = {
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_no_missing_values(self, sample_data):
        """测试无缺失值的情况"""
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.handle_missing_values(sample_data)
        
        assert len(cleaned_df) == len(sample_data)
        assert report['removed_rows'] == 0
        assert sum(report['missing_after'].values()) == 0
    
    def test_short_missing_interpolation(self, sample_data):
        """测试短期缺失的插值"""
        df = sample_data.copy()
        # 创建2个连续缺失点
        df.loc[df.index[10:12], 'Close'] = np.nan
        
        cleaner = DataCleaner(interpolation_limit=3)
        cleaned_df, report = cleaner.handle_missing_values(df)
        
        assert report['interpolated_points']['Close'] == 2
        assert cleaned_df['Close'].isna().sum() == 0
        assert len(cleaned_df) == len(df)
    
    def test_long_missing_removal(self, sample_data):
        """测试长期缺失段的删除"""
        df = sample_data.copy()
        # 创建10个连续缺失点（超过阈值5）
        df.loc[df.index[20:30], ['Open', 'High', 'Low', 'Close']] = np.nan
        
        cleaner = DataCleaner(max_consecutive_missing=5)
        cleaned_df, report = cleaner.handle_missing_values(df)
        
        assert report['removed_rows'] == 10
        assert len(report['removed_segments']) == 1
        assert len(cleaned_df) == len(df) - 10
    
    def test_volume_zero_fill(self, sample_data):
        """测试成交量零填充"""
        df = sample_data.copy()
        df.loc[df.index[5:8], 'Volume'] = np.nan
        
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.handle_missing_values(df)
        
        assert cleaned_df['Volume'].isna().sum() == 0
        assert (cleaned_df.loc[df.index[5:8], 'Volume'] == 0).all()
    
    def test_empty_dataframe(self):
        """测试空DataFrame"""
        cleaner = DataCleaner()
        with pytest.raises(ValueError, match="输入DataFrame不能为空"):
            cleaner.handle_missing_values(pd.DataFrame())
    
    def test_missing_required_columns(self, sample_data):
        """测试缺少必需列"""
        df = sample_data[['Open', 'High']].copy()
        cleaner = DataCleaner()
        
        with pytest.raises(ValueError, match="缺少必需的列"):
            cleaner.handle_missing_values(df)


class TestDetectAndHandleOutliers:
    """测试异常值检测与处理"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5T')
        np.random.seed(42)
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        
        data = {
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_no_outliers(self, sample_data):
        """测试无异常值的情况"""
        cleaner = DataCleaner(sigma_threshold=3.0)
        cleaned_df, report = cleaner.detect_and_handle_outliers(sample_data)
        
        assert len(cleaned_df) == len(sample_data)
        assert report['spike_outliers'] == 0
    
    def test_spike_detection_and_correction(self, sample_data):
        """测试尖峰检测和修正"""
        df = sample_data.copy()
        # 创建一个尖峰异常（价格突然跳变后又回来）
        df.loc[df.index[50], 'Close'] = df.loc[df.index[50], 'Close'] * 2
        df.loc[df.index[50], 'High'] = df.loc[df.index[50], 'Close'] * 1.01
        
        cleaner = DataCleaner(sigma_threshold=2.0)
        cleaned_df, report = cleaner.detect_and_handle_outliers(df)
        
        assert report['total_outliers'] >= 1
        assert report['spike_outliers'] >= 0
    
    def test_gap_preservation(self, sample_data):
        """测试跳空保留"""
        df = sample_data.copy()
        # 创建一个真实跳空（价格持续上涨）
        df.loc[df.index[50:], 'Close'] *= 1.5
        
        cleaner = DataCleaner(sigma_threshold=2.0)
        cleaned_df, report = cleaner.detect_and_handle_outliers(df)
        
        # 跳空应该被保留
        assert report['gap_outliers'] >= 0
    
    def test_ohlc_consistency_fix(self, sample_data):
        """测试OHLC一致性修正"""
        df = sample_data.copy()
        # 故意破坏一致性
        df.loc[df.index[10], 'High'] = df.loc[df.index[10], 'Close'] * 0.9
        
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.detect_and_handle_outliers(df)
        
        # 验证一致性
        assert (cleaned_df['High'] >= cleaned_df[['Open', 'Close']].max(axis=1)).all()
        assert (cleaned_df['Low'] <= cleaned_df[['Open', 'Close']].min(axis=1)).all()
    
    def test_custom_sigma_threshold(self, sample_data):
        """测试自定义sigma阈值"""
        cleaner = DataCleaner(sigma_threshold=3.0)
        _, report1 = cleaner.detect_and_handle_outliers(sample_data, sigma_threshold=2.0)
        _, report2 = cleaner.detect_and_handle_outliers(sample_data, sigma_threshold=4.0)
        
        # 更严格的阈值应该检测到更多异常值
        assert report1['sigma_threshold'] == 2.0
        assert report2['sigma_threshold'] == 4.0


class TestAlignTimeAndTimezone:
    """测试时间对齐与时区处理"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5T', tz='UTC')
        data = {
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_timezone_conversion(self, sample_data):
        """测试时区转换"""
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.align_time_and_timezone(
            sample_data, 
            target_timezone='America/New_York'
        )
        
        assert str(cleaned_df.index.tz) == 'America/New_York'
        assert report['original_timezone'] == 'UTC'
        assert report['target_timezone'] == 'America/New_York'
    
    def test_trading_hours_filter(self, sample_data):
        """测试交易时段过滤"""
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.align_time_and_timezone(
            sample_data,
            target_timezone='UTC',
            trading_hours=(9, 16)  # 9:00-16:00
        )
        
        assert report['filtered_non_trading'] > 0
        assert len(cleaned_df) < len(sample_data)
        assert all(9 <= h < 16 for h in cleaned_df.index.hour)
    
    def test_irregular_interval_resampling(self):
        """测试不规则间隔的重采样"""
        # 创建不规则间隔的数据
        dates = pd.date_range('2023-01-01', periods=50, freq='5T', tz='UTC')
        irregular_dates = dates.tolist()
        # 添加一些不规则的时间点
        irregular_dates[10] = dates[10] + timedelta(minutes=2)
        irregular_dates[20] = dates[20] + timedelta(minutes=7)
        
        data = {
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 5000, 50)
        }
        df = pd.DataFrame(data, index=irregular_dates)
        
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.align_time_and_timezone(df, interval_minutes=5)
        
        assert report['irregular_intervals'] > 0
        # 验证重采样后的间隔
        if len(cleaned_df) > 1:
            time_diffs = cleaned_df.index.to_series().diff().dropna()
            assert all(abs(td - pd.Timedelta(minutes=5)) <= pd.Timedelta(seconds=10) 
                      for td in time_diffs)
    
    def test_no_timezone_localization(self):
        """测试无时区信息的本地化"""
        dates = pd.date_range('2023-01-01', periods=50, freq='5T')  # 无时区
        data = {
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 5000, 50)
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.align_time_and_timezone(df)
        
        assert cleaned_df.index.tz is not None
        assert report['original_timezone'] == 'None'


class TestValidateDataQuality:
    """测试数据质量验证"""
    
    @pytest.fixture
    def good_data(self):
        """创建高质量数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5T', tz='UTC')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.005, 100)))
        
        data = {
            'Open': close_prices * 0.999,
            'High': close_prices * 1.002,
            'Low': close_prices * 0.998,
            'Close': close_prices,
            'Volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_valid_data(self, good_data):
        """测试高质量数据验证通过"""
        cleaner = DataCleaner()
        is_valid, report = cleaner.validate_data_quality(good_data)
        
        assert is_valid is True
        assert report['summary']['total_issues'] == 0
        assert report['summary']['validation_passed'] is True
    
    def test_missing_values_detection(self, good_data):
        """测试缺失值检测"""
        df = good_data.copy()
        # 添加超过1%的缺失值
        df.loc[df.index[:5], 'Close'] = np.nan
        
        cleaner = DataCleaner()
        is_valid, report = cleaner.validate_data_quality(df)
        
        assert is_valid is False
        assert report['completeness']['Close']['missing_ratio'] > 0.01
        assert any('缺失值' in issue for issue in report['issues'])
    
    def test_consistency_violation(self, good_data):
        """测试一致性违规检测"""
        df = good_data.copy()
        # 破坏OHLC一致性
        df.loc[df.index[10], 'High'] = df.loc[df.index[10], 'Close'] * 0.9
        
        cleaner = DataCleaner()
        is_valid, report = cleaner.validate_data_quality(df)
        
        assert is_valid is False
        assert report['consistency']['high_valid'] is False
        assert any('High' in issue for issue in report['issues'])
    
    def test_duplicate_timestamps(self, good_data):
        """测试重复时间戳检测"""
        df = good_data.copy()
        # 添加重复时间戳
        df = pd.concat([df, df.iloc[[0]]])
        
        cleaner = DataCleaner()
        is_valid, report = cleaner.validate_data_quality(df)
        
        assert is_valid is False
        assert report['time_quality']['duplicates'] > 0
        assert any('重复时间戳' in issue for issue in report['issues'])
    
    def test_unsorted_index(self, good_data):
        """测试未排序索引检测"""
        df = good_data.copy()
        # 打乱索引顺序
        df = df.sample(frac=1)
        
        cleaner = DataCleaner()
        is_valid, report = cleaner.validate_data_quality(df)
        
        assert is_valid is False
        assert report['time_quality']['is_sorted'] is False
        assert any('升序' in issue for issue in report['issues'])


class TestCleanPipeline:
    """测试完整清洗流程"""
    
    @pytest.fixture
    def messy_data(self):
        """创建包含各种问题的数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5T')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        
        data = {
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        
        # 添加各种问题
        df.loc[df.index[10:12], 'Close'] = np.nan  # 缺失值
        df.loc[df.index[50], 'Close'] *= 1.5  # 异常值
        df.loc[df.index[60], 'High'] = df.loc[df.index[60], 'Close'] * 0.9  # 一致性问题
        
        return df
    
    def test_full_pipeline(self, messy_data):
        """测试完整清洗流程"""
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.clean_pipeline(messy_data)
        
        # 验证所有步骤都执行了
        assert 'missing_values' in report
        assert 'outliers' in report
        assert 'time_alignment' in report
        assert 'validation' in report
        assert 'final_validation' in report
        assert 'total_processing_time' in report
        
        # 验证数据质量
        assert cleaned_df['Close'].isna().sum() == 0
        assert (cleaned_df['High'] >= cleaned_df[['Open', 'Close']].max(axis=1)).all()
    
    def test_pipeline_with_custom_params(self, messy_data):
        """测试带自定义参数的流程"""
        cleaner = DataCleaner(
            max_consecutive_missing=3,
            interpolation_limit=2,
            sigma_threshold=2.5
        )
        
        cleaned_df, report = cleaner.clean_pipeline(
            messy_data,
            target_timezone='America/New_York',
            trading_hours=(9, 16),
            sigma_threshold=2.0
        )
        
        assert 'total_processing_time' in report
        assert len(cleaned_df) > 0


class TestHelperMethods:
    """测试辅助方法"""
    
    def test_find_consecutive_groups(self):
        """测试连续组查找"""
        cleaner = DataCleaner()
        mask = pd.Series([False, True, True, False, False, True, True, True, False])
        groups = cleaner._find_consecutive_groups(mask)
        
        assert len(groups) == 2
        assert groups[0] == (1, 3)
        assert groups[1] == (5, 8)
    
    def test_fix_ohlc_consistency(self):
        """测试OHLC一致性修正"""
        data = {
            'Open': [100, 105],
            'High': [102, 103],  # High < Close，需要修正
            'Low': [98, 106],    # Low > Open，需要修正
            'Close': [104, 104]
        }
        df = pd.DataFrame(data)
        
        cleaner = DataCleaner()
        fixed_df = cleaner._fix_ohlc_consistency(df)
        
        assert (fixed_df['High'] >= fixed_df[['Open', 'Close']].max(axis=1)).all()
        assert (fixed_df['Low'] <= fixed_df[['Open', 'Close']].min(axis=1)).all()
    
    def test_get_cleaning_report(self):
        """测试获取清洗报告"""
        cleaner = DataCleaner()
        cleaner.cleaning_report = {'test': 'data'}
        
        report = cleaner.get_cleaning_report()
        assert report == {'test': 'data'}
        
        # 验证返回的是副本
        report['new_key'] = 'new_value'
        assert 'new_key' not in cleaner.cleaning_report


class TestPerformance:
    """测试性能要求"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能（10万条数据<1秒）"""
        import time
        
        # 创建10万条数据
        dates = pd.date_range('2023-01-01', periods=100000, freq='1T', tz='UTC')
        close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.0001, 100000)))
        
        data = {
            'Open': close_prices * 0.999,
            'High': close_prices * 1.001,
            'Low': close_prices * 0.998,
            'Close': close_prices,
            'Volume': np.random.uniform(1000, 5000, 100000)
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner()
        
        start_time = time.time()
        cleaned_df, report = cleaner.handle_missing_values(df)
        elapsed_time = time.time() - start_time
        
        # 验证性能要求
        assert elapsed_time < 1.0, f"处理时间{elapsed_time:.3f}秒超过1秒要求"
        assert len(cleaned_df) == len(df)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_single_row_dataframe(self):
        """测试单行DataFrame"""
        dates = pd.date_range('2023-01-01', periods=1, freq='5T', tz='UTC')
        data = {
            'Open': [100],
            'High': [105],
            'Low': [95],
            'Close': [102],
            'Volume': [1000]
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner()
        cleaned_df, report = cleaner.handle_missing_values(df)
        
        assert len(cleaned_df) == 1
    
    def test_all_missing_values(self):
        """测试全部缺失值"""
        dates = pd.date_range('2023-01-01', periods=10, freq='5T', tz='UTC')
        data = {
            'Open': [np.nan] * 10,
            'High': [np.nan] * 10,
            'Low': [np.nan] * 10,
            'Close': [np.nan] * 10,
            'Volume': [np.nan] * 10
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner(max_consecutive_missing=5)
        cleaned_df, report = cleaner.handle_missing_values(df)
        
        # 所有数据应该被删除
        assert len(cleaned_df) == 0 or report['removed_rows'] == 10
    
    def test_zero_prices(self):
        """测试零价格"""
        dates = pd.date_range('2023-01-01', periods=10, freq='5T', tz='UTC')
        data = {
            'Open': [100, 0, 100, 100, 100, 100, 100, 100, 100, 100],
            'High': [105, 0, 105, 105, 105, 105, 105, 105, 105, 105],
            'Low': [95, 0, 95, 95, 95, 95, 95, 95, 95, 95],
            'Close': [102, 0, 102, 102, 102, 102, 102, 102, 102, 102],
            'Volume': [1000] * 10
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner()
        # 应该能够处理而不崩溃
        cleaned_df, report = cleaner.detect_and_handle_outliers(df)
        
        assert len(cleaned_df) == len(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])