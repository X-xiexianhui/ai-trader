"""
数据清洗模块单元测试
TASK-030: 编写数据清洗单元测试
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.cleaner import (
    MissingValueHandler,
    PriceAnomalyHandler,
    VolumeAnomalyHandler,
    OHLCConsistencyFixer,
    TimeAligner,
    DataNormalizer,
    DataCleaningPipeline,
    DataQualityComparator,
    DataQualityScorer
)


class TestMissingValueHandler(unittest.TestCase):
    """测试缺失值处理器"""
    
    def setUp(self):
        """设置测试数据"""
        self.handler = MissingValueHandler(max_consecutive_missing=5)
        
        # 创建测试数据
        self.data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, 104, np.nan, np.nan, 107, 108, 109],
            'volume': [1000, 1100, 1200, np.nan, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    def test_ffill_method(self):
        """测试前向填充"""
        result, report = self.handler.handle_missing(self.data, method='ffill')
        
        # 检查缺失值是否被填充
        self.assertEqual(result['close'].isna().sum(), 0)
        self.assertEqual(result['volume'].isna().sum(), 0)
        
        # 检查报告
        self.assertIn('missing_handled', report)
        self.assertGreater(report['missing_handled'], 0)
    
    def test_interpolate_method(self):
        """测试线性插值"""
        result, report = self.handler.handle_missing(self.data, method='interpolate')
        
        # 检查缺失值是否被填充
        self.assertEqual(result['close'].isna().sum(), 0)
        self.assertEqual(result['volume'].isna().sum(), 0)
    
    def test_drop_method(self):
        """测试删除长缺失段"""
        # 创建有长缺失段的数据
        data_with_long_gap = pd.DataFrame({
            'close': [100, 101, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 108, 109]
        })
        
        result, report = self.handler.handle_missing(data_with_long_gap, method='drop')
        
        # 检查是否删除了长缺失段
        self.assertLess(len(result), len(data_with_long_gap))
        self.assertGreater(report['rows_dropped'], 0)


class TestPriceAnomalyHandler(unittest.TestCase):
    """测试价格异常值处理器"""
    
    def setUp(self):
        """设置测试数据"""
        self.handler = PriceAnomalyHandler(spike_threshold=5.0)
        
        # 创建测试数据，包含一个明显的尖峰
        prices = [100, 101, 102, 200, 103, 104, 105]  # 200是尖峰
        self.data = pd.DataFrame({'close': prices})
    
    def test_spike_detection(self):
        """测试尖峰检测"""
        result, report = self.handler.handle_price_anomalies(self.data)
        
        # 检查是否检测到尖峰
        self.assertGreater(report['spikes_detected'], 0)
        
        # 检查尖峰是否被修正
        self.assertGreater(report['spikes_fixed'], 0)
        
        # 检查修正后的值是否合理（应该接近前后值的均值）
        self.assertLess(result.loc[3, 'close'], 150)  # 修正后应该远小于200


class TestVolumeAnomalyHandler(unittest.TestCase):
    """测试成交量异常处理器"""
    
    def setUp(self):
        """设置测试数据"""
        self.handler = VolumeAnomalyHandler(volume_threshold=3.0)
        
        # 创建测试数据
        volumes = [1000] * 20 + [50000] + [1000] * 10  # 50000是异常大成交量
        volumes[5] = 0  # 添加零成交量
        self.data = pd.DataFrame({'volume': volumes})
    
    def test_high_volume_anomaly(self):
        """测试异常大成交量处理"""
        result, report = self.handler.handle_volume_anomalies(self.data)
        
        # 检查是否检测到异常
        self.assertGreater(report['high_volume_anomalies'], 0)
        
        # 检查异常值是否被cap
        self.assertLess(result['volume'].max(), 50000)
    
    def test_zero_volume(self):
        """测试零成交量处理"""
        result, report = self.handler.handle_volume_anomalies(self.data)
        
        # 检查是否检测到零成交量
        self.assertGreater(report['zero_volume_count'], 0)
        
        # 检查零成交量是否被填充
        self.assertEqual((result['volume'] == 0).sum(), 0)


class TestOHLCConsistencyFixer(unittest.TestCase):
    """测试OHLC一致性修正器"""
    
    def setUp(self):
        """设置测试数据"""
        self.fixer = OHLCConsistencyFixer()
        
        # 创建有不一致的测试数据
        self.data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 104, 103],  # 第2行high < open
            'low': [95, 96, 104],     # 第3行low > close
            'close': [102, 103, 103]
        })
    
    def test_high_low_consistency(self):
        """测试High >= Low"""
        # 创建high < low的情况
        data = pd.DataFrame({
            'open': [100],
            'high': [95],  # high < low
            'low': [105],
            'close': [102]
        })
        
        result, report = self.fixer.fix_ohlc_consistency(data)
        
        # 检查是否修正
        self.assertGreaterEqual(result.loc[0, 'high'], result.loc[0, 'low'])
        self.assertGreater(report['high_low_fixes'], 0)
    
    def test_high_oc_consistency(self):
        """测试High >= max(Open, Close)"""
        result, report = self.fixer.fix_ohlc_consistency(self.data)
        
        # 检查所有行的high >= max(open, close)
        for idx in result.index:
            max_oc = max(result.loc[idx, 'open'], result.loc[idx, 'close'])
            self.assertGreaterEqual(result.loc[idx, 'high'], max_oc)
    
    def test_low_oc_consistency(self):
        """测试Low <= min(Open, Close)"""
        result, report = self.fixer.fix_ohlc_consistency(self.data)
        
        # 检查所有行的low <= min(open, close)
        for idx in result.index:
            min_oc = min(result.loc[idx, 'open'], result.loc[idx, 'close'])
            self.assertLessEqual(result.loc[idx, 'low'], min_oc)


class TestTimeAligner(unittest.TestCase):
    """测试时间对齐器"""
    
    def setUp(self):
        """设置测试数据"""
        self.aligner = TimeAligner(target_freq='5min')
        
        # 创建不规则时间间隔的数据
        dates = pd.to_datetime([
            '2024-01-01 09:00:00',
            '2024-01-01 09:03:00',  # 3分钟间隔
            '2024-01-01 09:05:00',
            '2024-01-01 09:12:00',  # 7分钟间隔
            '2024-01-01 09:15:00'
        ])
        
        self.data = pd.DataFrame({
            'datetime': dates,
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def test_time_alignment(self):
        """测试时间对齐"""
        result, report = self.aligner.align_time(self.data)
        
        # 检查记录数
        self.assertGreater(len(result), 0)
        
        # 检查时间间隔是否为5分钟
        if len(result) > 1:
            time_diffs = result['datetime'].diff().dropna()
            # 大部分时间间隔应该是5分钟
            self.assertTrue(all(td.total_seconds() % 300 == 0 for td in time_diffs))


class TestDataNormalizer(unittest.TestCase):
    """测试数据标准化器"""
    
    def setUp(self):
        """设置测试数据"""
        self.normalizer = DataNormalizer()
        
        self.data = pd.DataFrame({
            'close': [100, 110, 120, 130, 140],
            'volume': [1000, 2000, 3000, 4000, 5000]
        })
    
    def test_standard_normalization(self):
        """测试标准化"""
        result, params = self.normalizer.normalize(
            self.data,
            method='standard',
            columns=['close', 'volume']
        )
        
        # 检查均值接近0，标准差接近1
        self.assertAlmostEqual(result['close'].mean(), 0, places=10)
        self.assertAlmostEqual(result['close'].std(), 1, places=10)
        
        # 检查参数
        self.assertIn('close', params['columns'])
        self.assertIn('mean', params['columns']['close'])
    
    def test_minmax_normalization(self):
        """测试MinMax标准化"""
        result, params = self.normalizer.normalize(
            self.data,
            method='minmax',
            columns=['close']
        )
        
        # 检查值在[0, 1]范围内
        self.assertGreaterEqual(result['close'].min(), 0)
        self.assertLessEqual(result['close'].max(), 1)
    
    def test_inverse_transform(self):
        """测试反标准化"""
        # 先标准化
        normalized, params = self.normalizer.normalize(
            self.data,
            method='standard',
            columns=['close']
        )
        
        # 再反标准化
        denormalized = self.normalizer.inverse_transform(
            normalized,
            columns=['close']
        )
        
        # 检查是否恢复原值
        np.testing.assert_array_almost_equal(
            denormalized['close'].values,
            self.data['close'].values,
            decimal=5
        )


class TestDataCleaningPipeline(unittest.TestCase):
    """测试数据清洗管道"""
    
    def setUp(self):
        """设置测试数据"""
        self.pipeline = DataCleaningPipeline()
        
        # 创建包含各种问题的测试数据
        n = 50
        dates = pd.date_range('2024-01-01', periods=n, freq='5min')
        
        self.data = pd.DataFrame({
            'datetime': dates,
            'open': 100 + np.random.randn(n) * 2,
            'high': 102 + np.random.randn(n) * 2,
            'low': 98 + np.random.randn(n) * 2,
            'close': 100 + np.random.randn(n) * 2,
            'volume': np.random.randint(1000, 10000, n)
        })
        
        # 添加问题
        self.data.loc[5:7, 'close'] = np.nan  # 缺失值
        self.data.loc[10, 'close'] = 200  # 价格尖峰
        self.data.loc[15, 'volume'] = 100000  # 成交量异常
        self.data.loc[20, 'high'] = self.data.loc[20, 'low'] - 1  # OHLC不一致
    
    def test_full_pipeline(self):
        """测试完整清洗流程"""
        result, report = self.pipeline.clean(
            self.data,
            symbol="TEST",
            handle_missing=True,
            fix_price_anomalies=True,
            fix_volume_anomalies=True,
            fix_ohlc=True,
            align_time=False,
            normalize=False
        )
        
        # 检查报告
        self.assertIn('original_records', report)
        self.assertIn('final_records', report)
        self.assertIn('steps', report)
        
        # 检查清洗步骤
        self.assertGreater(len(report['steps']), 0)
        
        # 检查数据质量改善
        # 缺失值应该被处理
        self.assertEqual(result['close'].isna().sum(), 0)
        
        # OHLC应该一致
        for idx in result.index:
            if all(col in result.columns for col in ['open', 'high', 'low', 'close']):
                self.assertGreaterEqual(result.loc[idx, 'high'], result.loc[idx, 'low'])


class TestDataQualityComparator(unittest.TestCase):
    """测试数据质量对比器"""
    
    def setUp(self):
        """设置测试数据"""
        self.comparator = DataQualityComparator()
        
        # 创建清洗前后的数据
        self.before = pd.DataFrame({
            'close': [100, np.nan, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        self.after = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],  # 缺失值已填充
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
    
    def test_comparison(self):
        """测试对比功能"""
        comparison = self.comparator.compare(self.before, self.after)
        
        # 检查报告结构
        self.assertIn('record_count', comparison)
        self.assertIn('missing_values', comparison)
        self.assertIn('statistics', comparison)
        
        # 检查缺失值变化
        self.assertIn('close', comparison['missing_values'])
        self.assertEqual(comparison['missing_values']['close']['before'], 1)
        self.assertEqual(comparison['missing_values']['close']['after'], 0)
    
    def test_report_generation(self):
        """测试报告生成"""
        comparison = self.comparator.compare(self.before, self.after)
        report_text = self.comparator.generate_report(comparison)
        
        # 检查报告内容
        self.assertIn('数据清洗前后对比报告', report_text)
        self.assertIn('记录数量变化', report_text)
        self.assertIn('缺失值变化', report_text)


class TestDataQualityScorer(unittest.TestCase):
    """测试数据质量评分器"""
    
    def setUp(self):
        """设置测试数据"""
        self.scorer = DataQualityScorer()
    
    def test_perfect_data_score(self):
        """测试完美数据的评分"""
        # 创建完美的数据
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        perfect_data = pd.DataFrame({
            'datetime': dates,
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': np.random.randint(1000, 2000, 100)
        })
        
        # 确保OHLC一致性
        for idx in perfect_data.index:
            perfect_data.loc[idx, 'high'] = max(
                perfect_data.loc[idx, 'high'],
                perfect_data.loc[idx, 'open'],
                perfect_data.loc[idx, 'close']
            )
            perfect_data.loc[idx, 'low'] = min(
                perfect_data.loc[idx, 'low'],
                perfect_data.loc[idx, 'open'],
                perfect_data.loc[idx, 'close']
            )
        
        score_report = self.scorer.score(perfect_data)
        
        # 检查评分结构
        self.assertIn('total_score', score_report)
        self.assertIn('scores', score_report)
        self.assertIn('grade', score_report)
        
        # 完美数据应该得高分
        self.assertGreater(score_report['total_score'], 80)
    
    def test_poor_data_score(self):
        """测试低质量数据的评分"""
        # 创建有问题的数据
        poor_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'open': [100] * 50,
            'high': [95] * 50,  # high < low，不一致
            'low': [105] * 50,
            'close': [np.nan] * 25 + [100] * 25,  # 一半缺失
            'volume': [1000] * 50
        })
        
        score_report = self.scorer.score(poor_data)
        
        # 低质量数据应该得低分
        self.assertLess(score_report['total_score'], 70)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def test_empty_dataframe(self):
        """测试空数据框"""
        empty_df = pd.DataFrame()
        
        handler = MissingValueHandler()
        result, report = handler.handle_missing(empty_df)
        
        # 应该返回空数据框，不报错
        self.assertEqual(len(result), 0)
    
    def test_single_row(self):
        """测试单行数据"""
        single_row = pd.DataFrame({
            'close': [100],
            'volume': [1000]
        })
        
        pipeline = DataCleaningPipeline()
        result, report = pipeline.clean(single_row)
        
        # 应该正常处理
        self.assertEqual(len(result), 1)
    
    def test_all_missing(self):
        """测试全部缺失的列"""
        all_missing = pd.DataFrame({
            'close': [np.nan] * 10,
            'volume': [1000] * 10
        })
        
        handler = MissingValueHandler()
        result, report = handler.handle_missing(all_missing, method='ffill')
        
        # 应该能处理，但close列可能仍有缺失
        self.assertIsNotNone(result)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMissingValueHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceAnomalyHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestVolumeAnomalyHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestOHLCConsistencyFixer))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeAligner))
    suite.addTests(loader.loadTestsFromTestCase(TestDataNormalizer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaningPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityComparator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)