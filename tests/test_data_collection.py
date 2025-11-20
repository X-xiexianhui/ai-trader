"""
数据采集模块单元测试
Unit Tests for Data Collection Module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import YahooFinanceDownloader
from src.data.cache import DataCache
from src.data.updater import DataUpdater
from src.data.validator import DataValidator
from src.data.cleaner import DataCleaner, DataCleaningPipeline
from src.data.manager import DataVersionControl, MultiSymbolDataManager


class TestYahooFinanceDownloader(unittest.TestCase):
    """测试Yahoo Finance下载器"""
    
    def setUp(self):
        """测试前准备"""
        self.downloader = YahooFinanceDownloader()
    
    def test_download_single_symbol(self):
        """测试单个品种下载"""
        data = self.downloader.download(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            interval="1d"
        )
        
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        self.assertIn('datetime', data.columns)
        self.assertIn('close', data.columns)
    
    def test_download_with_invalid_symbol(self):
        """测试无效品种代码"""
        with self.assertRaises(RuntimeError):
            self.downloader.download(
                symbol="INVALID_SYMBOL_12345",
                start="2024-01-01",
                end="2024-01-31",
                interval="1d"
            )
    
    def test_get_symbol_info(self):
        """测试获取品种信息"""
        info = self.downloader.get_symbol_info("AAPL")
        
        self.assertIsInstance(info, dict)
        self.assertIn('symbol', info)
        self.assertEqual(info['symbol'], 'AAPL')


class TestDataCache(unittest.TestCase):
    """测试数据缓存"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        # 保存数据
        success = self.cache.save(self.test_data, 'TEST', '5m')
        self.assertTrue(success)
        
        # 加载数据
        loaded_data = self.cache.load('TEST', '5m', check_expiry=False)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), len(self.test_data))
    
    def test_cache_exists(self):
        """测试缓存存在检查"""
        # 保存数据
        self.cache.save(self.test_data, 'TEST', '5m')
        
        # 检查存在
        self.assertTrue(self.cache.exists('TEST', '5m'))
        self.assertFalse(self.cache.exists('NONEXISTENT', '5m'))
    
    def test_delete_cache(self):
        """测试删除缓存"""
        # 保存数据
        self.cache.save(self.test_data, 'TEST', '5m')
        self.assertTrue(self.cache.exists('TEST', '5m'))
        
        # 删除缓存
        success = self.cache.delete('TEST', '5m')
        self.assertTrue(success)
        self.assertFalse(self.cache.exists('TEST', '5m'))
    
    def test_get_metadata(self):
        """测试获取元数据"""
        # 保存数据
        self.cache.save(self.test_data, 'TEST', '5m')
        
        # 获取元数据
        metadata = self.cache.get_metadata('TEST', '5m')
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata['symbol'], 'TEST')
        self.assertEqual(metadata['rows'], len(self.test_data))


class TestDataValidator(unittest.TestCase):
    """测试数据验证器"""
    
    def setUp(self):
        """测试前准备"""
        self.validator = DataValidator()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_check_missing_values(self):
        """测试缺失值检查"""
        # 添加缺失值
        data_with_missing = self.test_data.copy()
        data_with_missing.loc[10, 'close'] = np.nan
        
        result = self.validator.check_missing_values(data_with_missing)
        self.assertIsInstance(result, dict)
        self.assertIn('passed', result)
        self.assertIn('total_missing', result)
    
    def test_check_ohlc_consistency(self):
        """测试OHLC一致性检查"""
        # 创建不一致的数据
        data_inconsistent = self.test_data.copy()
        data_inconsistent.loc[10, 'high'] = 90  # High < Low
        
        result = self.validator.check_ohlc_consistency(data_inconsistent)
        self.assertIsInstance(result, dict)
        self.assertIn('passed', result)
        self.assertFalse(result['passed'])
    
    def test_detect_price_outliers(self):
        """测试价格异常值检测"""
        # 添加异常值
        data_with_outliers = self.test_data.copy()
        data_with_outliers.loc[50, 'close'] = 200  # 明显异常
        
        result = self.validator.detect_price_outliers(data_with_outliers)
        self.assertIsInstance(result, dict)
        self.assertIn('outlier_count', result)
        self.assertGreater(result['outlier_count'], 0)
    
    def test_validate_and_fix(self):
        """测试验证和修复"""
        # 创建有问题的数据
        data_with_issues = self.test_data.copy()
        data_with_issues.loc[10, 'close'] = np.nan
        data_with_issues.loc[20, 'high'] = 90
        
        validated_data, report = self.validator.validate(
            data_with_issues,
            symbol='TEST',
            fix_issues=True
        )
        
        self.assertIsNotNone(validated_data)
        self.assertIsInstance(report, dict)
        self.assertIn('quality_score', report)
        self.assertGreater(report['quality_score'], 0)


class TestDataCleaner(unittest.TestCase):
    """测试数据清洗器"""
    
    def setUp(self):
        """测试前准备"""
        self.cleaner = DataCleaner()
        
        # 创建测试数据
        import pytz
        et_tz = pytz.timezone('America/New_York')
        dates = pd.date_range('2024-01-01 09:00', periods=100, freq='5min', tz=et_tz)
        
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_convert_timezone(self):
        """测试时区转换"""
        converted_data = self.cleaner.convert_timezone(self.test_data)
        
        self.assertIsNotNone(converted_data)
        self.assertIn('datetime', converted_data.columns)
        # 检查时区是否为UTC
        self.assertEqual(str(converted_data['datetime'].dt.tz), 'UTC')
    
    def test_filter_trading_hours(self):
        """测试交易时段过滤"""
        filtered_data = self.cleaner.filter_trading_hours(
            self.test_data,
            start_time='09:30',
            end_time='16:00',
            include_extended=False
        )
        
        self.assertIsNotNone(filtered_data)
        self.assertLessEqual(len(filtered_data), len(self.test_data))
    
    def test_clean_pipeline(self):
        """测试完整清洗流程"""
        cleaned_data, report = self.cleaner.clean(
            self.test_data,
            symbol='TEST',
            convert_timezone=True,
            filter_trading_hours=False,
            validate=True,
            fix_issues=True
        )
        
        self.assertIsNotNone(cleaned_data)
        self.assertIsInstance(report, dict)
        self.assertIn('final_records', report)


class TestDataVersionControl(unittest.TestCase):
    """测试数据版本控制"""
    
    def setUp(self):
        """测试前准备"""
        self.version_control = DataVersionControl()
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'close': np.random.randn(100) + 100
        })
    
    def test_create_version(self):
        """测试创建版本"""
        version_id = self.version_control.create_version(
            symbol='TEST',
            interval='5m',
            data=self.test_data
        )
        
        self.assertIsNotNone(version_id)
        self.assertIsInstance(version_id, str)
    
    def test_get_versions(self):
        """测试获取版本列表"""
        # 创建版本
        self.version_control.create_version('TEST', '5m', self.test_data)
        
        # 获取版本
        versions = self.version_control.get_versions('TEST', '5m')
        self.assertIsInstance(versions, list)
        self.assertGreater(len(versions), 0)
    
    def test_get_latest_version(self):
        """测试获取最新版本"""
        # 创建版本
        self.version_control.create_version('TEST', '5m', self.test_data)
        
        # 获取最新版本
        latest = self.version_control.get_latest_version('TEST', '5m')
        self.assertIsNotNone(latest)
        self.assertIsInstance(latest, dict)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestYahooFinanceDownloader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCache))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCleaner))
    suite.addTests(loader.loadTestsFromTestCase(TestDataVersionControl))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回结果
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)