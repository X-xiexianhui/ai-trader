"""
模块5测试用例
测试数据下载、存储和回测功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import shutil

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.downloader import DataDownloader, IncrementalUpdater
from src.data.storage import DataStorage, DataCache
from src.backtest.execution import OrderExecutor, SlippageModel, CommissionModel
from src.backtest.recorder import BacktestRecorder


class TestDataDownloader(unittest.TestCase):
    """测试数据下载器"""
    
    def setUp(self):
        """设置测试环境"""
        self.downloader = DataDownloader(max_retries=2, retry_delay=1)
    
    def test_downloader_initialization(self):
        """测试下载器初始化"""
        self.assertEqual(self.downloader.max_retries, 2)
        self.assertEqual(self.downloader.retry_delay, 1)
    
    def test_standardize_columns(self):
        """测试列名标准化"""
        # 创建测试数据
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        })
        
        standardized = self.downloader._standardize_columns(df)
        
        # 验证列名
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(standardized.columns), expected_columns)
    
    def test_validate_data(self):
        """测试数据验证"""
        # 创建有效数据
        valid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        self.assertTrue(self.downloader._validate_data(valid_data))
        
        # 创建无效数据（OHLC不一致）
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [98, 99],  # high < low
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1100]
        })
        
        self.assertFalse(self.downloader._validate_data(invalid_data))


class TestDataStorage(unittest.TestCase):
    """测试数据存储"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = DataStorage(base_path=self.temp_dir)
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.test_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_parquet(self):
        """测试Parquet保存和加载"""
        # 保存数据
        success = self.storage.save_parquet(self.test_data, 'TEST')
        self.assertTrue(success)
        
        # 加载数据
        loaded_data = self.storage.load_parquet('TEST')
        self.assertIsNotNone(loaded_data)
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(self.test_data, loaded_data)
    
    def test_save_and_load_hdf5(self):
        """测试HDF5保存和加载"""
        # 保存数据
        success = self.storage.save_hdf5(self.test_data, 'TEST')
        self.assertTrue(success)
        
        # 加载数据
        loaded_data = self.storage.load_hdf5('TEST')
        self.assertIsNotNone(loaded_data)
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(self.test_data, loaded_data)
    
    def test_get_file_info(self):
        """测试获取文件信息"""
        # 保存数据
        self.storage.save_parquet(self.test_data, 'TEST')
        
        # 获取文件信息
        info = self.storage.get_file_info('TEST', format='parquet')
        self.assertIsNotNone(info)
        self.assertIn('size_mb', info)
        self.assertIn('path', info)
    
    def test_list_files(self):
        """测试列出文件"""
        # 保存多个文件
        self.storage.save_parquet(self.test_data, 'TEST1')
        self.storage.save_parquet(self.test_data, 'TEST2')
        
        # 列出文件
        files = self.storage.list_files(format='parquet')
        self.assertIn('TEST1', files)
        self.assertIn('TEST2', files)


class TestDataCache(unittest.TestCase):
    """测试数据缓存"""
    
    def setUp(self):
        """设置测试环境"""
        self.cache = DataCache(max_size=3)
        
        # 创建测试数据
        self.test_data = pd.DataFrame({
            'value': [1, 2, 3]
        })
    
    def test_cache_put_and_get(self):
        """测试缓存存取"""
        # 放入缓存
        self.cache.put('TEST', self.test_data)
        
        # 从缓存获取
        cached_data = self.cache.get('TEST')
        self.assertIsNotNone(cached_data)
        pd.testing.assert_frame_equal(self.test_data, cached_data)
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        result = self.cache.get('NONEXISTENT')
        self.assertIsNone(result)
    
    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        # 添加超过最大大小的数据
        for i in range(5):
            self.cache.put(f'TEST{i}', self.test_data)
        
        # 验证缓存大小
        self.assertEqual(self.cache.size(), 3)


class TestOrderExecutor(unittest.TestCase):
    """测试订单执行器"""
    
    def setUp(self):
        """设置测试环境"""
        self.executor = OrderExecutor(
            slippage_perc=0.001,
            commission=0.001
        )
    
    def test_execute_market_order_buy(self):
        """测试市价买单执行"""
        result = self.executor.execute_market_order(
            price=100.0,
            size=10,
            order_type='buy'
        )
        
        self.assertEqual(result['order_type'], 'buy')
        self.assertEqual(result['size'], 10)
        self.assertGreater(result['executed_price'], 100.0)  # 买入有滑点
        self.assertGreater(result['commission'], 0)
        self.assertEqual(result['status'], 'executed')
    
    def test_execute_market_order_sell(self):
        """测试市价卖单执行"""
        result = self.executor.execute_market_order(
            price=100.0,
            size=10,
            order_type='sell'
        )
        
        self.assertEqual(result['order_type'], 'sell')
        self.assertLess(result['executed_price'], 100.0)  # 卖出有滑点
    
    def test_check_stop_loss_long(self):
        """测试多头止损"""
        # 价格下跌触发止损
        triggered = self.executor.check_stop_loss(
            current_price=98.0,
            entry_price=100.0,
            stop_loss_pct=0.02,
            position_type='long'
        )
        self.assertTrue(triggered)
        
        # 价格未达到止损
        not_triggered = self.executor.check_stop_loss(
            current_price=99.5,
            entry_price=100.0,
            stop_loss_pct=0.02,
            position_type='long'
        )
        self.assertFalse(not_triggered)
    
    def test_check_take_profit_long(self):
        """测试多头止盈"""
        # 价格上涨触发止盈
        triggered = self.executor.check_take_profit(
            current_price=105.0,
            entry_price=100.0,
            take_profit_pct=0.05,
            position_type='long'
        )
        self.assertTrue(triggered)
    
    def test_calculate_position_pnl(self):
        """测试持仓盈亏计算"""
        pnl_info = self.executor.calculate_position_pnl(
            entry_price=100.0,
            current_price=105.0,
            size=10,
            position_type='long'
        )
        
        self.assertEqual(pnl_info['pnl'], 50.0)  # (105-100)*10
        self.assertEqual(pnl_info['pnl_pct'], 5.0)


class TestBacktestRecorder(unittest.TestCase):
    """测试回测记录器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.recorder = BacktestRecorder(output_dir=self.temp_dir)
        
        # 添加测试数据
        for i in range(10):
            self.recorder.record_trade({
                'entry_date': datetime.now() - timedelta(days=10-i),
                'exit_date': datetime.now() - timedelta(days=9-i),
                'entry_price': 100.0,
                'exit_price': 105.0 if i % 2 == 0 else 95.0,
                'size': 10,
                'pnl': 50.0 if i % 2 == 0 else -50.0,
                'pnl_pct': 5.0 if i % 2 == 0 else -5.0,
                'bars_held': 10
            })
        
        # 添加权益曲线数据
        for i in range(100):
            self.recorder.record_equity(
                timestamp=datetime.now() - timedelta(hours=100-i),
                equity=100000 + i * 100,
                cash=50000,
                position_value=50000 + i * 100
            )
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_record_trade(self):
        """测试交易记录"""
        initial_count = len(self.recorder.trades)
        
        self.recorder.record_trade({
            'pnl': 100.0,
            'pnl_pct': 10.0
        })
        
        self.assertEqual(len(self.recorder.trades), initial_count + 1)
    
    def test_calculate_metrics(self):
        """测试指标计算"""
        metrics = self.recorder.calculate_metrics()
        
        self.assertIn('total_trades', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # 验证胜率
        self.assertEqual(metrics['total_trades'], 10)
        self.assertEqual(metrics['winning_trades'], 5)
        self.assertEqual(metrics['win_rate'], 50.0)
    
    def test_generate_report(self):
        """测试报告生成"""
        report_path = self.recorder.generate_report()
        self.assertTrue(Path(report_path).exists())
    
    def test_save_trades(self):
        """测试保存交易记录"""
        filepath = self.recorder.save_trades()
        self.assertTrue(Path(filepath).exists())
        
        # 验证文件内容
        df = pd.read_csv(filepath)
        self.assertEqual(len(df), 10)
    
    def test_save_equity_curve(self):
        """测试保存权益曲线"""
        filepath = self.recorder.save_equity_curve()
        self.assertTrue(Path(filepath).exists())


class TestIncrementalUpdater(unittest.TestCase):
    """测试增量更新器"""
    
    def setUp(self):
        """设置测试环境"""
        self.downloader = DataDownloader()
        self.updater = IncrementalUpdater(self.downloader)
        
        # 创建现有数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.existing_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
    
    def test_update_with_empty_data(self):
        """测试空数据更新"""
        empty_data = pd.DataFrame()
        updated_data, new_records = self.updater.update('TEST', empty_data)
        
        # 空数据应该触发全量下载（但可能失败）
        self.assertIsInstance(updated_data, pd.DataFrame)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDataDownloader))
    suite.addTests(loader.loadTestsFromTestCase(TestDataStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestDataCache))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestRecorder))
    suite.addTests(loader.loadTestsFromTestCase(TestIncrementalUpdater))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)