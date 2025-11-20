"""
回测模块单元测试
"""

import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from src.backtest.metrics import PerformanceMetrics
from src.backtest.signal_generator import (
    TradingSignal, SignalType, TechnicalSignalGenerator,
    AIModelSignalGenerator, EnsembleSignalGenerator
)
from src.backtest.risk_manager import (
    RiskManager, Position, PositionSizeMethod, RiskLevel
)
from src.backtest.gpu_backtest import GPUBacktestCalculator, calculate_metrics_gpu
from src.backtest.report_generator import ReportGenerator
from src.backtest.visualizer import BacktestVisualizer


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标计算"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.randn(252) * 0.02 + 0.001
        equity = (1 + returns).cumprod() * 100000
        
        self.equity_curve = pd.Series(equity, index=dates)
        self.returns = pd.Series(returns, index=dates)
        
        self.metrics = PerformanceMetrics(
            equity_curve=self.equity_curve,
            returns=self.returns,
            initial_capital=100000
        )
    
    def test_total_return(self):
        """测试总收益计算"""
        total_return = self.metrics.total_return()
        self.assertIsInstance(total_return, float)
        self.assertGreater(total_return, -1)  # 不应该亏损超过100%
    
    def test_sharpe_ratio(self):
        """测试夏普比率计算"""
        sharpe = self.metrics.sharpe_ratio()
        self.assertIsInstance(sharpe, float)
        self.assertGreater(sharpe, -10)  # 合理范围
        self.assertLess(sharpe, 10)
    
    def test_max_drawdown(self):
        """测试最大回撤计算"""
        max_dd = self.metrics.max_drawdown()
        self.assertIsInstance(max_dd, float)
        self.assertLessEqual(max_dd, 0)  # 回撤应该是负数或0
        self.assertGreaterEqual(max_dd, -1)  # 不应该超过-100%
    
    def test_win_rate(self):
        """测试胜率计算"""
        win_rate = self.metrics.win_rate()
        self.assertIsInstance(win_rate, float)
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)
    
    def test_get_all_metrics(self):
        """测试获取所有指标"""
        all_metrics = self.metrics.get_all_metrics()
        self.assertIsInstance(all_metrics, dict)
        self.assertIn('total_return', all_metrics)
        self.assertIn('sharpe_ratio', all_metrics)
        self.assertIn('max_drawdown', all_metrics)


class TestSignalGenerator(unittest.TestCase):
    """测试信号生成器"""
    
    def setUp(self):
        """设置测试数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        self.data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
        }, index=dates)
    
    def test_trading_signal_creation(self):
        """测试交易信号创建"""
        signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=1.5,
            confidence=0.8
        )
        
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.strength, 1.5)
        self.assertEqual(signal.confidence, 0.8)
    
    def test_technical_signal_generator(self):
        """测试技术指标信号生成器"""
        generator = TechnicalSignalGenerator(
            short_window=10,
            long_window=20
        )
        
        signal = generator.generate(self.data)
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL, SignalType.HOLD])
        self.assertGreaterEqual(signal.confidence, 0)
        self.assertLessEqual(signal.confidence, 1)
    
    def test_ai_model_signal_generator(self):
        """测试AI模型信号生成器"""
        generator = AIModelSignalGenerator(device='cpu')
        
        # 不设置模型时应该返回HOLD
        signal = generator.generate(self.data)
        self.assertEqual(signal.signal_type, SignalType.HOLD)
    
    def test_ensemble_signal_generator(self):
        """测试集成信号生成器"""
        tech_gen = TechnicalSignalGenerator()
        ai_gen = AIModelSignalGenerator(device='cpu')
        
        ensemble_gen = EnsembleSignalGenerator(
            generators=[tech_gen, ai_gen],
            weights=[0.6, 0.4],
            voting_method='weighted'
        )
        
        signal = ensemble_gen.generate(self.data)
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertIn(signal.signal_type, [SignalType.BUY, SignalType.SELL, SignalType.HOLD])


class TestRiskManager(unittest.TestCase):
    """测试风险管理器"""
    
    def setUp(self):
        """设置测试数据"""
        self.risk_manager = RiskManager(
            initial_capital=100000,
            max_position_size=0.1,
            position_size_method=PositionSizeMethod.PERCENT_EQUITY
        )
        
        self.signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=1.5,
            confidence=0.8
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.risk_manager.initial_capital, 100000)
        self.assertEqual(self.risk_manager.current_capital, 100000)
        self.assertEqual(len(self.risk_manager.positions), 0)
    
    def test_calculate_position_size(self):
        """测试仓位计算"""
        size = self.risk_manager.calculate_position_size(
            signal=self.signal,
            current_price=50000,
            atr=1000
        )
        
        self.assertIsInstance(size, float)
        self.assertGreater(size, 0)
    
    def test_open_position(self):
        """测试开仓"""
        timestamp = pd.Timestamp.now()
        position = self.risk_manager.open_position(
            symbol='BTCUSDT',
            signal=self.signal,
            current_price=50000,
            timestamp=timestamp,
            atr=1000
        )
        
        self.assertIsNotNone(position)
        self.assertIsInstance(position, Position)
        self.assertEqual(position.symbol, 'BTCUSDT')
        self.assertGreater(position.size, 0)  # 买入信号，仓位应该为正
    
    def test_close_position(self):
        """测试平仓"""
        timestamp = pd.Timestamp.now()
        
        # 先开仓
        self.risk_manager.open_position(
            symbol='BTCUSDT',
            signal=self.signal,
            current_price=50000,
            timestamp=timestamp,
            atr=1000
        )
        
        # 再平仓
        pnl = self.risk_manager.close_position(
            symbol='BTCUSDT',
            current_price=51000,
            timestamp=timestamp,
            reason='test'
        )
        
        self.assertIsNotNone(pnl)
        self.assertIsInstance(pnl, float)
        self.assertEqual(len(self.risk_manager.positions), 0)
    
    def test_update_positions(self):
        """测试更新持仓"""
        timestamp = pd.Timestamp.now()
        
        # 开仓
        self.risk_manager.open_position(
            symbol='BTCUSDT',
            signal=self.signal,
            current_price=50000,
            timestamp=timestamp,
            atr=1000
        )
        
        # 更新价格
        self.risk_manager.update_positions(
            prices={'BTCUSDT': 51000},
            timestamp=timestamp
        )
        
        # 检查持仓是否更新
        if 'BTCUSDT' in self.risk_manager.positions:
            position = self.risk_manager.positions['BTCUSDT']
            self.assertEqual(position.current_price, 51000)
    
    def test_statistics(self):
        """测试统计信息"""
        stats = self.risk_manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('initial_capital', stats)
        self.assertIn('current_capital', stats)
        self.assertIn('total_trades', stats)
        self.assertIn('win_rate', stats)


class TestGPUBacktest(unittest.TestCase):
    """测试GPU加速回测"""
    
    def setUp(self):
        """设置测试数据"""
        self.calculator = GPUBacktestCalculator(device='cpu')  # 使用CPU进行测试
        self.returns = np.random.randn(1000) * 0.01 + 0.0001
    
    def test_calculate_returns(self):
        """测试收益率计算"""
        prices = torch.tensor([100, 101, 102, 101, 103], dtype=torch.float32)
        returns = self.calculator.calculate_returns(prices)
        
        self.assertEqual(len(returns), len(prices))
        self.assertEqual(returns[0].item(), 0)  # 第一个值应该是0
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        returns_tensor = self.calculator.to_tensor(self.returns)
        sharpe = self.calculator.calculate_sharpe_ratio(returns_tensor)
        
        self.assertIsInstance(sharpe.item(), float)
    
    def test_calculate_drawdown(self):
        """测试回撤计算"""
        equity = torch.tensor([100, 110, 105, 115, 110], dtype=torch.float32)
        drawdown, max_dd = self.calculator.calculate_drawdown(equity)
        
        self.assertEqual(len(drawdown), len(equity))
        self.assertLessEqual(max_dd.item(), 0)
    
    def test_batch_calculate_metrics(self):
        """测试批量计算指标"""
        batch_size = 10
        returns_batch = self.calculator.to_tensor(
            np.random.randn(batch_size, 100) * 0.01
        )
        
        metrics = self.calculator.batch_calculate_metrics(returns_batch)
        
        self.assertIsInstance(metrics, dict)
        self.assertEqual(len(metrics['total_return']), batch_size)
        self.assertEqual(len(metrics['sharpe_ratio']), batch_size)
    
    def test_calculate_metrics_gpu(self):
        """测试GPU指标计算便捷函数"""
        metrics = calculate_metrics_gpu(self.returns, device='cpu')
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)


class TestReportGenerator(unittest.TestCase):
    """测试报告生成器"""
    
    def setUp(self):
        """设置测试数据"""
        self.generator = ReportGenerator(output_dir='tests/test_reports')
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        returns = np.random.randn(100) * 0.02 + 0.001
        self.equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        
        self.results = {
            'initial_capital': 100000,
            'final_capital': 110000,
            'total_return': 10000,
            'total_return_pct': 0.10,
            'annual_return': 0.40,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.15,
            'total_trades': 50,
            'winning_trades': 30,
            'losing_trades': 20,
            'win_rate': 0.60
        }
        
        self.trades = [
            {
                'timestamp': dates[i],
                'action': 'close',
                'price': 50000 + i * 100,
                'size': 0.1,
                'pnl': np.random.randn() * 100
            }
            for i in range(20)
        ]
    
    def test_generate_summary(self):
        """测试生成摘要"""
        summary = self.generator.generate_summary(self.results)
        
        self.assertIsInstance(summary, str)
        self.assertIn('总收益率', summary)
        self.assertIn('夏普比率', summary)
    
    def test_generate_report(self):
        """测试生成报告"""
        report_files = self.generator.generate_report(
            results=self.results,
            trades=self.trades,
            equity_curve=self.equity_curve,
            format='json'  # 只生成JSON以加快测试
        )
        
        self.assertIsInstance(report_files, dict)
        self.assertIn('json', report_files)


class TestVisualizer(unittest.TestCase):
    """测试可视化器"""
    
    def setUp(self):
        """设置测试数据"""
        self.visualizer = BacktestVisualizer(output_dir='tests/test_plots')
        
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        returns = np.random.randn(252) * 0.02 + 0.001
        self.equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
        self.returns = pd.Series(returns, index=dates)
        
        self.trades = [
            {
                'timestamp': dates[i * 5],
                'action': 'close',
                'pnl': np.random.randn() * 1000,
                'return': np.random.randn() * 0.02
            }
            for i in range(50)
        ]
        
        self.results = {
            'total_return_pct': 0.15,
            'annual_return': 0.20,
            'max_drawdown': -0.10,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'total_trades': 50,
            'winning_trades': 30,
            'losing_trades': 20,
            'win_rate': 0.60,
            'avg_win': 500,
            'avg_loss': -300
        }
    
    def test_plot_equity_curve(self):
        """测试绘制权益曲线"""
        path = self.visualizer.plot_equity_curve(self.equity_curve)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.png'))
    
    def test_plot_drawdown(self):
        """测试绘制回撤图"""
        path = self.visualizer.plot_drawdown(self.equity_curve)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.png'))
    
    def test_plot_returns_distribution(self):
        """测试绘制收益率分布"""
        path = self.visualizer.plot_returns_distribution(self.returns)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.png'))


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskManager))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestReportGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualizer))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # 打印测试结果摘要
    print("\n" + "=" * 70)
    print("测试结果摘要")
    print("=" * 70)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)