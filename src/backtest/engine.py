"""
回测引擎核心实现

基于Backtrader框架，支持GPU加速
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from ..utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class TradingStrategy(bt.Strategy):
    """
    交易策略基类
    
    集成AI模型进行交易决策
    """
    
    params = (
        ('model', None),  # AI模型
        ('initial_cash', 100000),  # 初始资金
        ('commission', 0.001),  # 手续费率
        ('slippage', 0.0005),  # 滑点
        ('position_size', 0.95),  # 最大仓位比例
        ('stop_loss', 0.02),  # 止损比例
        ('take_profit', 0.05),  # 止盈比例
        ('use_gpu', True),  # 是否使用GPU
    )
    
    def __init__(self):
        """初始化策略"""
        self.order = None
        self.entry_price = None
        self.entry_time = None
        self.trades_log = []
        
        # GPU管理器
        if self.params.use_gpu:
            self.gpu_manager = get_gpu_manager()
            self.device = self.gpu_manager.get_device()
        else:
            self.device = None
        
        logger.info(f"策略初始化完成，使用设备: {self.device}")
    
    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datas[0].datetime.date(0)
        logger.debug(f'{dt.isoformat()} {txt}')
    
    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
                self.entry_price = order.executed.price
                self.entry_time = self.datas[0].datetime.datetime(0)
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
                
                # 记录交易
                if self.entry_price:
                    pnl = order.executed.price - self.entry_price
                    pnl_pct = pnl / self.entry_price
                    holding_time = self.datas[0].datetime.datetime(0) - self.entry_time
                    
                    self.trades_log.append({
                        'entry_time': self.entry_time,
                        'exit_time': self.datas[0].datetime.datetime(0),
                        'entry_price': self.entry_price,
                        'exit_price': order.executed.price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'holding_time': holding_time.total_seconds() / 3600  # 小时
                    })
                
                self.entry_price = None
                self.entry_time = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return
        
        self.log(f'交易盈亏, 毛利: {trade.pnl:.2f}, 净利: {trade.pnlcomm:.2f}')
    
    def next(self):
        """
        策略主逻辑
        
        子类需要重写此方法实现具体策略
        """
        # 检查是否有未完成订单
        if self.order:
            return
        
        # 获取当前数据
        current_data = self._get_current_data()
        
        # 使用AI模型预测
        if self.params.model:
            action = self._get_model_action(current_data)
            self._execute_action(action)
    
    def _get_current_data(self) -> Dict:
        """获取当前市场数据"""
        return {
            'open': self.datas[0].open[0],
            'high': self.datas[0].high[0],
            'low': self.datas[0].low[0],
            'close': self.datas[0].close[0],
            'volume': self.datas[0].volume[0],
            'datetime': self.datas[0].datetime.datetime(0)
        }
    
    def _get_model_action(self, data: Dict) -> Dict:
        """
        使用AI模型获取交易动作
        
        Returns:
            Dict: {
                'direction': 0/1/2 (平仓/做多/做空),
                'position_size': float,
                'stop_loss': float,
                'take_profit': float
            }
        """
        # 这里应该调用实际的AI模型
        # 暂时返回默认动作
        return {
            'direction': 0,
            'position_size': 0.0,
            'stop_loss': self.params.stop_loss,
            'take_profit': self.params.take_profit
        }
    
    def _execute_action(self, action: Dict):
        """执行交易动作"""
        direction = action['direction']
        position_size = action.get('position_size', self.params.position_size)
        
        # 当前持仓
        position = self.position.size
        
        if direction == 0:  # 平仓
            if position != 0:
                self.order = self.close()
        
        elif direction == 1:  # 做多
            if position <= 0:
                # 计算买入数量
                cash = self.broker.getcash()
                price = self.datas[0].close[0]
                size = int((cash * position_size) / price)
                
                if size > 0:
                    self.order = self.buy(size=size)
        
        elif direction == 2:  # 做空
            if position >= 0:
                # 计算卖出数量
                cash = self.broker.getcash()
                price = self.datas[0].close[0]
                size = int((cash * position_size) / price)
                
                if size > 0:
                    self.order = self.sell(size=size)
    
    def stop(self):
        """策略结束"""
        self.log(f'最终资产: {self.broker.getvalue():.2f}')


class BacktestEngine:
    """
    回测引擎
    
    基于Backtrader框架，支持GPU加速
    """
    
    def __init__(self, config: Dict):
        """
        初始化回测引擎
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.cerebro = bt.Cerebro()
        
        # 设置初始资金
        initial_cash = config.get('initial_cash', 100000)
        self.cerebro.broker.setcash(initial_cash)
        
        # 设置手续费
        commission = config.get('commission', 0.001)
        self.cerebro.broker.setcommission(commission=commission)
        
        # 设置滑点
        slippage = config.get('slippage', 0.0005)
        # Backtrader的滑点设置
        # self.cerebro.broker.set_slippage_perc(slippage)
        
        # GPU支持
        self.use_gpu = config.get('use_gpu', False)
        if self.use_gpu:
            self.gpu_manager = get_gpu_manager()
            logger.info(f"回测引擎使用GPU: {self.gpu_manager.get_device_type()}")
        
        logger.info(f"回测引擎初始化完成，初始资金: {initial_cash}")
    
    def add_data(self, data: pd.DataFrame, name: str = 'data'):
        """
        添加数据到回测引擎
        
        Args:
            data: OHLCV数据，DataFrame格式
            name: 数据名称
        """
        # 确保数据有正确的列名
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"数据缺少必需列: {col}")
        
        # 确保索引是datetime类型
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex")
        
        # 创建Backtrader数据源
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # 使用索引作为datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # 不使用openinterest
        )
        
        self.cerebro.adddata(bt_data, name=name)
        logger.info(f"添加数据: {name}, 长度: {len(data)}")
    
    def add_strategy(self, strategy_class, **kwargs):
        """
        添加策略
        
        Args:
            strategy_class: 策略类
            **kwargs: 策略参数
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"添加策略: {strategy_class.__name__}")
    
    def add_analyzer(self, analyzer_class, **kwargs):
        """
        添加分析器
        
        Args:
            analyzer_class: 分析器类
            **kwargs: 分析器参数
        """
        self.cerebro.addanalyzer(analyzer_class, **kwargs)
    
    def run(self) -> List:
        """
        运行回测
        
        Returns:
            List: 策略实例列表
        """
        logger.info("开始回测...")
        start_value = self.cerebro.broker.getvalue()
        logger.info(f"初始资产: {start_value:.2f}")
        
        # 运行回测
        results = self.cerebro.run()
        
        end_value = self.cerebro.broker.getvalue()
        logger.info(f"最终资产: {end_value:.2f}")
        logger.info(f"收益: {end_value - start_value:.2f} ({(end_value/start_value - 1)*100:.2f}%)")
        
        return results
    
    def plot(self, **kwargs):
        """
        绘制回测结果
        
        Args:
            **kwargs: 绘图参数
        """
        self.cerebro.plot(**kwargs)
    
    def get_results(self, results: List) -> Dict:
        """
        提取回测结果
        
        Args:
            results: 策略实例列表
            
        Returns:
            Dict: 回测结果字典
        """
        if not results:
            return {}
        
        strategy = results[0]
        
        # 基本信息
        result = {
            'initial_value': self.config.get('initial_cash', 100000),
            'final_value': self.cerebro.broker.getvalue(),
            'total_return': self.cerebro.broker.getvalue() / self.config.get('initial_cash', 100000) - 1,
        }
        
        # 交易记录
        if hasattr(strategy, 'trades_log'):
            result['trades'] = strategy.trades_log
            result['num_trades'] = len(strategy.trades_log)
            
            if strategy.trades_log:
                # 胜率
                winning_trades = [t for t in strategy.trades_log if t['pnl'] > 0]
                result['win_rate'] = len(winning_trades) / len(strategy.trades_log)
                
                # 平均盈亏
                result['avg_pnl'] = np.mean([t['pnl'] for t in strategy.trades_log])
                result['avg_pnl_pct'] = np.mean([t['pnl_pct'] for t in strategy.trades_log])
                
                # 平均持仓时间
                result['avg_holding_time'] = np.mean([t['holding_time'] for t in strategy.trades_log])
        
        # 分析器结果
        if hasattr(strategy, 'analyzers'):
            for analyzer in strategy.analyzers:
                analyzer_name = analyzer.__class__.__name__
                result[analyzer_name] = analyzer.get_analysis()
        
        return result


def create_backtest_engine(config: Dict) -> BacktestEngine:
    """
    创建回测引擎实例
    
    Args:
        config: 配置字典
        
    Returns:
        BacktestEngine: 回测引擎实例
    """
    return BacktestEngine(config)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='5min')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # 确保OHLC关系正确
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # 创建回测引擎
    config = {
        'initial_cash': 100000,
        'commission': 0.001,
        'slippage': 0.0005,
        'use_gpu': False
    }
    
    engine = create_backtest_engine(config)
    engine.add_data(data, name='test_data')
    engine.add_strategy(TradingStrategy)
    
    # 添加分析器
    engine.add_analyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    engine.add_analyzer(bt.analyzers.DrawDown, _name='drawdown')
    engine.add_analyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    results = engine.run()
    
    # 获取结果
    backtest_results = engine.get_results(results)
    print("\n回测结果:")
    for key, value in backtest_results.items():
        if key != 'trades':
            print(f"  {key}: {value}")