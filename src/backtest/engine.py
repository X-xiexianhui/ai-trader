"""
Backtrader回测引擎集成
实现完整的回测框架
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtrader回测引擎
    
    功能：
    1. 数据源适配
    2. 策略接口
    3. 订单执行
    4. 性能分析
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        stake: int = 1
    ):
        """
        初始化回测引擎
        
        Args:
            initial_cash: 初始资金
            commission: 手续费率
            slippage: 滑点率
            stake: 每次交易手数
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.stake = stake
        
        # 创建Cerebro引擎
        self.cerebro = bt.Cerebro()
        
        # 设置初始资金
        self.cerebro.broker.setcash(initial_cash)
        
        # 设置手续费
        self.cerebro.broker.setcommission(commission=commission)
        
        # 设置交易手数
        self.cerebro.addsizer(bt.sizers.FixedSize, stake=stake)
        
        # 添加分析器
        self._add_analyzers()
        
        logger.info(f"回测引擎初始化完成，初始资金: {initial_cash}")
    
    def _add_analyzers(self):
        """添加性能分析器"""
        # 收益分析
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # 夏普比率
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        # 回撤分析
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        # 交易分析
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        # 年化收益
        self.cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
        
        # 时间收益
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    
    def add_data(
        self,
        data: pd.DataFrame,
        name: str = 'data'
    ) -> None:
        """
        添加数据源
        
        Args:
            data: OHLCV DataFrame，索引为DatetimeIndex
            name: 数据名称
        """
        # 确保数据格式正确
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据索引必须是DatetimeIndex")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"数据必须包含列: {required_columns}")
        
        # 创建Backtrader数据源
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,  # 使用索引作为时间
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        # 添加到Cerebro
        self.cerebro.adddata(bt_data, name=name)
        
        logger.info(f"添加数据源 '{name}'，共 {len(data)} 条记录")
    
    def add_strategy(
        self,
        strategy_class: type,
        **kwargs
    ) -> None:
        """
        添加交易策略
        
        Args:
            strategy_class: 策略类
            **kwargs: 策略参数
        """
        self.cerebro.addstrategy(strategy_class, **kwargs)
        logger.info(f"添加策略: {strategy_class.__name__}")
    
    def run(self) -> List[bt.Strategy]:
        """
        运行回测
        
        Returns:
            策略实例列表
        """
        logger.info("开始回测...")
        
        # 记录初始资金
        start_value = self.cerebro.broker.getvalue()
        logger.info(f"初始资金: {start_value:.2f}")
        
        # 运行回测
        results = self.cerebro.run()
        
        # 记录最终资金
        end_value = self.cerebro.broker.getvalue()
        logger.info(f"最终资金: {end_value:.2f}")
        logger.info(f"收益: {end_value - start_value:.2f} ({(end_value/start_value - 1)*100:.2f}%)")
        
        return results
    
    def get_results(self, strategy: bt.Strategy) -> Dict[str, Any]:
        """
        获取回测结果
        
        Args:
            strategy: 策略实例
            
        Returns:
            结果字典
        """
        results = {}
        
        # 基本信息
        results['initial_cash'] = self.initial_cash
        results['final_value'] = self.cerebro.broker.getvalue()
        results['total_return'] = (results['final_value'] / self.initial_cash - 1) * 100
        
        # 收益分析
        if hasattr(strategy.analyzers, 'returns'):
            returns_analyzer = strategy.analyzers.returns.get_analysis()
            results['returns'] = returns_analyzer
        
        # 夏普比率
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe_analyzer = strategy.analyzers.sharpe.get_analysis()
            results['sharpe_ratio'] = sharpe_analyzer.get('sharperatio', None)
        
        # 回撤分析
        if hasattr(strategy.analyzers, 'drawdown'):
            dd_analyzer = strategy.analyzers.drawdown.get_analysis()
            results['max_drawdown'] = dd_analyzer.get('max', {}).get('drawdown', None)
            results['max_drawdown_period'] = dd_analyzer.get('max', {}).get('len', None)
        
        # 交易分析
        if hasattr(strategy.analyzers, 'trades'):
            trades_analyzer = strategy.analyzers.trades.get_analysis()
            results['total_trades'] = trades_analyzer.get('total', {}).get('total', 0)
            results['won_trades'] = trades_analyzer.get('won', {}).get('total', 0)
            results['lost_trades'] = trades_analyzer.get('lost', {}).get('total', 0)
            
            if results['total_trades'] > 0:
                results['win_rate'] = results['won_trades'] / results['total_trades'] * 100
            else:
                results['win_rate'] = 0
            
            # 盈亏比
            won_pnl = trades_analyzer.get('won', {}).get('pnl', {}).get('total', 0)
            lost_pnl = abs(trades_analyzer.get('lost', {}).get('pnl', {}).get('total', 0))
            if lost_pnl > 0:
                results['profit_factor'] = won_pnl / lost_pnl
            else:
                results['profit_factor'] = float('inf') if won_pnl > 0 else 0
        
        # 年化收益
        if hasattr(strategy.analyzers, 'annual_return'):
            annual_analyzer = strategy.analyzers.annual_return.get_analysis()
            results['annual_returns'] = annual_analyzer
        
        return results
    
    def plot(
        self,
        style: str = 'candlestick',
        **kwargs
    ) -> None:
        """
        绘制回测结果
        
        Args:
            style: 图表样式
            **kwargs: 其他绘图参数
        """
        try:
            self.cerebro.plot(style=style, **kwargs)
        except Exception as e:
            logger.error(f"绘图失败: {str(e)}")
    
    def reset(self) -> None:
        """重置回测引擎"""
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        self.cerebro.addsizer(bt.sizers.FixedSize, stake=self.stake)
        self._add_analyzers()
        logger.info("回测引擎已重置")


class PandasDataFeed(bt.feeds.PandasData):
    """
    自定义Pandas数据源
    支持额外的特征列
    """
    
    # 定义额外的数据列
    lines = ('feature1', 'feature2', 'feature3')
    
    params = (
        ('feature1', -1),
        ('feature2', -1),
        ('feature3', -1),
    )