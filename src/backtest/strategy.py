"""
PPO策略接口
用于Backtrader回测的策略包装器
"""

import backtrader as bt
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PPOStrategy(bt.Strategy):
    """
    PPO策略的Backtrader包装器
    
    将训练好的PPO模型集成到Backtrader回测框架中
    """
    
    params = (
        ('ppo_model', None),           # PPO模型实例
        ('transformer_model', None),    # Transformer模型实例
        ('feature_scaler', None),       # 特征归一化器
        ('stop_loss', 0.02),           # 止损比例（2%）
        ('take_profit', 0.05),         # 止盈比例（5%）
        ('max_position', 1),           # 最大持仓
        ('verbose', True),             # 是否打印日志
    )
    
    def __init__(self):
        """初始化策略"""
        # 持仓信息
        self.position_size = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        
        # 交易记录
        self.trades = []
        self.orders = []
        
        # 性能指标
        self.total_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info("PPO策略初始化完成")
    
    def log(self, txt, dt=None):
        """日志输出"""
        if self.p.verbose:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} {txt}')
    
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
                self.entry_bar = len(self)
                
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, '
                        f'成本: {order.executed.value:.2f}, '
                        f'手续费: {order.executed.comm:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')
        
        self.order = None
    
    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return
        
        # 记录交易
        pnl = trade.pnl
        pnl_pct = (trade.pnlcomm / trade.price) * 100
        
        self.log(f'交易盈亏: {pnl:.2f} ({pnl_pct:.2f}%)')
        
        # 更新统计
        self.total_pnl += pnl
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # 保存交易记录
        trade_record = {
            'entry_date': bt.num2date(trade.dtopen),
            'exit_date': bt.num2date(trade.dtclose),
            'entry_price': trade.price,
            'exit_price': trade.price + pnl / trade.size,
            'size': trade.size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'bars_held': trade.barlen
        }
        self.trades.append(trade_record)
    
    def next(self):
        """策略主逻辑"""
        # 获取当前数据
        current_price = self.datas[0].close[0]
        
        # 检查止损止盈
        if self.position:
            # 检查止损
            if self.check_stop_loss(current_price):
                self.log(f'触发止损 @ {current_price:.2f}')
                self.close()
                return
            
            # 检查止盈
            if self.check_take_profit(current_price):
                self.log(f'触发止盈 @ {current_price:.2f}')
                self.close()
                return
        
        # 如果有PPO模型，使用模型决策
        if self.p.ppo_model is not None:
            action = self.get_ppo_action()
            self.execute_action(action)
        else:
            # 简单的示例策略（移动平均交叉）
            self.simple_strategy()
    
    def get_ppo_action(self) -> Dict[str, Any]:
        """
        从PPO模型获取动作
        
        Returns:
            动作字典
        """
        # TODO: 实现状态提取和模型推理
        # 这里需要：
        # 1. 提取当前市场状态
        # 2. 使用Transformer生成状态向量
        # 3. 使用PPO模型生成动作
        
        # 占位符实现
        action = {
            'direction': 0,  # 0=平仓, 1=做多, 2=做空
            'position_size': 0.5,
            'stop_loss': self.p.stop_loss,
            'take_profit': self.p.take_profit
        }
        
        return action
    
    def execute_action(self, action: Dict[str, Any]):
        """
        执行PPO动作
        
        Args:
            action: 动作字典
        """
        direction = action['direction']
        
        if direction == 0:  # 平仓
            if self.position:
                self.close()
        
        elif direction == 1:  # 做多
            if not self.position:
                size = int(self.broker.getcash() * action['position_size'] / self.datas[0].close[0])
                if size > 0:
                    self.buy(size=size)
        
        elif direction == 2:  # 做空
            if not self.position:
                size = int(self.broker.getcash() * action['position_size'] / self.datas[0].close[0])
                if size > 0:
                    self.sell(size=size)
    
    def simple_strategy(self):
        """简单的示例策略"""
        # 如果没有持仓
        if not self.position:
            # 简单的买入信号（示例）
            if self.datas[0].close[0] > self.datas[0].close[-1]:
                self.buy()
        else:
            # 简单的卖出信号（示例）
            if self.datas[0].close[0] < self.datas[0].close[-1]:
                self.sell()
    
    def check_stop_loss(self, current_price: float) -> bool:
        """
        检查止损
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止损
        """
        if self.entry_price == 0:
            return False
        
        loss_pct = abs(current_price - self.entry_price) / self.entry_price
        
        if self.position.size > 0:  # 多头
            return current_price < self.entry_price * (1 - self.p.stop_loss)
        else:  # 空头
            return current_price > self.entry_price * (1 + self.p.stop_loss)
    
    def check_take_profit(self, current_price: float) -> bool:
        """
        检查止盈
        
        Args:
            current_price: 当前价格
            
        Returns:
            是否触发止盈
        """
        if self.entry_price == 0:
            return False
        
        if self.position.size > 0:  # 多头
            return current_price > self.entry_price * (1 + self.p.take_profit)
        else:  # 空头
            return current_price < self.entry_price * (1 - self.p.take_profit)
    
    def stop(self):
        """策略结束时调用"""
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades * 100 if total_trades > 0 else 0
        
        self.log(f'策略结束 - 总盈亏: {self.total_pnl:.2f}, '
                f'胜率: {win_rate:.2f}% ({self.win_count}/{total_trades})')