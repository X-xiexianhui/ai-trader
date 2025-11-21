"""
订单执行模拟模块
实现订单执行、滑点和手续费模拟
"""

import backtrader as bt
import numpy as np
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SlippageModel(bt.CommInfoBase):
    """
    滑点模型
    
    模拟实际交易中的价格滑点
    """
    
    params = (
        ('slippage_perc', 0.0005),  # 滑点百分比（0.05%）
        ('slippage_fixed', 0.0),     # 固定滑点
        ('slip_open', True),         # 开仓是否有滑点
        ('slip_limit', True),        # 限价单是否有滑点
        ('slip_match', True),        # 是否匹配滑点
        ('slip_out', False),         # 平仓是否有滑点
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """
        计算手续费（这里主要用于滑点）
        
        Args:
            size: 交易数量
            price: 交易价格
            pseudoexec: 是否是模拟执行
            
        Returns:
            手续费金额
        """
        return 0.0  # 手续费在CommissionModel中计算
    
    def get_slippage(self, price: float, size: int) -> float:
        """
        计算滑点
        
        Args:
            price: 原始价格
            size: 交易数量（正数=买入，负数=卖出）
            
        Returns:
            滑点后的价格
        """
        # 计算百分比滑点
        perc_slip = price * self.p.slippage_perc
        
        # 买入时价格上滑，卖出时价格下滑
        if size > 0:  # 买入
            slippage = perc_slip + self.p.slippage_fixed
        else:  # 卖出
            slippage = -(perc_slip + self.p.slippage_fixed)
        
        slipped_price = price + slippage
        
        logger.debug(f"滑点计算: 原价={price:.4f}, 滑点={slippage:.4f}, 成交价={slipped_price:.4f}")
        
        return slipped_price


class CommissionModel(bt.CommInfoBase):
    """
    手续费模型
    
    支持：
    1. 比例手续费
    2. 固定手续费
    3. 最小手续费
    """
    
    params = (
        ('commission', 0.001),      # 手续费率（0.1%）
        ('commtype', bt.CommInfoBase.COMM_PERC),  # 手续费类型
        ('stocklike', True),        # 股票模式
        ('percabs', False),         # 百分比是否为绝对值
        ('min_commission', 0.0),    # 最小手续费
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """
        计算手续费
        
        Args:
            size: 交易数量
            price: 交易价格
            pseudoexec: 是否是模拟执行
            
        Returns:
            手续费金额
        """
        # 计算基础手续费
        if self.p.commtype == bt.CommInfoBase.COMM_PERC:
            # 百分比手续费
            commission = abs(size) * price * self.p.commission
        else:
            # 固定手续费
            commission = abs(size) * self.p.commission
        
        # 应用最小手续费
        commission = max(commission, self.p.min_commission)
        
        logger.debug(f"手续费计算: 数量={size}, 价格={price:.4f}, 手续费={commission:.4f}")
        
        return commission


class OrderExecutor:
    """
    订单执行器
    
    功能：
    1. 市价单执行
    2. 限价单执行
    3. 止损止盈触发
    4. 部分成交处理
    """
    
    def __init__(
        self,
        slippage_perc: float = 0.0005,
        commission: float = 0.001,
        min_commission: float = 0.0
    ):
        """
        初始化订单执行器
        
        Args:
            slippage_perc: 滑点百分比
            commission: 手续费率
            min_commission: 最小手续费
        """
        self.slippage_perc = slippage_perc
        self.commission = commission
        self.min_commission = min_commission
        
        # 创建滑点模型
        self.slippage_model = SlippageModel(
            slippage_perc=slippage_perc
        )
        
        # 创建手续费模型
        self.commission_model = CommissionModel(
            commission=commission,
            min_commission=min_commission
        )
    
    def execute_market_order(
        self,
        price: float,
        size: int,
        order_type: str = 'buy'
    ) -> Dict[str, Any]:
        """
        执行市价单
        
        Args:
            price: 当前市场价格
            size: 交易数量
            order_type: 订单类型（'buy'或'sell'）
            
        Returns:
            执行结果字典
        """
        # 应用滑点
        if order_type == 'buy':
            exec_price = self.slippage_model.get_slippage(price, size)
        else:
            exec_price = self.slippage_model.get_slippage(price, -size)
        
        # 计算手续费
        commission = self.commission_model._getcommission(size, exec_price, False)
        
        # 计算总成本
        if order_type == 'buy':
            total_cost = size * exec_price + commission
        else:
            total_cost = size * exec_price - commission
        
        result = {
            'order_type': order_type,
            'size': size,
            'requested_price': price,
            'executed_price': exec_price,
            'slippage': exec_price - price,
            'commission': commission,
            'total_cost': total_cost,
            'status': 'executed'
        }
        
        logger.info(f"市价单执行: {order_type} {size}@{exec_price:.4f}, 手续费={commission:.4f}")
        
        return result
    
    def execute_limit_order(
        self,
        current_price: float,
        limit_price: float,
        size: int,
        order_type: str = 'buy'
    ) -> Optional[Dict[str, Any]]:
        """
        执行限价单
        
        Args:
            current_price: 当前市场价格
            limit_price: 限价
            size: 交易数量
            order_type: 订单类型
            
        Returns:
            执行结果字典，未触发返回None
        """
        # 检查是否触发
        if order_type == 'buy':
            if current_price > limit_price:
                return None  # 买入限价单未触发
            exec_price = limit_price
        else:
            if current_price < limit_price:
                return None  # 卖出限价单未触发
            exec_price = limit_price
        
        # 计算手续费
        commission = self.commission_model._getcommission(size, exec_price, False)
        
        # 计算总成本
        if order_type == 'buy':
            total_cost = size * exec_price + commission
        else:
            total_cost = size * exec_price - commission
        
        result = {
            'order_type': order_type,
            'size': size,
            'limit_price': limit_price,
            'executed_price': exec_price,
            'commission': commission,
            'total_cost': total_cost,
            'status': 'executed'
        }
        
        logger.info(f"限价单执行: {order_type} {size}@{exec_price:.4f}, 手续费={commission:.4f}")
        
        return result
    
    def check_stop_loss(
        self,
        current_price: float,
        entry_price: float,
        stop_loss_pct: float,
        position_type: str = 'long'
    ) -> bool:
        """
        检查止损触发
        
        Args:
            current_price: 当前价格
            entry_price: 入场价格
            stop_loss_pct: 止损百分比
            position_type: 持仓类型（'long'或'short'）
            
        Returns:
            是否触发止损
        """
        if position_type == 'long':
            # 多头止损：价格下跌超过止损比例
            loss_pct = (entry_price - current_price) / entry_price
            triggered = loss_pct >= stop_loss_pct
        else:
            # 空头止损：价格上涨超过止损比例
            loss_pct = (current_price - entry_price) / entry_price
            triggered = loss_pct >= stop_loss_pct
        
        if triggered:
            logger.info(f"止损触发: {position_type} 持仓，亏损={loss_pct*100:.2f}%")
        
        return triggered
    
    def check_take_profit(
        self,
        current_price: float,
        entry_price: float,
        take_profit_pct: float,
        position_type: str = 'long'
    ) -> bool:
        """
        检查止盈触发
        
        Args:
            current_price: 当前价格
            entry_price: 入场价格
            take_profit_pct: 止盈百分比
            position_type: 持仓类型
            
        Returns:
            是否触发止盈
        """
        if position_type == 'long':
            # 多头止盈：价格上涨超过止盈比例
            profit_pct = (current_price - entry_price) / entry_price
            triggered = profit_pct >= take_profit_pct
        else:
            # 空头止盈：价格下跌超过止盈比例
            profit_pct = (entry_price - current_price) / entry_price
            triggered = profit_pct >= take_profit_pct
        
        if triggered:
            logger.info(f"止盈触发: {position_type} 持仓，盈利={profit_pct*100:.2f}%")
        
        return triggered
    
    def calculate_position_pnl(
        self,
        entry_price: float,
        current_price: float,
        size: int,
        position_type: str = 'long'
    ) -> Dict[str, float]:
        """
        计算持仓盈亏
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            size: 持仓数量
            position_type: 持仓类型
            
        Returns:
            盈亏信息字典
        """
        if position_type == 'long':
            pnl = (current_price - entry_price) * size
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - current_price) * size
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        return {
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_price': entry_price,
            'current_price': current_price,
            'size': size,
            'position_type': position_type
        }