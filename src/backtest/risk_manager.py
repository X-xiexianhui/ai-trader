"""
风险管理器

实现仓位管理、止损止盈、风险控制等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

from .signal_generator import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class PositionSizeMethod(Enum):
    """仓位大小计算方法"""
    FIXED = "fixed"                    # 固定仓位
    PERCENT_EQUITY = "percent_equity"  # 权益百分比
    KELLY = "kelly"                    # 凯利公式
    VOLATILITY = "volatility"          # 波动率调整
    ATR = "atr"                        # ATR调整


class RiskLevel(Enum):
    """风险等级"""
    CONSERVATIVE = 1  # 保守
    MODERATE = 2      # 适中
    AGGRESSIVE = 3    # 激进


class Position:
    """持仓信息"""
    
    def __init__(self,
                 symbol: str,
                 size: float,
                 entry_price: float,
                 entry_time: pd.Timestamp,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None):
        """
        初始化持仓
        
        Args:
            symbol: 交易品种
            size: 持仓数量（正数为多头，负数为空头）
            entry_price: 入场价格
            entry_time: 入场时间
            stop_loss: 止损价格
            take_profit: 止盈价格
        """
        self.symbol = symbol
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
    
    def update_price(self, price: float):
        """更新当前价格"""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.size
    
    def is_long(self) -> bool:
        """是否为多头"""
        return self.size > 0
    
    def is_short(self) -> bool:
        """是否为空头"""
        return self.size < 0
    
    def should_stop_loss(self) -> bool:
        """是否应该止损"""
        if self.stop_loss is None:
            return False
        
        if self.is_long():
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss
    
    def should_take_profit(self) -> bool:
        """是否应该止盈"""
        if self.take_profit is None:
            return False
        
        if self.is_long():
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit
    
    def get_return(self) -> float:
        """获取收益率"""
        return self.unrealized_pnl / (abs(self.size) * self.entry_price)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl,
            'return': self.get_return()
        }


class RiskManager:
    """风险管理器"""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 max_position_size: float = 0.1,
                 max_total_risk: float = 0.02,
                 max_single_risk: float = 0.01,
                 position_size_method: PositionSizeMethod = PositionSizeMethod.PERCENT_EQUITY,
                 risk_level: RiskLevel = RiskLevel.MODERATE,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 trailing_stop: bool = False,
                 trailing_stop_pct: float = 0.01):
        """
        初始化风险管理器
        
        Args:
            initial_capital: 初始资金
            max_position_size: 最大单个仓位占比
            max_total_risk: 最大总风险敞口
            max_single_risk: 单笔交易最大风险
            position_size_method: 仓位计算方法
            risk_level: 风险等级
            stop_loss_pct: 止损百分比
            take_profit_pct: 止盈百分比
            trailing_stop: 是否使用移动止损
            trailing_stop_pct: 移动止损百分比
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_total_risk = max_total_risk
        self.max_single_risk = max_single_risk
        self.position_size_method = position_size_method
        self.risk_level = risk_level
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop = trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        # 持仓管理
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # 交易历史
        self.trades_history = []
        
        # 风险指标
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        logger.info(f"风险管理器初始化，初始资金: {initial_capital}")
    
    def calculate_position_size(self,
                               signal: TradingSignal,
                               current_price: float,
                               volatility: Optional[float] = None,
                               atr: Optional[float] = None) -> float:
        """
        计算仓位大小
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            volatility: 波动率
            atr: ATR值
            
        Returns:
            float: 仓位大小
        """
        if self.position_size_method == PositionSizeMethod.FIXED:
            # 固定仓位
            base_size = self.current_capital * self.max_position_size / current_price
        
        elif self.position_size_method == PositionSizeMethod.PERCENT_EQUITY:
            # 权益百分比
            risk_capital = self.current_capital * self.max_single_risk
            position_value = risk_capital / self.stop_loss_pct
            base_size = position_value / current_price
        
        elif self.position_size_method == PositionSizeMethod.KELLY:
            # 凯利公式
            if self.total_trades > 10:
                win_rate = self.winning_trades / self.total_trades
                avg_win = self._calculate_avg_win()
                avg_loss = self._calculate_avg_loss()
                
                if avg_loss != 0:
                    kelly_pct = win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))
                    kelly_pct = max(0, min(kelly_pct, self.max_position_size))
                else:
                    kelly_pct = self.max_position_size
            else:
                kelly_pct = self.max_position_size / 2
            
            base_size = self.current_capital * kelly_pct / current_price
        
        elif self.position_size_method == PositionSizeMethod.VOLATILITY:
            # 波动率调整
            if volatility is not None and volatility > 0:
                target_volatility = 0.02  # 目标波动率2%
                vol_adjustment = target_volatility / volatility
                vol_adjustment = max(0.5, min(vol_adjustment, 2.0))
            else:
                vol_adjustment = 1.0
            
            base_size = self.current_capital * self.max_position_size * vol_adjustment / current_price
        
        elif self.position_size_method == PositionSizeMethod.ATR:
            # ATR调整
            if atr is not None and atr > 0:
                risk_capital = self.current_capital * self.max_single_risk
                base_size = risk_capital / atr
            else:
                base_size = self.current_capital * self.max_position_size / current_price
        
        else:
            base_size = self.current_capital * self.max_position_size / current_price
        
        # 根据信号强度和置信度调整
        size_multiplier = signal.strength * signal.confidence
        adjusted_size = base_size * size_multiplier
        
        # 确保不超过最大仓位
        max_size = self.current_capital * self.max_position_size / current_price
        adjusted_size = min(adjusted_size, max_size)
        
        return adjusted_size
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           is_long: bool,
                           atr: Optional[float] = None) -> float:
        """
        计算止损价格
        
        Args:
            entry_price: 入场价格
            is_long: 是否为多头
            atr: ATR值
            
        Returns:
            float: 止损价格
        """
        if atr is not None:
            # 使用ATR计算止损
            stop_distance = atr * 2
        else:
            # 使用百分比计算止损
            stop_distance = entry_price * self.stop_loss_pct
        
        if is_long:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_take_profit(self,
                             entry_price: float,
                             is_long: bool,
                             atr: Optional[float] = None) -> float:
        """
        计算止盈价格
        
        Args:
            entry_price: 入场价格
            is_long: 是否为多头
            atr: ATR值
            
        Returns:
            float: 止盈价格
        """
        if atr is not None:
            # 使用ATR计算止盈
            profit_distance = atr * 4
        else:
            # 使用百分比计算止盈
            profit_distance = entry_price * self.take_profit_pct
        
        if is_long:
            take_profit = entry_price + profit_distance
        else:
            take_profit = entry_price - profit_distance
        
        return take_profit
    
    def can_open_position(self, symbol: str, signal: TradingSignal) -> bool:
        """
        检查是否可以开仓
        
        Args:
            symbol: 交易品种
            signal: 交易信号
            
        Returns:
            bool: 是否可以开仓
        """
        # 检查是否已有持仓
        if symbol in self.positions:
            logger.warning(f"{symbol} 已有持仓，不能重复开仓")
            return False
        
        # 检查信号强度
        if signal.confidence < 0.5:
            logger.warning(f"{symbol} 信号置信度过低: {signal.confidence:.2f}")
            return False
        
        # 检查总风险敞口
        total_exposure = sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
        if total_exposure / self.current_capital > self.max_total_risk:
            logger.warning(f"总风险敞口过大: {total_exposure/self.current_capital:.2%}")
            return False
        
        return True
    
    def open_position(self,
                     symbol: str,
                     signal: TradingSignal,
                     current_price: float,
                     timestamp: pd.Timestamp,
                     volatility: Optional[float] = None,
                     atr: Optional[float] = None) -> Optional[Position]:
        """
        开仓
        
        Args:
            symbol: 交易品种
            signal: 交易信号
            current_price: 当前价格
            timestamp: 时间戳
            volatility: 波动率
            atr: ATR值
            
        Returns:
            Optional[Position]: 持仓对象
        """
        if not self.can_open_position(symbol, signal):
            return None
        
        # 计算仓位大小
        size = self.calculate_position_size(signal, current_price, volatility, atr)
        
        # 根据信号类型确定方向
        if signal.signal_type == SignalType.SELL:
            size = -size  # 空头
        
        is_long = size > 0
        
        # 计算止损止盈
        stop_loss = self.calculate_stop_loss(current_price, is_long, atr)
        take_profit = self.calculate_take_profit(current_price, is_long, atr)
        
        # 创建持仓
        position = Position(
            symbol=symbol,
            size=size,
            entry_price=current_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        
        # 记录交易
        self.trades_history.append({
            'action': 'open',
            'symbol': symbol,
            'size': size,
            'price': current_price,
            'timestamp': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
        
        logger.info(f"开仓 {symbol}: size={size:.2f}, price={current_price:.2f}, "
                   f"stop_loss={stop_loss:.2f}, take_profit={take_profit:.2f}")
        
        return position
    
    def close_position(self,
                      symbol: str,
                      current_price: float,
                      timestamp: pd.Timestamp,
                      reason: str = "manual") -> Optional[float]:
        """
        平仓
        
        Args:
            symbol: 交易品种
            current_price: 当前价格
            timestamp: 时间戳
            reason: 平仓原因
            
        Returns:
            Optional[float]: 盈亏
        """
        if symbol not in self.positions:
            logger.warning(f"{symbol} 没有持仓")
            return None
        
        position = self.positions[symbol]
        position.update_price(current_price)
        
        # 计算盈亏
        pnl = position.unrealized_pnl
        
        # 更新资金
        self.current_capital += pnl
        self.total_pnl += pnl
        
        # 更新统计
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # 记录交易
        self.trades_history.append({
            'action': 'close',
            'symbol': symbol,
            'size': position.size,
            'price': current_price,
            'timestamp': timestamp,
            'pnl': pnl,
            'return': position.get_return(),
            'reason': reason
        })
        
        # 移动到已平仓列表
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"平仓 {symbol}: price={current_price:.2f}, pnl={pnl:.2f}, "
                   f"return={position.get_return():.2%}, reason={reason}")
        
        return pnl
    
    def update_positions(self, prices: Dict[str, float], timestamp: pd.Timestamp):
        """
        更新所有持仓
        
        Args:
            prices: 价格字典
            timestamp: 时间戳
        """
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position.update_price(current_price)
                
                # 检查止损止盈
                if position.should_stop_loss():
                    positions_to_close.append((symbol, current_price, "stop_loss"))
                elif position.should_take_profit():
                    positions_to_close.append((symbol, current_price, "take_profit"))
                elif self.trailing_stop:
                    # 更新移动止损
                    self._update_trailing_stop(position)
        
        # 平仓
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, timestamp, reason)
    
    def _update_trailing_stop(self, position: Position):
        """更新移动止损"""
        if position.is_long():
            # 多头：价格上涨时提高止损
            new_stop = position.current_price * (1 - self.trailing_stop_pct)
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:
            # 空头：价格下跌时降低止损
            new_stop = position.current_price * (1 + self.trailing_stop_pct)
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
    
    def _calculate_avg_win(self) -> float:
        """计算平均盈利"""
        wins = [t['pnl'] for t in self.trades_history 
                if t.get('action') == 'close' and t.get('pnl', 0) > 0]
        return np.mean(wins) if wins else 0
    
    def _calculate_avg_loss(self) -> float:
        """计算平均亏损"""
        losses = [t['pnl'] for t in self.trades_history 
                  if t.get('action') == 'close' and t.get('pnl', 0) < 0]
        return np.mean(losses) if losses else 0
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': self.total_pnl,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions)
        }
    
    def get_current_positions(self) -> List[Dict]:
        """获取当前持仓"""
        return [pos.to_dict() for pos in self.positions.values()]


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from .signal_generator import TradingSignal, SignalType
    
    # 创建风险管理器
    risk_manager = RiskManager(
        initial_capital=100000,
        max_position_size=0.1,
        position_size_method=PositionSizeMethod.PERCENT_EQUITY
    )
    
    # 模拟交易
    print("测试风险管理器...")
    
    # 开仓
    signal = TradingSignal(SignalType.BUY, strength=1.5, confidence=0.8)
    timestamp = pd.Timestamp.now()
    position = risk_manager.open_position(
        symbol='BTCUSDT',
        signal=signal,
        current_price=50000,
        timestamp=timestamp,
        atr=1000
    )
    
    if position:
        print(f"\n持仓信息: {position.to_dict()}")
    
    # 更新价格
    risk_manager.update_positions({'BTCUSDT': 51000}, timestamp)
    
    # 平仓
    pnl = risk_manager.close_position('BTCUSDT', 51000, timestamp, "manual")
    print(f"\n平仓盈亏: {pnl:.2f}")
    
    # 统计信息
    stats = risk_manager.get_statistics()
    print("\n统计信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")