"""
回测性能指标计算

包含各种交易性能评估指标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """性能指标计算器"""
    
    def __init__(self, returns: pd.Series, trades: Optional[List[Dict]] = None):
        """
        初始化性能指标计算器
        
        Args:
            returns: 收益率序列
            trades: 交易记录列表
        """
        self.returns = returns
        self.trades = trades
        self.metrics = {}
    
    def calculate_all(self) -> Dict:
        """
        计算所有指标
        
        Returns:
            Dict: 所有性能指标
        """
        self.metrics = {
            # 收益指标
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'annualized_return': self.annualized_return(),
            
            # 风险指标
            'volatility': self.volatility(),
            'max_drawdown': self.max_drawdown(),
            'var_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),
            
            # 风险调整收益
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'information_ratio': self.information_ratio(),
            
            # 交易行为指标
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'max_consecutive_wins': self.max_consecutive_wins(),
            'max_consecutive_losses': self.max_consecutive_losses(),
        }
        
        return self.metrics
    
    def total_return(self) -> float:
        """累积收益率"""
        return (1 + self.returns).prod() - 1
    
    def cagr(self, periods_per_year: int = 252 * 78) -> float:
        """
        年化复合增长率
        
        Args:
            periods_per_year: 每年的周期数（5分钟K线约为252*78）
        """
        total_return = self.total_return()
        n_periods = len(self.returns)
        years = n_periods / periods_per_year
        
        if years > 0:
            return (1 + total_return) ** (1 / years) - 1
        return 0.0
    
    def annualized_return(self, periods_per_year: int = 252 * 78) -> float:
        """年化收益率"""
        return self.returns.mean() * periods_per_year
    
    def volatility(self, periods_per_year: int = 252 * 78) -> float:
        """年化波动率"""
        return self.returns.std() * np.sqrt(periods_per_year)
    
    def max_drawdown(self) -> float:
        """
        最大回撤
        
        Returns:
            float: 最大回撤（负值）
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        风险价值（VaR）
        
        Args:
            confidence: 置信水平
            
        Returns:
            float: VaR值（负值表示损失）
        """
        return self.returns.quantile(1 - confidence)
    
    def conditional_var(self, confidence: float = 0.95) -> float:
        """
        条件风险价值（CVaR/Expected Shortfall）
        
        Args:
            confidence: 置信水平
            
        Returns:
            float: CVaR值
        """
        var = self.value_at_risk(confidence)
        return self.returns[self.returns <= var].mean()
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02, 
                     periods_per_year: int = 252 * 78) -> float:
        """
        夏普比率
        
        Args:
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年的周期数
            
        Returns:
            float: 夏普比率
        """
        excess_returns = self.returns - risk_free_rate / periods_per_year
        if excess_returns.std() == 0:
            return 0.0
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    
    def sortino_ratio(self, risk_free_rate: float = 0.02,
                      periods_per_year: int = 252 * 78) -> float:
        """
        索提诺比率（只考虑下行波动）
        
        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 每年的周期数
            
        Returns:
            float: 索提诺比率
        """
        excess_returns = self.returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / downside_returns.std() * np.sqrt(periods_per_year)
    
    def calmar_ratio(self) -> float:
        """
        卡玛比率（年化收益率/最大回撤）
        
        Returns:
            float: 卡玛比率
        """
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return self.cagr() / max_dd
    
    def information_ratio(self, benchmark_returns: Optional[pd.Series] = None,
                         periods_per_year: int = 252 * 78) -> float:
        """
        信息比率
        
        Args:
            benchmark_returns: 基准收益率
            periods_per_year: 每年的周期数
            
        Returns:
            float: 信息比率
        """
        if benchmark_returns is None:
            benchmark_returns = pd.Series(0, index=self.returns.index)
        
        active_returns = self.returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return active_returns.mean() / tracking_error * np.sqrt(periods_per_year)
    
    def win_rate(self) -> float:
        """
        胜率
        
        Returns:
            float: 胜率（0-1之间）
        """
        if not self.trades:
            # 如果没有交易记录，使用收益率计算
            winning_periods = (self.returns > 0).sum()
            return winning_periods / len(self.returns)
        
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return winning_trades / len(self.trades) if self.trades else 0.0
    
    def profit_factor(self) -> float:
        """
        盈亏比（总盈利/总亏损）
        
        Returns:
            float: 盈亏比
        """
        if not self.trades:
            gross_profit = self.returns[self.returns > 0].sum()
            gross_loss = abs(self.returns[self.returns < 0].sum())
        else:
            gross_profit = sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0)
            gross_loss = abs(sum(t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def avg_win(self) -> float:
        """平均盈利"""
        if not self.trades:
            winning_returns = self.returns[self.returns > 0]
            return winning_returns.mean() if len(winning_returns) > 0 else 0.0
        
        winning_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) > 0]
        return np.mean(winning_trades) if winning_trades else 0.0
    
    def avg_loss(self) -> float:
        """平均亏损"""
        if not self.trades:
            losing_returns = self.returns[self.returns < 0]
            return losing_returns.mean() if len(losing_returns) > 0 else 0.0
        
        losing_trades = [t.get('pnl', 0) for t in self.trades if t.get('pnl', 0) < 0]
        return np.mean(losing_trades) if losing_trades else 0.0
    
    def max_consecutive_wins(self) -> int:
        """最大连续盈利次数"""
        if not self.trades:
            wins = (self.returns > 0).astype(int)
        else:
            wins = pd.Series([1 if t.get('pnl', 0) > 0 else 0 for t in self.trades])
        
        max_streak = 0
        current_streak = 0
        
        for win in wins:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def max_consecutive_losses(self) -> int:
        """最大连续亏损次数"""
        if not self.trades:
            losses = (self.returns < 0).astype(int)
        else:
            losses = pd.Series([1 if t.get('pnl', 0) < 0 else 0 for t in self.trades])
        
        max_streak = 0
        current_streak = 0
        
        for loss in losses:
            if loss:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def print_metrics(self):
        """打印所有指标"""
        if not self.metrics:
            self.calculate_all()
        
        print("\n" + "="*60)
        print("回测性能指标")
        print("="*60)
        
        print("\n收益指标:")
        print(f"  总收益率: {self.metrics['total_return']:.2%}")
        print(f"  年化收益率(CAGR): {self.metrics['cagr']:.2%}")
        print(f"  年化收益率(算术): {self.metrics['annualized_return']:.2%}")
        
        print("\n风险指标:")
        print(f"  年化波动率: {self.metrics['volatility']:.2%}")
        print(f"  最大回撤: {self.metrics['max_drawdown']:.2%}")
        print(f"  VaR(95%): {self.metrics['var_95']:.2%}")
        print(f"  CVaR(95%): {self.metrics['cvar_95']:.2%}")
        
        print("\n风险调整收益:")
        print(f"  夏普比率: {self.metrics['sharpe_ratio']:.3f}")
        print(f"  索提诺比率: {self.metrics['sortino_ratio']:.3f}")
        print(f"  卡玛比率: {self.metrics['calmar_ratio']:.3f}")
        print(f"  信息比率: {self.metrics['information_ratio']:.3f}")
        
        print("\n交易行为指标:")
        print(f"  胜率: {self.metrics['win_rate']:.2%}")
        print(f"  盈亏比: {self.metrics['profit_factor']:.3f}")
        print(f"  平均盈利: {self.metrics['avg_win']:.4f}")
        print(f"  平均亏损: {self.metrics['avg_loss']:.4f}")
        print(f"  最大连续盈利: {self.metrics['max_consecutive_wins']}")
        print(f"  最大连续亏损: {self.metrics['max_consecutive_losses']}")
        
        print("="*60 + "\n")


def calculate_metrics(returns: pd.Series, trades: Optional[List[Dict]] = None) -> Dict:
    """
    计算性能指标的便捷函数
    
    Args:
        returns: 收益率序列
        trades: 交易记录列表
        
    Returns:
        Dict: 性能指标字典
    """
    calculator = PerformanceMetrics(returns, trades)
    return calculator.calculate_all()


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    n_periods = 10000
    returns = pd.Series(np.random.randn(n_periods) * 0.01 + 0.0001)
    
    # 创建测试交易记录
    trades = []
    for i in range(100):
        pnl = np.random.randn() * 100
        trades.append({
            'pnl': pnl,
            'pnl_pct': pnl / 10000,
            'holding_time': np.random.randint(1, 100)
        })
    
    # 计算指标
    calculator = PerformanceMetrics(returns, trades)
    metrics = calculator.calculate_all()
    
    # 打印结果
    calculator.print_metrics()