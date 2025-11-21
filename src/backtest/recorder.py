"""
回测结果记录模块
实现详细的回测结果记录和可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BacktestRecorder:
    """
    回测结果记录器
    
    功能：
    1. 记录每笔交易
    2. 记录账户权益曲线
    3. 记录持仓变化
    4. 生成性能指标
    5. 可视化结果
    """
    
    def __init__(self, output_dir: str = 'results/backtest'):
        """
        初始化记录器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.positions: List[Dict] = []
        self.orders: List[Dict] = []
        
        logger.info(f"回测记录器初始化，输出目录: {self.output_dir}")
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        记录交易
        
        Args:
            trade: 交易信息字典
        """
        self.trades.append(trade)
        logger.debug(f"记录交易: {trade}")
    
    def record_equity(
        self,
        timestamp: datetime,
        equity: float,
        cash: float,
        position_value: float
    ) -> None:
        """
        记录权益
        
        Args:
            timestamp: 时间戳
            equity: 总权益
            cash: 现金
            position_value: 持仓价值
        """
        record = {
            'timestamp': timestamp,
            'equity': equity,
            'cash': cash,
            'position_value': position_value
        }
        self.equity_curve.append(record)
    
    def record_position(
        self,
        timestamp: datetime,
        symbol: str,
        size: int,
        price: float,
        value: float
    ) -> None:
        """
        记录持仓
        
        Args:
            timestamp: 时间戳
            symbol: 品种代码
            size: 持仓数量
            price: 持仓价格
            value: 持仓价值
        """
        record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'size': size,
            'price': price,
            'value': value
        }
        self.positions.append(record)
    
    def record_order(self, order: Dict[str, Any]) -> None:
        """
        记录订单
        
        Args:
            order: 订单信息字典
        """
        self.orders.append(order)
        logger.debug(f"记录订单: {order}")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        计算性能指标
        
        Returns:
            指标字典
        """
        if not self.trades or not self.equity_curve:
            logger.warning("没有足够的数据计算指标")
            return {}
        
        # 转换为DataFrame
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        metrics = {}
        
        # 基本指标
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['pnl'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['pnl'] < 0])
        
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] * 100
        else:
            metrics['win_rate'] = 0
        
        # 盈亏指标
        metrics['total_pnl'] = trades_df['pnl'].sum()
        metrics['avg_pnl'] = trades_df['pnl'].mean()
        metrics['max_win'] = trades_df['pnl'].max()
        metrics['max_loss'] = trades_df['pnl'].min()
        
        # 盈亏比
        winning_pnl = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        losing_pnl = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        if losing_pnl > 0:
            metrics['profit_factor'] = winning_pnl / losing_pnl
        else:
            metrics['profit_factor'] = float('inf') if winning_pnl > 0 else 0
        
        # 收益率指标
        initial_equity = equity_df['equity'].iloc[0]
        final_equity = equity_df['equity'].iloc[-1]
        metrics['total_return'] = (final_equity / initial_equity - 1) * 100
        
        # 计算日收益率
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # 夏普比率（假设无风险利率为0）
        if len(equity_df) > 1:
            returns_std = equity_df['returns'].std()
            if returns_std > 0:
                metrics['sharpe_ratio'] = equity_df['returns'].mean() / returns_std * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
        else:
            metrics['sharpe_ratio'] = 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        metrics['max_drawdown'] = equity_df['drawdown'].min()
        
        # 持仓时长
        if 'bars_held' in trades_df.columns:
            metrics['avg_bars_held'] = trades_df['bars_held'].mean()
            metrics['max_bars_held'] = trades_df['bars_held'].max()
            metrics['min_bars_held'] = trades_df['bars_held'].min()
        
        logger.info(f"计算完成，共 {len(metrics)} 个指标")
        
        return metrics
    
    def generate_report(self, filename: str = 'backtest_report.json') -> str:
        """
        生成回测报告
        
        Args:
            filename: 报告文件名
            
        Returns:
            报告文件路径
        """
        # 计算指标
        metrics = self.calculate_metrics()
        
        # 构建报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'trades_count': len(self.trades),
            'equity_points': len(self.equity_curve),
            'summary': {
                'total_return': f"{metrics.get('total_return', 0):.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'max_drawdown': f"{metrics.get('max_drawdown', 0):.2f}%",
                'win_rate': f"{metrics.get('win_rate', 0):.2f}%",
                'profit_factor': f"{metrics.get('profit_factor', 0):.2f}"
            }
        }
        
        # 保存报告
        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"报告已保存: {report_path}")
        
        return str(report_path)
    
    def save_trades(self, filename: str = 'trades.csv') -> str:
        """
        保存交易记录
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if not self.trades:
            logger.warning("没有交易记录")
            return ""
        
        trades_df = pd.DataFrame(self.trades)
        filepath = self.output_dir / filename
        trades_df.to_csv(filepath, index=False)
        
        logger.info(f"交易记录已保存: {filepath}")
        
        return str(filepath)
    
    def save_equity_curve(self, filename: str = 'equity_curve.csv') -> str:
        """
        保存权益曲线
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if not self.equity_curve:
            logger.warning("没有权益数据")
            return ""
        
        equity_df = pd.DataFrame(self.equity_curve)
        filepath = self.output_dir / filename
        equity_df.to_csv(filepath, index=False)
        
        logger.info(f"权益曲线已保存: {filepath}")
        
        return str(filepath)
    
    def plot_equity_curve(self, filename: str = 'equity_curve.png') -> str:
        """
        绘制权益曲线
        
        Args:
            filename: 文件名
            
        Returns:
            图片路径
        """
        if not self.equity_curve:
            logger.warning("没有权益数据")
            return ""
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['timestamp'], equity_df['equity'], label='总权益', linewidth=2)
        plt.plot(equity_df['timestamp'], equity_df['cash'], label='现金', alpha=0.7)
        plt.plot(equity_df['timestamp'], equity_df['position_value'], label='持仓价值', alpha=0.7)
        
        plt.xlabel('时间')
        plt.ylabel('金额')
        plt.title('账户权益曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"权益曲线图已保存: {filepath}")
        
        return str(filepath)
    
    def plot_drawdown(self, filename: str = 'drawdown.png') -> str:
        """
        绘制回撤曲线
        
        Args:
            filename: 文件名
            
        Returns:
            图片路径
        """
        if not self.equity_curve:
            logger.warning("没有权益数据")
            return ""
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, 
                        alpha=0.3, color='red', label='回撤')
        plt.plot(equity_df['timestamp'], equity_df['drawdown'], 
                color='red', linewidth=2)
        
        plt.xlabel('时间')
        plt.ylabel('回撤 (%)')
        plt.title('回撤曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"回撤曲线图已保存: {filepath}")
        
        return str(filepath)
    
    def plot_trade_distribution(self, filename: str = 'trade_distribution.png') -> str:
        """
        绘制交易分布
        
        Args:
            filename: 文件名
            
        Returns:
            图片路径
        """
        if not self.trades:
            logger.warning("没有交易记录")
            return ""
        
        trades_df = pd.DataFrame(self.trades)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 盈亏分布
        axes[0, 0].hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('盈亏')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('盈亏分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 盈亏百分比分布
        axes[0, 1].hist(trades_df['pnl_pct'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('盈亏 (%)')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('盈亏百分比分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 持仓时长分布
        if 'bars_held' in trades_df.columns:
            axes[1, 0].hist(trades_df['bars_held'], bins=30, edgecolor='black', alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('持仓K线数')
            axes[1, 0].set_ylabel('频数')
            axes[1, 0].set_title('持仓时长分布')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 累计盈亏
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        axes[1, 1].plot(range(len(trades_df)), trades_df['cumulative_pnl'], linewidth=2)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1, 1].set_xlabel('交易序号')
        axes[1, 1].set_ylabel('累计盈亏')
        axes[1, 1].set_title('累计盈亏曲线')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交易分布图已保存: {filepath}")
        
        return str(filepath)
    
    def generate_full_report(self) -> Dict[str, str]:
        """
        生成完整报告（包括所有图表和数据）
        
        Returns:
            文件路径字典
        """
        logger.info("生成完整回测报告...")
        
        files = {}
        
        # 保存数据
        files['report'] = self.generate_report()
        files['trades'] = self.save_trades()
        files['equity_curve'] = self.save_equity_curve()
        
        # 生成图表
        files['equity_plot'] = self.plot_equity_curve()
        files['drawdown_plot'] = self.plot_drawdown()
        files['distribution_plot'] = self.plot_trade_distribution()
        
        logger.info(f"完整报告生成完成，共 {len(files)} 个文件")
        
        return files
    
    def clear(self) -> None:
        """清空所有记录"""
        self.trades.clear()
        self.equity_curve.clear()
        self.positions.clear()
        self.orders.clear()
        logger.info("记录已清空")