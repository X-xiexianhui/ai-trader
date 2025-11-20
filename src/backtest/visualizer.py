"""
回测可视化

使用matplotlib和plotly生成交互式图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestVisualizer:
    """回测可视化器"""
    
    def __init__(self, output_dir: str = "plots", style: str = "seaborn-v0_8-darkgrid"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
            style: matplotlib样式
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置样式
        try:
            plt.style.use(style)
        except:
            logger.warning(f"样式 {style} 不可用，使用默认样式")
        
        logger.info(f"可视化器初始化，输出目录: {self.output_dir}")
    
    def plot_equity_curve(self,
                         equity_curve: pd.Series,
                         benchmark: Optional[pd.Series] = None,
                         title: str = "权益曲线",
                         save_path: Optional[str] = None) -> str:
        """
        绘制权益曲线
        
        Args:
            equity_curve: 权益曲线
            benchmark: 基准曲线
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制权益曲线
        ax.plot(equity_curve.index, equity_curve.values, 
               label='策略权益', linewidth=2, color='#2E86AB')
        
        # 绘制基准曲线
        if benchmark is not None:
            ax.plot(benchmark.index, benchmark.values,
                   label='基准', linewidth=2, color='#A23B72', alpha=0.7)
        
        # 添加初始资金线
        ax.axhline(y=equity_curve.iloc[0], color='gray', 
                  linestyle='--', alpha=0.5, label='初始资金')
        
        # 格式化
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('权益', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "equity_curve.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"权益曲线已保存: {save_path}")
        return str(save_path)
    
    def plot_drawdown(self,
                     equity_curve: pd.Series,
                     title: str = "回撤分析",
                     save_path: Optional[str] = None) -> str:
        """
        绘制回撤图
        
        Args:
            equity_curve: 权益曲线
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 计算回撤
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # 上图：权益曲线
        ax1.plot(equity_curve.index, equity_curve.values, 
                label='权益曲线', linewidth=2, color='#2E86AB')
        ax1.plot(cummax.index, cummax.values,
                label='历史最高', linewidth=1, linestyle='--', 
                color='#F18F01', alpha=0.7)
        ax1.set_ylabel('权益', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 下图：回撤
        ax2.fill_between(drawdown.index, drawdown.values, 0,
                        color='#C73E1D', alpha=0.3, label='回撤')
        ax2.plot(drawdown.index, drawdown.values,
                color='#C73E1D', linewidth=1)
        ax2.set_xlabel('日期', fontsize=12)
        ax2.set_ylabel('回撤', fontsize=12)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 格式化y轴为百分比
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "drawdown.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"回撤图已保存: {save_path}")
        return str(save_path)
    
    def plot_returns_distribution(self,
                                 returns: pd.Series,
                                 title: str = "收益率分布",
                                 save_path: Optional[str] = None) -> str:
        """
        绘制收益率分布图
        
        Args:
            returns: 收益率序列
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：直方图
        ax1.hist(returns, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.axvline(returns.mean(), color='#F18F01', linestyle='--', 
                   linewidth=2, label=f'均值: {returns.mean():.4f}')
        ax1.axvline(returns.median(), color='#C73E1D', linestyle='--',
                   linewidth=2, label=f'中位数: {returns.median():.4f}')
        ax1.set_xlabel('收益率', fontsize=12)
        ax1.set_ylabel('频数', fontsize=12)
        ax1.set_title('收益率直方图', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 右图：Q-Q图
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q图（正态性检验）', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "returns_distribution.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"收益率分布图已保存: {save_path}")
        return str(save_path)
    
    def plot_monthly_returns(self,
                           returns: pd.Series,
                           title: str = "月度收益热力图",
                           save_path: Optional[str] = None) -> str:
        """
        绘制月度收益热力图
        
        Args:
            returns: 收益率序列
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 计算月度收益
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建年月矩阵
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto',
                      vmin=-0.1, vmax=0.1)
        
        # 设置刻度
        ax.set_xticks(np.arange(len(pivot_table.columns)))
        ax.set_yticks(np.arange(len(pivot_table.index)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticklabels(pivot_table.index)
        
        # 添加数值标签
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.1%}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('年份', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('收益率', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "monthly_returns.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"月度收益热力图已保存: {save_path}")
        return str(save_path)
    
    def plot_trade_analysis(self,
                          trades: List[Dict],
                          title: str = "交易分析",
                          save_path: Optional[str] = None) -> str:
        """
        绘制交易分析图
        
        Args:
            trades: 交易记录
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 提取平仓交易
        closed_trades = [t for t in trades if t.get('action') == 'close']
        
        if not closed_trades:
            logger.warning("没有平仓交易，无法生成交易分析图")
            return ""
        
        # 提取数据
        pnls = [t.get('pnl', 0) for t in closed_trades]
        returns = [t.get('return', 0) for t in closed_trades]
        timestamps = [t.get('timestamp') for t in closed_trades]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 累积盈亏
        cumulative_pnl = np.cumsum(pnls)
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                linewidth=2, color='#2E86AB')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('交易次数', fontsize=11)
        ax1.set_ylabel('累积盈亏', fontsize=11)
        ax1.set_title('累积盈亏曲线', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 单笔盈亏分布
        colors = ['#06A77D' if p > 0 else '#C73E1D' for p in pnls]
        ax2.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('交易次数', fontsize=11)
        ax2.set_ylabel('单笔盈亏', fontsize=11)
        ax2.set_title('单笔盈亏分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 盈亏直方图
        ax3.hist(pnls, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(pnls), color='#F18F01', linestyle='--',
                   linewidth=2, label=f'均值: {np.mean(pnls):.2f}')
        ax3.set_xlabel('盈亏', fontsize=11)
        ax3.set_ylabel('频数', fontsize=11)
        ax3.set_title('盈亏分布直方图', fontsize=12, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 胜率统计
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        
        ax4.pie([wins, losses], labels=['盈利', '亏损'],
               colors=['#06A77D', '#C73E1D'], autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 11})
        ax4.set_title(f'胜率统计 (总交易: {len(pnls)})', 
                     fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "trade_analysis.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交易分析图已保存: {save_path}")
        return str(save_path)
    
    def plot_performance_summary(self,
                               results: Dict,
                               title: str = "性能摘要",
                               save_path: Optional[str] = None) -> str:
        """
        绘制性能摘要图
        
        Args:
            results: 回测结果
            title: 图表标题
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 关键指标
        metrics = {
            '总收益率': results.get('total_return_pct', 0),
            '年化收益': results.get('annual_return', 0),
            '最大回撤': results.get('max_drawdown', 0),
            '夏普比率': results.get('sharpe_ratio', 0) / 10,  # 缩放以便显示
            '胜率': results.get('win_rate', 0)
        }
        
        colors = ['#06A77D' if v > 0 else '#C73E1D' for v in metrics.values()]
        ax1.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.7)
        ax1.set_xlabel('数值', fontsize=11)
        ax1.set_title('关键指标', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. 风险调整收益
        risk_metrics = {
            '夏普比率': results.get('sharpe_ratio', 0),
            '索提诺比率': results.get('sortino_ratio', 0),
            '卡玛比率': results.get('calmar_ratio', 0),
            '信息比率': results.get('information_ratio', 0)
        }
        
        ax2.bar(range(len(risk_metrics)), list(risk_metrics.values()),
               color='#2E86AB', alpha=0.7)
        ax2.set_xticks(range(len(risk_metrics)))
        ax2.set_xticklabels(list(risk_metrics.keys()), rotation=45, ha='right')
        ax2.set_ylabel('比率', fontsize=11)
        ax2.set_title('风险调整收益', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 交易统计
        trade_stats = {
            '总交易': results.get('total_trades', 0),
            '盈利交易': results.get('winning_trades', 0),
            '亏损交易': results.get('losing_trades', 0)
        }
        
        colors_trade = ['#2E86AB', '#06A77D', '#C73E1D']
        ax3.bar(range(len(trade_stats)), list(trade_stats.values()),
               color=colors_trade, alpha=0.7)
        ax3.set_xticks(range(len(trade_stats)))
        ax3.set_xticklabels(list(trade_stats.keys()))
        ax3.set_ylabel('次数', fontsize=11)
        ax3.set_title('交易统计', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 盈亏对比
        avg_win = results.get('avg_win', 0)
        avg_loss = abs(results.get('avg_loss', 0))
        
        ax4.bar(['平均盈利', '平均亏损'], [avg_win, avg_loss],
               color=['#06A77D', '#C73E1D'], alpha=0.7)
        ax4.set_ylabel('金额', fontsize=11)
        ax4.set_title('平均盈亏对比', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # 保存
        if save_path is None:
            save_path = self.output_dir / "performance_summary.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能摘要图已保存: {save_path}")
        return str(save_path)
    
    def generate_all_plots(self,
                          equity_curve: pd.Series,
                          returns: pd.Series,
                          trades: List[Dict],
                          results: Dict,
                          benchmark: Optional[pd.Series] = None) -> Dict[str, str]:
        """
        生成所有图表
        
        Args:
            equity_curve: 权益曲线
            returns: 收益率序列
            trades: 交易记录
            results: 回测结果
            benchmark: 基准曲线
            
        Returns:
            Dict[str, str]: 生成的图表文件路径
        """
        plots = {}
        
        logger.info("开始生成所有图表...")
        
        plots['equity_curve'] = self.plot_equity_curve(equity_curve, benchmark)
        plots['drawdown'] = self.plot_drawdown(equity_curve)
        plots['returns_distribution'] = self.plot_returns_distribution(returns)
        plots['monthly_returns'] = self.plot_monthly_returns(returns)
        plots['trade_analysis'] = self.plot_trade_analysis(trades)
        plots['performance_summary'] = self.plot_performance_summary(results)
        
        logger.info(f"所有图表生成完成，共 {len(plots)} 个")
        
        return plots


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # 权益曲线
    returns = np.random.randn(252) * 0.02 + 0.001
    equity_curve = pd.Series((1 + returns).cumprod() * 100000, index=dates)
    
    # 收益率
    returns_series = pd.Series(returns, index=dates)
    
    # 交易记录
    trades = []
    for i in range(50):
        trades.append({
            'timestamp': dates[i * 5],
            'action': 'close',
            'pnl': np.random.randn() * 1000,
            'return': np.random.randn() * 0.02
        })
    
    # 回测结果
    results = {
        'total_return_pct': 0.15,
        'annual_return': 0.20,
        'max_drawdown': -0.10,
        'sharpe_ratio': 1.5,
        'sortino_ratio': 2.0,
        'calmar_ratio': 2.5,
        'information_ratio': 1.2,
        'total_trades': 50,
        'winning_trades': 30,
        'losing_trades': 20,
        'win_rate': 0.60,
        'avg_win': 500,
        'avg_loss': -300
    }
    
    # 创建可视化器
    visualizer = BacktestVisualizer()
    
    # 生成所有图表
    plots = visualizer.generate_all_plots(
        equity_curve=equity_curve,
        returns=returns_series,
        trades=trades,
        results=results
    )
    
    print("\n生成的图表:")
    for name, path in plots.items():
        print(f"  {name}: {path}")