"""
评估工具集

包含多Seed稳定性测试、压力测试、基准策略对比等功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple, Any
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MultiSeedStabilityTest:
    """
    多Seed稳定性测试
    
    使用多个随机种子训练模型，评估结果稳定性
    """
    
    def __init__(
        self,
        train_func: Callable,
        eval_func: Callable,
        n_seeds: int = 10,
        base_seed: int = 42
    ):
        """
        初始化多Seed稳定性测试
        
        Args:
            train_func: 训练函数，接收(data, seed, **kwargs)，返回模型
            eval_func: 评估函数，接收(model, data, **kwargs)，返回指标字典
            n_seeds: 测试的种子数量
            base_seed: 基础随机种子
        """
        self.train_func = train_func
        self.eval_func = eval_func
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        
        self.results = []
        self.statistics = {}
        
        logger.info(f"多Seed稳定性测试初始化: {n_seeds}个种子")
    
    def run_test(
        self,
        train_data: Any,
        test_data: Any,
        **kwargs
    ) -> Dict:
        """
        执行多Seed测试
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            **kwargs: 传递给train_func和eval_func的额外参数
            
        Returns:
            Dict: 测试结果统计
        """
        self.results = []
        
        for i in range(self.n_seeds):
            seed = self.base_seed + i
            logger.info(f"\n测试 Seed {i+1}/{self.n_seeds} (seed={seed})")
            
            try:
                # 训练模型
                model = self.train_func(train_data, seed=seed, **kwargs)
                
                # 评估模型
                metrics = self.eval_func(model, test_data, **kwargs)
                
                self.results.append({
                    'seed': seed,
                    'metrics': metrics,
                    'model': model
                })
                
                logger.info(f"Seed {seed} 完成: {metrics}")
                
            except Exception as e:
                logger.error(f"Seed {seed} 失败: {str(e)}")
                continue
        
        # 计算统计量
        self.statistics = self._calculate_statistics()
        
        logger.info(f"\n多Seed测试完成，成功{len(self.results)}/{self.n_seeds}个")
        return self.statistics
    
    def _calculate_statistics(self) -> Dict:
        """计算统计量"""
        if not self.results:
            return {}
        
        # 提取所有指标
        all_metrics = {}
        for result in self.results:
            for metric_name, value in result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # 计算统计量
        statistics = {}
        for metric_name, values in all_metrics.items():
            values = np.array(values)
            
            statistics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf'),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'is_stable': float(np.std(values) / np.mean(values)) < 0.1 if np.mean(values) != 0 else False
            }
        
        return statistics
    
    def print_summary(self):
        """打印稳定性测试摘要"""
        if not self.statistics:
            print("没有测试结果")
            return
        
        print("\n" + "="*80)
        print("多Seed稳定性测试摘要")
        print("="*80)
        
        print(f"\n测试配置:")
        print(f"  种子数量: {self.n_seeds}")
        print(f"  成功数量: {len(self.results)}")
        print(f"  基础种子: {self.base_seed}")
        
        print(f"\n性能统计:")
        print("-" * 80)
        print(f"{'指标':<20} | {'均值':>10} | {'标准差':>10} | {'CV':>8} | {'范围':>20} | {'稳定性':>8}")
        print("-" * 80)
        
        for metric_name, stats in self.statistics.items():
            stability = "✓ 稳定" if stats['is_stable'] else "✗ 不稳定"
            range_str = f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            
            print(f"{metric_name:<20} | "
                  f"{stats['mean']:>10.4f} | "
                  f"{stats['std']:>10.4f} | "
                  f"{stats['cv']:>8.4f} | "
                  f"{range_str:>20} | "
                  f"{stability:>8}")
        
        print("="*80 + "\n")
    
    def plot_distributions(self, save_path: Optional[str] = None):
        """
        绘制指标分布图
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            n_metrics = len(self.statistics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
            
            if n_metrics == 1:
                axes = [axes]
            
            for idx, (metric_name, stats) in enumerate(self.statistics.items()):
                ax = axes[idx]
                
                # 提取该指标的所有值
                values = [r['metrics'][metric_name] for r in self.results]
                
                # 绘制直方图和箱线图
                ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.4f}")
                ax.axvline(stats['median'], color='green', linestyle='--', label=f"Median: {stats['median']:.4f}")
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric_name}\nCV: {stats["cv"]:.4f}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"分布图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")


class StressTest:
    """
    压力测试
    
    在极端市场条件下测试模型表现
    """
    
    def __init__(self):
        """初始化压力测试"""
        self.test_results = {}
        logger.info("压力测试初始化")
    
    def run_stress_tests(
        self,
        model: Any,
        data: pd.DataFrame,
        eval_func: Callable,
        extreme_events: Optional[List[Tuple[str, str]]] = None
    ) -> Dict:
        """
        执行压力测试
        
        Args:
            model: 训练好的模型
            data: 完整数据
            eval_func: 评估函数
            extreme_events: 极端事件列表 [(名称, 开始日期, 结束日期), ...]
            
        Returns:
            Dict: 压力测试结果
        """
        self.test_results = {}
        
        # 1. 历史极端事件测试
        if extreme_events:
            logger.info("测试历史极端事件...")
            for event_name, start_date, end_date in extreme_events:
                event_data = data.loc[start_date:end_date]
                if len(event_data) > 0:
                    metrics = eval_func(model, event_data)
                    self.test_results[f'event_{event_name}'] = metrics
                    logger.info(f"  {event_name}: {metrics}")
        
        # 2. 高波动期测试
        logger.info("测试高波动期...")
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        high_vol_threshold = volatility.quantile(0.9)
        high_vol_data = data[volatility > high_vol_threshold]
        
        if len(high_vol_data) > 0:
            metrics = eval_func(model, high_vol_data)
            self.test_results['high_volatility'] = metrics
            logger.info(f"  高波动期: {metrics}")
        
        # 3. 大幅下跌期测试
        logger.info("测试大幅下跌期...")
        large_drops = returns < returns.quantile(0.05)
        drop_data = data[large_drops]
        
        if len(drop_data) > 0:
            metrics = eval_func(model, drop_data)
            self.test_results['large_drops'] = metrics
            logger.info(f"  大幅下跌期: {metrics}")
        
        # 4. 连续下跌期测试
        logger.info("测试连续下跌期...")
        consecutive_drops = (returns < 0).rolling(window=5).sum() >= 4
        consec_drop_data = data[consecutive_drops]
        
        if len(consec_drop_data) > 0:
            metrics = eval_func(model, consec_drop_data)
            self.test_results['consecutive_drops'] = metrics
            logger.info(f"  连续下跌期: {metrics}")
        
        logger.info("压力测试完成")
        return self.test_results
    
    def print_summary(self):
        """打印压力测试摘要"""
        if not self.test_results:
            print("没有测试结果")
            return
        
        print("\n" + "="*80)
        print("压力测试摘要")
        print("="*80)
        
        # 获取所有指标名称
        metric_names = set()
        for metrics in self.test_results.values():
            metric_names.update(metrics.keys())
        
        print(f"\n{'测试场景':<30} | " + " | ".join([f"{m:>12}" for m in sorted(metric_names)]))
        print("-" * 80)
        
        for scenario, metrics in self.test_results.items():
            values = [f"{metrics.get(m, 0):>12.4f}" for m in sorted(metric_names)]
            print(f"{scenario:<30} | " + " | ".join(values))
        
        print("="*80 + "\n")


class BenchmarkComparison:
    """
    基准策略对比
    
    将AI策略与传统基准策略对比
    """
    
    def __init__(self):
        """初始化基准对比"""
        self.results = {}
        logger.info("基准策略对比初始化")
    
    def add_strategy(
        self,
        name: str,
        returns: pd.Series,
        trades: Optional[List[Dict]] = None
    ):
        """
        添加策略结果
        
        Args:
            name: 策略名称
            returns: 收益率序列
            trades: 交易记录
        """
        from ..backtest.metrics import PerformanceMetrics
        
        calculator = PerformanceMetrics(returns, trades)
        metrics = calculator.calculate_all()
        
        self.results[name] = {
            'returns': returns,
            'trades': trades,
            'metrics': metrics
        }
        
        logger.info(f"添加策略: {name}")
    
    def create_buy_and_hold(
        self,
        prices: pd.Series,
        initial_cash: float = 100000
    ) -> pd.Series:
        """
        创建买入持有策略
        
        Args:
            prices: 价格序列
            initial_cash: 初始资金
            
        Returns:
            Series: 收益率序列
        """
        # 第一天全仓买入
        shares = initial_cash / prices.iloc[0]
        portfolio_value = shares * prices
        returns = portfolio_value.pct_change().fillna(0)
        
        return returns
    
    def create_ma_crossover(
        self,
        prices: pd.Series,
        fast_window: int = 10,
        slow_window: int = 30
    ) -> pd.Series:
        """
        创建移动平均交叉策略
        
        Args:
            prices: 价格序列
            fast_window: 快速均线窗口
            slow_window: 慢速均线窗口
            
        Returns:
            Series: 收益率序列
        """
        fast_ma = prices.rolling(window=fast_window).mean()
        slow_ma = prices.rolling(window=slow_window).mean()
        
        # 交叉信号
        signal = (fast_ma > slow_ma).astype(int)
        position = signal.diff()
        
        # 计算收益
        returns = prices.pct_change()
        strategy_returns = returns * signal.shift(1)
        
        return strategy_returns.fillna(0)
    
    def create_momentum(
        self,
        prices: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        创建动量策略
        
        Args:
            prices: 价格序列
            lookback: 回看窗口
            
        Returns:
            Series: 收益率序列
        """
        momentum = prices.pct_change(lookback)
        signal = (momentum > 0).astype(int)
        
        returns = prices.pct_change()
        strategy_returns = returns * signal.shift(1)
        
        return strategy_returns.fillna(0)
    
    def print_comparison(self):
        """打印对比结果"""
        if not self.results:
            print("没有策略结果")
            return
        
        print("\n" + "="*100)
        print("策略对比")
        print("="*100)
        
        # 获取所有指标名称
        metric_names = set()
        for result in self.results.values():
            metric_names.update(result['metrics'].keys())
        
        # 选择关键指标
        key_metrics = ['total_return', 'cagr', 'sharpe_ratio', 'max_drawdown', 
                      'win_rate', 'profit_factor']
        display_metrics = [m for m in key_metrics if m in metric_names]
        
        print(f"\n{'策略':<20} | " + " | ".join([f"{m:>15}" for m in display_metrics]))
        print("-" * 100)
        
        for strategy_name, result in self.results.items():
            metrics = result['metrics']
            values = [f"{metrics.get(m, 0):>15.4f}" for m in display_metrics]
            print(f"{strategy_name:<20} | " + " | ".join(values))
        
        # 找出最佳策略
        print("\n最佳策略:")
        for metric in display_metrics:
            scores = {name: result['metrics'].get(metric, 0) 
                     for name, result in self.results.items()}
            
            # 对于max_drawdown，越小越好
            if 'drawdown' in metric.lower():
                best_strategy = min(scores, key=lambda k: abs(scores[k]))
            else:
                best_strategy = max(scores, key=scores.get)
            
            print(f"  {metric}: {best_strategy} ({scores[best_strategy]:.4f})")
        
        print("="*100 + "\n")
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """
        绘制策略对比图
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 累积收益曲线
            ax1 = axes[0, 0]
            for name, result in self.results.items():
                cumulative = (1 + result['returns']).cumprod()
                ax1.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
            ax1.set_ylabel('Cumulative Return')
            ax1.set_title('Cumulative Returns Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 夏普比率对比
            ax2 = axes[0, 1]
            sharpe_ratios = {name: result['metrics']['sharpe_ratio'] 
                           for name, result in self.results.items()}
            ax2.bar(range(len(sharpe_ratios)), list(sharpe_ratios.values()))
            ax2.set_xticks(range(len(sharpe_ratios)))
            ax2.set_xticklabels(list(sharpe_ratios.keys()), rotation=45, ha='right')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.set_title('Sharpe Ratio Comparison')
            ax2.grid(True, alpha=0.3)
            
            # 3. 最大回撤对比
            ax3 = axes[1, 0]
            max_drawdowns = {name: result['metrics']['max_drawdown'] 
                           for name, result in self.results.items()}
            ax3.bar(range(len(max_drawdowns)), list(max_drawdowns.values()))
            ax3.set_xticks(range(len(max_drawdowns)))
            ax3.set_xticklabels(list(max_drawdowns.keys()), rotation=45, ha='right')
            ax3.set_ylabel('Max Drawdown')
            ax3.set_title('Maximum Drawdown Comparison')
            ax3.grid(True, alpha=0.3)
            
            # 4. 年化收益率对比
            ax4 = axes[1, 1]
            cagrs = {name: result['metrics']['cagr'] 
                    for name, result in self.results.items()}
            ax4.bar(range(len(cagrs)), list(cagrs.values()))
            ax4.set_xticks(range(len(cagrs)))
            ax4.set_xticklabels(list(cagrs.keys()), rotation=45, ha='right')
            ax4.set_ylabel('CAGR')
            ax4.set_title('CAGR Comparison')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"对比图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("评估工具集测试完成")