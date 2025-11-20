"""
市场状态识别

识别不同的市场状态（牛市、熊市、震荡市、高波动）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    市场状态检测器
    
    基于趋势和波动率识别市场状态
    """
    
    def __init__(
        self,
        trend_window: int = 20,
        volatility_window: int = 20,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.03
    ):
        """
        初始化市场状态检测器
        
        Args:
            trend_window: 趋势计算窗口
            volatility_window: 波动率计算窗口
            trend_threshold: 趋势阈值
            volatility_threshold: 高波动阈值
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
        self.regimes = None
        self.regime_stats = {}
        
        logger.info(f"市场状态检测器初始化: "
                   f"趋势窗口={trend_window}, 波动窗口={volatility_window}")
    
    def detect_regimes(
        self,
        data: pd.DataFrame,
        price_column: str = 'close'
    ) -> pd.Series:
        """
        检测市场状态
        
        Args:
            data: 价格数据
            price_column: 价格列名
            
        Returns:
            Series: 市场状态标签
        """
        prices = data[price_column]
        
        # 计算收益率
        returns = prices.pct_change()
        
        # 计算趋势（移动平均斜率）
        trend = self._calculate_trend(prices)
        
        # 计算波动率
        volatility = returns.rolling(window=self.volatility_window).std()
        
        # 识别状态
        regimes = pd.Series(index=data.index, dtype=str)
        
        for i in range(len(data)):
            if pd.isna(trend.iloc[i]) or pd.isna(volatility.iloc[i]):
                regimes.iloc[i] = 'unknown'
                continue
            
            t = trend.iloc[i]
            v = volatility.iloc[i]
            
            # 高波动市场
            if v > self.volatility_threshold:
                regimes.iloc[i] = 'high_volatility'
            # 牛市：明显上升趋势
            elif t > self.trend_threshold:
                regimes.iloc[i] = 'bull'
            # 熊市：明显下降趋势
            elif t < -self.trend_threshold:
                regimes.iloc[i] = 'bear'
            # 震荡市：无明显趋势
            else:
                regimes.iloc[i] = 'sideways'
        
        self.regimes = regimes
        
        # 计算各状态统计
        self._calculate_regime_stats(data, regimes, returns)
        
        logger.info(f"市场状态检测完成")
        self._print_regime_distribution()
        
        return regimes
    
    def _calculate_trend(self, prices: pd.Series) -> pd.Series:
        """
        计算趋势（线性回归斜率）
        
        Args:
            prices: 价格序列
            
        Returns:
            Series: 趋势值
        """
        trend = pd.Series(index=prices.index, dtype=float)
        
        for i in range(self.trend_window, len(prices)):
            window_prices = prices.iloc[i-self.trend_window:i].values
            x = np.arange(len(window_prices))
            
            # 线性回归
            slope, _, _, _, _ = stats.linregress(x, window_prices)
            
            # 归一化斜率（相对于价格）
            normalized_slope = slope / window_prices[-1] if window_prices[-1] != 0 else 0
            trend.iloc[i] = normalized_slope
        
        return trend
    
    def _calculate_regime_stats(
        self,
        data: pd.DataFrame,
        regimes: pd.Series,
        returns: pd.Series
    ):
        """计算各市场状态的统计信息"""
        self.regime_stats = {}
        
        for regime in regimes.unique():
            if regime == 'unknown':
                continue
            
            mask = regimes == regime
            regime_returns = returns[mask]
            
            self.regime_stats[regime] = {
                'count': mask.sum(),
                'percentage': mask.sum() / len(regimes) * 100,
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'sharpe': regime_returns.mean() / regime_returns.std() if regime_returns.std() != 0 else 0,
                'max_return': regime_returns.max(),
                'min_return': regime_returns.min()
            }
    
    def _print_regime_distribution(self):
        """打印市场状态分布"""
        if not self.regime_stats:
            return
        
        print("\n市场状态分布:")
        print("-" * 60)
        for regime, stats in self.regime_stats.items():
            print(f"{regime:20s}: {stats['count']:6d} ({stats['percentage']:5.2f}%)")
    
    def get_regime_periods(self) -> Dict[str, List[Tuple]]:
        """
        获取各状态的时间段
        
        Returns:
            Dict: {状态: [(开始时间, 结束时间), ...]}
        """
        if self.regimes is None:
            return {}
        
        periods = {}
        current_regime = None
        start_idx = None
        
        for idx, regime in enumerate(self.regimes):
            if regime == 'unknown':
                continue
            
            if regime != current_regime:
                # 保存上一个状态的时间段
                if current_regime is not None and start_idx is not None:
                    if current_regime not in periods:
                        periods[current_regime] = []
                    periods[current_regime].append((
                        self.regimes.index[start_idx],
                        self.regimes.index[idx-1]
                    ))
                
                # 开始新状态
                current_regime = regime
                start_idx = idx
        
        # 保存最后一个状态
        if current_regime is not None and start_idx is not None:
            if current_regime not in periods:
                periods[current_regime] = []
            periods[current_regime].append((
                self.regimes.index[start_idx],
                self.regimes.index[-1]
            ))
        
        return periods
    
    def print_summary(self):
        """打印市场状态摘要"""
        if not self.regime_stats:
            print("未检测到市场状态")
            return
        
        print("\n" + "="*80)
        print("市场状态分析摘要")
        print("="*80)
        
        print(f"\n检测参数:")
        print(f"  趋势窗口: {self.trend_window}")
        print(f"  波动窗口: {self.volatility_window}")
        print(f"  趋势阈值: {self.trend_threshold:.4f}")
        print(f"  波动阈值: {self.volatility_threshold:.4f}")
        
        print(f"\n市场状态统计:")
        print("-" * 80)
        print(f"{'状态':<20} | {'数量':>8} | {'占比':>8} | {'平均收益':>12} | {'波动率':>10} | {'夏普':>8}")
        print("-" * 80)
        
        for regime, stats in sorted(self.regime_stats.items(), 
                                    key=lambda x: x[1]['count'], 
                                    reverse=True):
            print(f"{regime:<20} | "
                  f"{stats['count']:>8} | "
                  f"{stats['percentage']:>7.2f}% | "
                  f"{stats['avg_return']:>12.6f} | "
                  f"{stats['volatility']:>10.6f} | "
                  f"{stats['sharpe']:>8.4f}")
        
        print("="*80 + "\n")
    
    def plot_regimes(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        save_path: Optional[str] = None
    ):
        """
        绘制市场状态图
        
        Args:
            data: 价格数据
            price_column: 价格列名
            save_path: 保存路径
        """
        if self.regimes is None:
            logger.warning("请先运行detect_regimes()")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            
            # 价格图
            ax1.plot(data.index, data[price_column], label='Price', color='black', linewidth=1)
            
            # 为不同状态着色
            regime_colors = {
                'bull': 'green',
                'bear': 'red',
                'sideways': 'gray',
                'high_volatility': 'orange'
            }
            
            for regime, color in regime_colors.items():
                mask = self.regimes == regime
                ax1.fill_between(data.index, 
                                data[price_column].min(), 
                                data[price_column].max(),
                                where=mask, 
                                alpha=0.2, 
                                color=color,
                                label=regime)
            
            ax1.set_ylabel('Price')
            ax1.set_title('Market Regimes')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 状态时间线
            regime_numeric = self.regimes.map({
                'bull': 3,
                'sideways': 2,
                'bear': 1,
                'high_volatility': 4,
                'unknown': 0
            })
            
            ax2.plot(data.index, regime_numeric, drawstyle='steps-post', linewidth=2)
            ax2.set_ylabel('Regime')
            ax2.set_yticks([0, 1, 2, 3, 4])
            ax2.set_yticklabels(['Unknown', 'Bear', 'Sideways', 'Bull', 'High Vol'])
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"市场状态图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")


class RegimeBasedEvaluator:
    """
    基于市场状态的评估器
    
    在不同市场状态下分别评估模型性能
    """
    
    def __init__(self, regime_detector: MarketRegimeDetector):
        """
        初始化评估器
        
        Args:
            regime_detector: 市场状态检测器
        """
        self.regime_detector = regime_detector
        self.results = {}
        
        logger.info("基于市场状态的评估器初始化")
    
    def evaluate_by_regime(
        self,
        data: pd.DataFrame,
        predictions: pd.Series,
        actuals: pd.Series,
        eval_func: callable
    ) -> Dict:
        """
        按市场状态评估
        
        Args:
            data: 完整数据
            predictions: 预测值
            actuals: 实际值
            eval_func: 评估函数，接收(predictions, actuals)，返回指标字典
            
        Returns:
            Dict: 各状态的评估结果
        """
        if self.regime_detector.regimes is None:
            logger.error("请先运行regime_detector.detect_regimes()")
            return {}
        
        regimes = self.regime_detector.regimes
        self.results = {}
        
        # 整体评估
        overall_metrics = eval_func(predictions, actuals)
        self.results['overall'] = overall_metrics
        
        # 按状态评估
        for regime in regimes.unique():
            if regime == 'unknown':
                continue
            
            mask = regimes == regime
            regime_predictions = predictions[mask]
            regime_actuals = actuals[mask]
            
            if len(regime_predictions) == 0:
                continue
            
            regime_metrics = eval_func(regime_predictions, regime_actuals)
            self.results[regime] = regime_metrics
        
        logger.info("按市场状态评估完成")
        return self.results
    
    def print_comparison(self):
        """打印各状态性能对比"""
        if not self.results:
            print("没有评估结果")
            return
        
        print("\n" + "="*80)
        print("市场状态性能对比")
        print("="*80)
        
        # 获取所有指标名称
        metric_names = set()
        for metrics in self.results.values():
            metric_names.update(metrics.keys())
        
        # 打印表格
        print(f"\n{'状态':<20} | " + " | ".join([f"{m:>12}" for m in sorted(metric_names)]))
        print("-" * 80)
        
        for regime, metrics in sorted(self.results.items()):
            values = [f"{metrics.get(m, 0):>12.4f}" for m in sorted(metric_names)]
            print(f"{regime:<20} | " + " | ".join(values))
        
        print("="*80 + "\n")
    
    def get_best_worst_regimes(self, metric_name: str) -> Tuple[str, str]:
        """
        获取指定指标下表现最好和最差的市场状态
        
        Args:
            metric_name: 指标名称
            
        Returns:
            Tuple: (最好状态, 最差状态)
        """
        if not self.results:
            return None, None
        
        regime_scores = {}
        for regime, metrics in self.results.items():
            if regime != 'overall' and metric_name in metrics:
                regime_scores[regime] = metrics[metric_name]
        
        if not regime_scores:
            return None, None
        
        best_regime = max(regime_scores, key=regime_scores.get)
        worst_regime = min(regime_scores, key=regime_scores.get)
        
        return best_regime, worst_regime


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # 模拟不同市场状态
    n = len(dates)
    prices = np.zeros(n)
    prices[0] = 100
    
    for i in range(1, n):
        if i < n // 4:  # 牛市
            drift = 0.001
            vol = 0.01
        elif i < n // 2:  # 震荡
            drift = 0.0
            vol = 0.015
        elif i < 3 * n // 4:  # 熊市
            drift = -0.0008
            vol = 0.012
        else:  # 高波动
            drift = 0.0
            vol = 0.03
        
        prices[i] = prices[i-1] * (1 + drift + np.random.randn() * vol)
    
    data = pd.DataFrame({
        'close': prices,
        'open': prices * (1 + np.random.randn(n) * 0.005),
        'high': prices * (1 + abs(np.random.randn(n)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(n)) * 0.01)
    }, index=dates)
    
    # 检测市场状态
    detector = MarketRegimeDetector(
        trend_window=20,
        volatility_window=20,
        trend_threshold=0.001,
        volatility_threshold=0.02
    )
    
    regimes = detector.detect_regimes(data)
    detector.print_summary()
    
    # 绘制市场状态
    # detector.plot_regimes(data)