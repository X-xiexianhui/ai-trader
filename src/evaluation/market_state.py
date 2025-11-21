"""
市场状态泛化模块
识别不同市场状态并评估模型在各状态下的性能

任务6.4.1-6.4.2实现:
1. 市场状态识别器
2. 分状态性能评估
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class MarketStateIdentifier:
    """
    任务6.4.1: 市场状态识别器
    
    识别市场状态：牛市、熊市、震荡市、高波动期
    """
    
    def __init__(
        self,
        trend_window: int = 20,
        volatility_window: int = 20,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.015
    ):
        """
        初始化市场状态识别器
        
        Args:
            trend_window: 趋势计算窗口
            volatility_window: 波动率计算窗口
            trend_threshold: 趋势强度阈值
            volatility_threshold: 高波动阈值
        """
        self.trend_window = trend_window
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        
        logger.info(f"市场状态识别器初始化")
        logger.info(f"  趋势窗口: {trend_window}")
        logger.info(f"  波动率窗口: {volatility_window}")
        logger.info(f"  趋势阈值: {trend_threshold}")
        logger.info(f"  波动率阈值: {volatility_threshold}")
    
    def identify_states(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        识别市场状态
        
        Args:
            data: 价格数据DataFrame
            price_col: 价格列名
            
        Returns:
            市场状态Series
        """
        logger.info("开始识别市场状态...")
        
        if price_col not in data.columns:
            raise ValueError(f"列 {price_col} 不存在")
        
        prices = data[price_col].copy()
        
        # 1. 计算趋势强度
        trend_strength = self._calculate_trend_strength(prices)
        
        # 2. 计算波动率
        volatility = self._calculate_volatility(prices)
        
        # 3. 识别状态
        states = self._classify_states(trend_strength, volatility)
        
        # 统计各状态数量
        state_counts = states.value_counts()
        logger.info("市场状态分布:")
        for state, count in state_counts.items():
            pct = count / len(states) * 100
            logger.info(f"  {state}: {count} ({pct:.1f}%)")
        
        return states
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """
        计算趋势强度
        
        使用线性回归斜率
        """
        trend_strength = pd.Series(index=prices.index, dtype=float)
        
        for i in range(self.trend_window, len(prices)):
            window = prices.iloc[i-self.trend_window:i]
            x = np.arange(len(window))
            
            # 线性回归
            slope, _, _, _, _ = stats.linregress(x, window.values)
            
            # 归一化斜率（相对于价格）
            normalized_slope = slope / window.mean() if window.mean() != 0 else 0
            
            trend_strength.iloc[i] = normalized_slope
        
        return trend_strength
    
    def _calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        计算波动率
        
        使用收益率标准差
        """
        returns = prices.pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        return volatility
    
    def _classify_states(
        self,
        trend_strength: pd.Series,
        volatility: pd.Series
    ) -> pd.Series:
        """
        分类市场状态
        
        状态定义:
        - bull_market: 上涨趋势（趋势强度 > 阈值）
        - bear_market: 下跌趋势（趋势强度 < -阈值）
        - ranging: 震荡市（趋势强度在阈值内）
        - high_volatility: 高波动期（波动率 > 阈值）
        """
        states = pd.Series(index=trend_strength.index, dtype=str)
        
        for i in range(len(trend_strength)):
            trend = trend_strength.iloc[i]
            vol = volatility.iloc[i]
            
            if pd.isna(trend) or pd.isna(vol):
                states.iloc[i] = 'unknown'
                continue
            
            # 优先判断高波动
            if vol > self.volatility_threshold:
                states.iloc[i] = 'high_volatility'
            # 判断趋势
            elif trend > self.trend_threshold:
                states.iloc[i] = 'bull_market'
            elif trend < -self.trend_threshold:
                states.iloc[i] = 'bear_market'
            else:
                states.iloc[i] = 'ranging'
        
        return states
    
    def plot_states(
        self,
        data: pd.DataFrame,
        states: pd.Series,
        price_col: str = 'close',
        filename: str = 'market_states.png',
        output_dir: str = 'results/market_state'
    ) -> str:
        """
        可视化市场状态
        
        Args:
            data: 价格数据
            states: 市场状态
            price_col: 价格列名
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # 价格曲线
        prices = data[price_col]
        ax1.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.7)
        
        # 用不同颜色标记不同状态
        state_colors = {
            'bull_market': 'green',
            'bear_market': 'red',
            'ranging': 'blue',
            'high_volatility': 'orange',
            'unknown': 'gray'
        }
        
        for state, color in state_colors.items():
            mask = states == state
            if mask.any():
                ax1.scatter(prices.index[mask], prices.values[mask], 
                          c=color, s=10, alpha=0.5, label=state)
        
        ax1.set_ylabel('价格')
        ax1.set_title('市场状态识别')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 状态分布
        state_counts = states.value_counts()
        colors = [state_colors.get(state, 'gray') for state in state_counts.index]
        ax2.bar(range(len(state_counts)), state_counts.values, color=colors)
        ax2.set_xticks(range(len(state_counts)))
        ax2.set_xticklabels(state_counts.index, rotation=45)
        ax2.set_ylabel('数量')
        ax2.set_title('市场状态分布')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"市场状态可视化已保存: {filepath}")
        
        return str(filepath)


class StateSpecificEvaluator:
    """
    任务6.4.2: 分状态性能评估
    
    评估模型在不同市场状态下的性能
    """
    
    def __init__(self):
        """初始化分状态性能评估器"""
        self.state_results: Dict[str, Dict] = {}
        
        logger.info("分状态性能评估器初始化")
    
    def evaluate_by_state(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        states: pd.Series,
        trades: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """
        按市场状态评估性能
        
        Args:
            predictions: 预测值
            actuals: 实际值
            states: 市场状态
            trades: 交易记录（可选）
            
        Returns:
            各状态的性能指标
        """
        logger.info("开始分状态性能评估...")
        
        # 确保索引对齐
        common_index = predictions.index.intersection(actuals.index).intersection(states.index)
        predictions = predictions.loc[common_index]
        actuals = actuals.loc[common_index]
        states = states.loc[common_index]
        
        self.state_results = {}
        
        # 对每个状态进行评估
        unique_states = states.unique()
        for state in unique_states:
            if state == 'unknown':
                continue
            
            logger.info(f"\n评估状态: {state}")
            
            # 筛选该状态的数据
            mask = states == state
            state_predictions = predictions[mask]
            state_actuals = actuals[mask]
            
            if len(state_predictions) == 0:
                logger.warning(f"状态 {state} 没有数据")
                continue
            
            # 计算指标
            metrics = self._calculate_metrics(state_predictions, state_actuals)
            
            # 如果有交易记录，计算交易指标
            if trades is not None:
                trade_metrics = self._calculate_trade_metrics(trades, states, state)
                metrics.update(trade_metrics)
            
            self.state_results[state] = metrics
            
            logger.info(f"状态 {state} 指标: {metrics}")
        
        logger.info(f"\n分状态评估完成，共评估 {len(self.state_results)} 个状态")
        
        return self.state_results
    
    def _calculate_metrics(
        self,
        predictions: pd.Series,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """计算基础指标"""
        metrics = {}
        
        # 样本数
        metrics['n_samples'] = len(predictions)
        
        # 回归指标
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        metrics['mse'] = float(mse)
        metrics['mae'] = float(mae)
        metrics['rmse'] = float(np.sqrt(mse))
        
        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics['r2'] = float(r2)
        
        # 方向准确率
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        direction_accuracy = np.mean(pred_direction == actual_direction)
        metrics['direction_accuracy'] = float(direction_accuracy)
        
        # 相关系数
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            metrics['correlation'] = float(correlation)
        else:
            metrics['correlation'] = 0.0
        
        return metrics
    
    def _calculate_trade_metrics(
        self,
        trades: pd.DataFrame,
        states: pd.Series,
        target_state: str
    ) -> Dict[str, float]:
        """计算交易指标"""
        metrics = {}
        
        # 筛选该状态的交易
        if 'timestamp' in trades.columns or 'entry_time' in trades.columns:
            time_col = 'timestamp' if 'timestamp' in trades.columns else 'entry_time'
            
            # 找到每笔交易对应的状态
            state_trades = []
            for _, trade in trades.iterrows():
                trade_time = trade[time_col]
                if trade_time in states.index:
                    if states.loc[trade_time] == target_state:
                        state_trades.append(trade)
            
            if not state_trades:
                metrics['n_trades'] = 0
                return metrics
            
            state_trades_df = pd.DataFrame(state_trades)
            
            # 交易数量
            metrics['n_trades'] = len(state_trades_df)
            
            # 盈利指标
            if 'pnl' in state_trades_df.columns:
                pnls = state_trades_df['pnl']
                metrics['total_pnl'] = float(pnls.sum())
                metrics['avg_pnl'] = float(pnls.mean())
                metrics['win_rate'] = float((pnls > 0).mean())
                
                # 盈亏比
                winning_trades = pnls[pnls > 0]
                losing_trades = pnls[pnls < 0]
                if len(winning_trades) > 0 and len(losing_trades) > 0:
                    profit_factor = winning_trades.mean() / abs(losing_trades.mean())
                    metrics['profit_factor'] = float(profit_factor)
        
        return metrics
    
    def compare_states(self) -> pd.DataFrame:
        """
        比较各状态的性能
        
        Returns:
            比较结果DataFrame
        """
        if not self.state_results:
            logger.warning("没有评估结果")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.state_results).T
        
        logger.info("状态性能比较:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def identify_weaknesses(
        self,
        metric: str = 'direction_accuracy',
        threshold: float = 0.5
    ) -> List[str]:
        """
        识别弱点状态
        
        Args:
            metric: 用于判断的指标
            threshold: 阈值
            
        Returns:
            弱点状态列表
        """
        if not self.state_results:
            logger.warning("没有评估结果")
            return []
        
        weak_states = []
        
        for state, metrics in self.state_results.items():
            if metric in metrics:
                if metrics[metric] < threshold:
                    weak_states.append(state)
        
        if weak_states:
            logger.warning(f"识别出弱点状态: {weak_states}")
        else:
            logger.info(f"所有状态的 {metric} 都达标")
        
        return weak_states
    
    def assess_generalization(self) -> Dict[str, Any]:
        """
        评估泛化能力
        
        Returns:
            泛化能力评估结果
        """
        if not self.state_results:
            logger.warning("没有评估结果")
            return {}
        
        # 收集所有状态的关键指标
        metrics_by_state = {}
        for state, metrics in self.state_results.items():
            for metric_name, value in metrics.items():
                if metric_name not in metrics_by_state:
                    metrics_by_state[metric_name] = []
                metrics_by_state[metric_name].append(value)
        
        # 计算各指标的变异系数
        generalization = {}
        for metric_name, values in metrics_by_state.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
                
                generalization[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'cv': float(cv),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        # 判断泛化能力
        avg_cv = np.mean([g['cv'] for g in generalization.values() if g['cv'] != float('inf')])
        
        generalization['overall'] = {
            'avg_cv': float(avg_cv),
            'is_generalizable': avg_cv < 0.3  # CV < 0.3 认为泛化能力好
        }
        
        logger.info(f"泛化能力评估: {'良好' if generalization['overall']['is_generalizable'] else '较差'}")
        logger.info(f"平均CV: {avg_cv:.4f}")
        
        return generalization
    
    def plot_state_comparison(
        self,
        metrics: List[str] = None,
        filename: str = 'state_comparison.png',
        output_dir: str = 'results/market_state'
    ) -> str:
        """
        绘制状态性能比较图
        
        Args:
            metrics: 要比较的指标列表
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        if not self.state_results:
            logger.warning("没有评估结果")
            return ""
        
        comparison_df = self.compare_states()
        
        # 如果没有指定指标，使用所有数值型指标
        if metrics is None:
            metrics = [col for col in comparison_df.columns 
                      if comparison_df[col].dtype in [np.float64, np.int64]]
        
        # 过滤存在的指标
        metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not metrics:
            logger.warning("没有可绘制的指标")
            return ""
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            data = comparison_df[metric]
            
            # 绘制柱状图
            colors = plt.cm.Set3(range(len(data)))
            bars = ax.bar(range(len(data)), data.values, color=colors)
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, data.values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} 各状态对比')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"状态性能比较图已保存: {filepath}")
        
        return str(filepath)
    
    def generate_report(
        self,
        filename: str = 'state_performance_report.txt',
        output_dir: str = 'results/market_state'
    ) -> str:
        """
        生成分状态性能报告
        
        Args:
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        if not self.state_results:
            logger.warning("没有评估结果")
            return ""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("分状态性能评估报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 各状态详细指标
        report_lines.append("各状态详细指标:")
        report_lines.append("-" * 80)
        
        for state, metrics in self.state_results.items():
            report_lines.append(f"\n{state}:")
            for metric_name, value in metrics.items():
                report_lines.append(f"  {metric_name}: {value:.4f}")
        
        report_lines.append("")
        
        # 状态比较
        comparison_df = self.compare_states()
        report_lines.append("状态性能比较:")
        report_lines.append("-" * 80)
        report_lines.append(comparison_df.to_string())
        report_lines.append("")
        
        # 弱点识别
        weak_states = self.identify_weaknesses()
        report_lines.append("弱点状态:")
        report_lines.append("-" * 80)
        if weak_states:
            for state in weak_states:
                report_lines.append(f"  - {state}")
        else:
            report_lines.append("  未发现明显弱点")
        report_lines.append("")
        
        # 泛化能力
        generalization = self.assess_generalization()
        report_lines.append("泛化能力评估:")
        report_lines.append("-" * 80)
        if 'overall' in generalization:
            overall = generalization['overall']
            report_lines.append(f"  平均CV: {overall['avg_cv']:.4f}")
            report_lines.append(f"  泛化能力: {'良好' if overall['is_generalizable'] else '较差'}")
        report_lines.append("")
        
        # 优化建议
        report_lines.append("优化建议:")
        report_lines.append("-" * 80)
        if weak_states:
            report_lines.append(f"  1. 针对弱点状态({', '.join(weak_states)})进行专项优化")
            report_lines.append("  2. 增加这些状态的训练数据")
            report_lines.append("  3. 考虑使用状态特定的模型或策略")
        
        if generalization.get('overall', {}).get('is_generalizable', True):
            report_lines.append("  模型泛化能力良好，可以投入使用")
        else:
            report_lines.append("  模型泛化能力较差，建议:")
            report_lines.append("  - 增加训练数据的多样性")
            report_lines.append("  - 使用更强的正则化")
            report_lines.append("  - 考虑集成学习方法")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"分状态性能报告已保存: {filepath}")
        
        return str(filepath)