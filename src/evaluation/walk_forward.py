"""
Walk-forward验证框架
实现时间序列交叉验证，避免数据泄露

任务6.1.1-6.1.3实现:
1. Walk-forward验证框架
2. 时间窗口滚动生成器
3. 多折验证结果汇总
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TimeWindowGenerator:
    """
    任务6.1.2: 时间窗口滚动生成器
    
    生成Walk-forward验证的时间窗口
    """
    
    def __init__(
        self,
        train_months: int = 24,
        val_months: int = 6,
        test_months: int = 6,
        step_months: int = 3
    ):
        """
        初始化时间窗口生成器
        
        Args:
            train_months: 训练窗口月数
            val_months: 验证窗口月数
            test_months: 测试窗口月数
            step_months: 滚动步长月数
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        self.step_months = step_months
        
        logger.info(f"时间窗口生成器初始化: train={train_months}月, val={val_months}月, "
                   f"test={test_months}月, step={step_months}月")
    
    def generate_folds(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Tuple[datetime, datetime]]]:
        """
        生成所有fold的时间窗口
        
        Args:
            start_date: 数据开始日期
            end_date: 数据结束日期
            
        Returns:
            fold列表，每个fold包含train/val/test的时间范围
        """
        folds = []
        current_start = start_date
        
        # 计算总窗口长度
        total_months = self.train_months + self.val_months + self.test_months
        
        while True:
            # 计算当前fold的各个窗口
            train_start = current_start
            train_end = self._add_months(train_start, self.train_months)
            
            val_start = train_end
            val_end = self._add_months(val_start, self.val_months)
            
            test_start = val_end
            test_end = self._add_months(test_start, self.test_months)
            
            # 检查是否超出数据范围
            if test_end > end_date:
                break
            
            fold = {
                'train': (train_start, train_end),
                'val': (val_start, val_end),
                'test': (test_start, test_end)
            }
            folds.append(fold)
            
            # 滚动到下一个窗口
            current_start = self._add_months(current_start, self.step_months)
        
        logger.info(f"生成了 {len(folds)} 个fold")
        
        return folds
    
    def _add_months(self, date: datetime, months: int) -> datetime:
        """
        给日期添加月数
        
        Args:
            date: 原始日期
            months: 要添加的月数
            
        Returns:
            新日期
        """
        # 简单实现：假设每月30天
        return date + timedelta(days=months * 30)
    
    def validate_folds(self, folds: List[Dict]) -> bool:
        """
        验证fold的有效性
        
        Args:
            folds: fold列表
            
        Returns:
            是否有效
        """
        if not folds:
            logger.error("没有生成任何fold")
            return False
        
        for i, fold in enumerate(folds):
            # 检查时间连续性
            train_start, train_end = fold['train']
            val_start, val_end = fold['val']
            test_start, test_end = fold['test']
            
            if train_end != val_start:
                logger.error(f"Fold {i}: 训练和验证窗口不连续")
                return False
            
            if val_end != test_start:
                logger.error(f"Fold {i}: 验证和测试窗口不连续")
                return False
            
            # 检查时间顺序
            if not (train_start < train_end < val_start < val_end < test_start < test_end):
                logger.error(f"Fold {i}: 时间顺序错误")
                return False
        
        # 检查fold之间无重叠
        for i in range(len(folds) - 1):
            current_test_end = folds[i]['test'][1]
            next_train_start = folds[i + 1]['train'][0]
            
            if current_test_end > next_train_start:
                logger.warning(f"Fold {i} 和 Fold {i+1} 存在重叠")
        
        logger.info("Fold验证通过")
        return True


class WalkForwardValidator:
    """
    任务6.1.1: Walk-forward验证框架
    
    实现时间序列的前向验证，避免数据泄露
    """
    
    def __init__(
        self,
        train_months: int = 24,
        val_months: int = 6,
        test_months: int = 6,
        step_months: int = 3,
        output_dir: str = 'results/walk_forward'
    ):
        """
        初始化Walk-forward验证器
        
        Args:
            train_months: 训练窗口月数
            val_months: 验证窗口月数
            test_months: 测试窗口月数
            step_months: 滚动步长月数
            output_dir: 输出目录
        """
        self.window_generator = TimeWindowGenerator(
            train_months, val_months, test_months, step_months
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict] = []
        
        logger.info(f"Walk-forward验证器初始化，输出目录: {self.output_dir}")
    
    def run_validation(
        self,
        data: pd.DataFrame,
        train_func: Callable,
        eval_func: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        运行Walk-forward验证
        
        Args:
            data: 完整数据集（带时间索引）
            train_func: 训练函数，接收(train_data, val_data, **kwargs)
            eval_func: 评估函数，接收(model, test_data, **kwargs)
            **kwargs: 传递给训练和评估函数的额外参数
            
        Returns:
            每个fold的结果列表
        """
        logger.info("开始Walk-forward验证...")
        
        # 生成时间窗口
        start_date = data.index.min()
        end_date = data.index.max()
        folds = self.window_generator.generate_folds(start_date, end_date)
        
        if not self.window_generator.validate_folds(folds):
            raise ValueError("Fold验证失败")
        
        self.results = []
        
        for i, fold in enumerate(folds):
            logger.info(f"\n{'='*60}")
            logger.info(f"处理 Fold {i+1}/{len(folds)}")
            logger.info(f"{'='*60}")
            
            # 划分数据
            train_start, train_end = fold['train']
            val_start, val_end = fold['val']
            test_start, test_end = fold['test']
            
            train_data = data.loc[train_start:train_end]
            val_data = data.loc[val_start:val_end]
            test_data = data.loc[test_start:test_end]
            
            logger.info(f"训练集: {train_start} 到 {train_end} ({len(train_data)} 条)")
            logger.info(f"验证集: {val_start} 到 {val_end} ({len(val_data)} 条)")
            logger.info(f"测试集: {test_start} 到 {test_end} ({len(test_data)} 条)")
            
            # 训练模型
            try:
                model = train_func(train_data, val_data, **kwargs)
            except Exception as e:
                logger.error(f"Fold {i+1} 训练失败: {e}")
                continue
            
            # 评估模型
            try:
                metrics = eval_func(model, test_data, **kwargs)
            except Exception as e:
                logger.error(f"Fold {i+1} 评估失败: {e}")
                continue
            
            # 记录结果
            result = {
                'fold': i + 1,
                'train_period': (train_start, train_end),
                'val_period': (val_start, val_end),
                'test_period': (test_start, test_end),
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'metrics': metrics
            }
            self.results.append(result)
            
            logger.info(f"Fold {i+1} 完成")
            logger.info(f"测试集指标: {metrics}")
        
        logger.info(f"\nWalk-forward验证完成，共 {len(self.results)} 个fold")
        
        return self.results
    
    def get_results(self) -> List[Dict]:
        """获取验证结果"""
        return self.results
    
    def save_results(self, filename: str = 'walk_forward_results.json') -> str:
        """
        保存验证结果
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if not self.results:
            logger.warning("没有结果可保存")
            return ""
        
        # 转换datetime为字符串
        results_serializable = []
        for result in self.results:
            result_copy = result.copy()
            result_copy['train_period'] = [
                result['train_period'][0].isoformat(),
                result['train_period'][1].isoformat()
            ]
            result_copy['val_period'] = [
                result['val_period'][0].isoformat(),
                result['val_period'][1].isoformat()
            ]
            result_copy['test_period'] = [
                result['test_period'][0].isoformat(),
                result['test_period'][1].isoformat()
            ]
            results_serializable.append(result_copy)
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存: {filepath}")
        
        return str(filepath)


class ValidationResultAggregator:
    """
    任务6.1.3: 多折验证结果汇总
    
    统计和分析多折验证的结果
    """
    
    def __init__(self, results: List[Dict]):
        """
        初始化结果汇总器
        
        Args:
            results: Walk-forward验证结果列表
        """
        self.results = results
        self.summary: Dict[str, Any] = {}
        
        logger.info(f"结果汇总器初始化，共 {len(results)} 个fold")
    
    def aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        汇总所有fold的指标
        
        Returns:
            指标统计字典（均值、标准差、CV等）
        """
        if not self.results:
            logger.warning("没有结果可汇总")
            return {}
        
        # 提取所有指标
        all_metrics = {}
        for result in self.results:
            metrics = result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # 计算统计量
        summary = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            summary[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'count': len(values_array)
            }
        
        self.summary = summary
        
        logger.info(f"指标汇总完成，共 {len(summary)} 个指标")
        
        return summary
    
    def analyze_stability(self, cv_threshold: float = 0.3) -> Dict[str, Any]:
        """
        分析模型稳定性
        
        Args:
            cv_threshold: CV阈值，低于此值认为稳定
            
        Returns:
            稳定性分析结果
        """
        if not self.summary:
            self.aggregate_metrics()
        
        stability_analysis = {
            'stable_metrics': [],
            'unstable_metrics': [],
            'overall_stability': 'unknown'
        }
        
        for metric_name, stats in self.summary.items():
            cv = stats['cv']
            if cv < cv_threshold:
                stability_analysis['stable_metrics'].append({
                    'metric': metric_name,
                    'cv': cv,
                    'mean': stats['mean']
                })
            else:
                stability_analysis['unstable_metrics'].append({
                    'metric': metric_name,
                    'cv': cv,
                    'mean': stats['mean']
                })
        
        # 判断整体稳定性
        stable_count = len(stability_analysis['stable_metrics'])
        total_count = len(self.summary)
        
        if stable_count / total_count > 0.7:
            stability_analysis['overall_stability'] = 'stable'
        elif stable_count / total_count > 0.4:
            stability_analysis['overall_stability'] = 'moderate'
        else:
            stability_analysis['overall_stability'] = 'unstable'
        
        logger.info(f"稳定性分析: {stable_count}/{total_count} 个指标稳定 "
                   f"(CV<{cv_threshold})")
        logger.info(f"整体稳定性: {stability_analysis['overall_stability']}")
        
        return stability_analysis
    
    def evaluate_generalization(self) -> Dict[str, Any]:
        """
        评估泛化能力
        
        Returns:
            泛化能力评估结果
        """
        if not self.results:
            return {}
        
        # 比较训练集和测试集性能
        generalization = {
            'train_test_gap': {},
            'performance_trend': {},
            'generalization_score': 0.0
        }
        
        # 这里需要训练集和测试集的指标对比
        # 简化实现：检查测试集性能的一致性
        test_metrics = []
        for result in self.results:
            test_metrics.append(result.get('metrics', {}))
        
        # 计算性能趋势
        if len(test_metrics) > 1:
            # 检查关键指标的趋势
            key_metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']
            
            for metric in key_metrics:
                values = [m.get(metric, 0) for m in test_metrics if metric in m]
                if len(values) > 1:
                    # 简单线性趋势
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    generalization['performance_trend'][metric] = float(slope)
        
        logger.info("泛化能力评估完成")
        
        return generalization
    
    def plot_metrics_distribution(
        self,
        metrics: List[str] = None,
        filename: str = 'metrics_distribution.png',
        output_dir: str = 'results/walk_forward'
    ) -> str:
        """
        绘制指标分布图
        
        Args:
            metrics: 要绘制的指标列表
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        if not self.results:
            logger.warning("没有结果可绘制")
            return ""
        
        # 提取指标数据
        metrics_data = {}
        for result in self.results:
            for key, value in result.get('metrics', {}).items():
                if isinstance(value, (int, float)):
                    if key not in metrics_data:
                        metrics_data[key] = []
                    metrics_data[key].append(value)
        
        if metrics is None:
            metrics = list(metrics_data.keys())[:6]  # 最多显示6个指标
        
        # 创建子图
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, metric in enumerate(metrics):
            if metric not in metrics_data:
                continue
            
            values = metrics_data[metric]
            ax = axes[i]
            
            # 绘制箱线图和散点图
            ax.boxplot([values], labels=[metric])
            ax.scatter([1]*len(values), values, alpha=0.5, color='red')
            
            # 添加统计信息
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axhline(y=mean_val, color='green', linestyle='--', 
                      label=f'Mean: {mean_val:.4f}')
            ax.set_title(f'{metric}\n(μ={mean_val:.4f}, σ={std_val:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"指标分布图已保存: {filepath}")
        
        return str(filepath)
    
    def generate_report(
        self,
        filename: str = 'validation_summary.txt',
        output_dir: str = 'results/walk_forward'
    ) -> str:
        """
        生成验证汇总报告
        
        Args:
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        if not self.summary:
            self.aggregate_metrics()
        
        stability = self.analyze_stability()
        generalization = self.evaluate_generalization()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Walk-forward验证汇总报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基本信息
        report_lines.append(f"总fold数: {len(self.results)}")
        report_lines.append("")
        
        # 指标统计
        report_lines.append("指标统计:")
        report_lines.append("-" * 80)
        for metric_name, stats in self.summary.items():
            report_lines.append(f"\n{metric_name}:")
            report_lines.append(f"  均值: {stats['mean']:.4f}")
            report_lines.append(f"  标准差: {stats['std']:.4f}")
            report_lines.append(f"  变异系数(CV): {stats['cv']:.4f}")
            report_lines.append(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        report_lines.append("")
        
        # 稳定性分析
        report_lines.append("稳定性分析:")
        report_lines.append("-" * 80)
        report_lines.append(f"整体稳定性: {stability['overall_stability']}")
        report_lines.append(f"稳定指标数: {len(stability['stable_metrics'])}")
        report_lines.append(f"不稳定指标数: {len(stability['unstable_metrics'])}")
        report_lines.append("")
        
        # 泛化能力
        report_lines.append("泛化能力评估:")
        report_lines.append("-" * 80)
        if generalization.get('performance_trend'):
            for metric, trend in generalization['performance_trend'].items():
                direction = "上升" if trend > 0 else "下降"
                report_lines.append(f"{metric} 趋势: {direction} ({trend:.6f})")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"验证汇总报告已保存: {filepath}")
        
        return str(filepath)