"""
过拟合检测模块
检测模型训练过程中的过拟合信号并提供修复建议

任务6.3.1-6.3.2实现:
1. 过拟合检测器
2. 多Seed稳定性测试
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class OverfittingDetector:
    """
    任务6.3.1: 过拟合检测器
    
    检测训练过程中的过拟合信号
    """
    
    def __init__(
        self,
        gap_threshold: float = 0.20,
        variance_threshold: float = 0.15,
        stagnation_patience: int = 5
    ):
        """
        初始化过拟合检测器
        
        Args:
            gap_threshold: 训练/验证性能差距阈值（默认20%）
            variance_threshold: 验证集方差增大阈值（默认15%）
            stagnation_patience: 验证集停滞容忍轮数
        """
        self.gap_threshold = gap_threshold
        self.variance_threshold = variance_threshold
        self.stagnation_patience = stagnation_patience
        
        self.signals: List[Dict] = []
        
        logger.info(f"过拟合检测器初始化")
        logger.info(f"  性能差距阈值: {gap_threshold*100}%")
        logger.info(f"  方差增大阈值: {variance_threshold*100}%")
        logger.info(f"  停滞容忍轮数: {stagnation_patience}")
    
    def detect(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        检测过拟合信号
        
        Args:
            train_history: 训练历史，格式: {metric: [values]}
            val_history: 验证历史，格式: {metric: [values]}
            
        Returns:
            检测结果字典
        """
        logger.info("开始过拟合检测...")
        
        self.signals = []
        detection_result = {
            'has_overfitting': False,
            'signals': [],
            'severity': 'none',  # none, mild, moderate, severe
            'recommendations': []
        }
        
        # 检查输入
        if not train_history or not val_history:
            logger.warning("训练或验证历史为空")
            return detection_result
        
        # 1. 检测训练/验证性能差距
        gap_signals = self._detect_performance_gap(train_history, val_history)
        self.signals.extend(gap_signals)
        
        # 2. 检测验证集提前达峰
        peak_signals = self._detect_early_peaking(val_history)
        self.signals.extend(peak_signals)
        
        # 3. 检测训练提升但验证停滞
        stagnation_signals = self._detect_validation_stagnation(
            train_history, val_history
        )
        self.signals.extend(stagnation_signals)
        
        # 4. 检测验证集方差增大
        variance_signals = self._detect_variance_increase(val_history)
        self.signals.extend(variance_signals)
        
        # 汇总结果
        if self.signals:
            detection_result['has_overfitting'] = True
            detection_result['signals'] = self.signals
            detection_result['severity'] = self._assess_severity()
            detection_result['recommendations'] = self._generate_recommendations()
            
            logger.warning(f"检测到 {len(self.signals)} 个过拟合信号")
            logger.warning(f"严重程度: {detection_result['severity']}")
        else:
            logger.info("未检测到明显的过拟合信号")
        
        return detection_result
    
    def _detect_performance_gap(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]]
    ) -> List[Dict]:
        """
        检测训练/验证性能差距
        
        信号1: 训练/验证性能差距 > 20%
        """
        signals = []
        
        for metric in train_history.keys():
            if metric not in val_history:
                continue
            
            train_values = train_history[metric]
            val_values = val_history[metric]
            
            if not train_values or not val_values:
                continue
            
            # 使用最后几个epoch的平均值
            n_recent = min(5, len(train_values))
            train_recent = np.mean(train_values[-n_recent:])
            val_recent = np.mean(val_values[-n_recent:])
            
            # 计算相对差距
            if train_recent != 0:
                gap = abs(train_recent - val_recent) / abs(train_recent)
                
                if gap > self.gap_threshold:
                    signals.append({
                        'type': 'performance_gap',
                        'metric': metric,
                        'train_value': float(train_recent),
                        'val_value': float(val_recent),
                        'gap': float(gap),
                        'threshold': self.gap_threshold,
                        'severity': 'severe' if gap > 0.3 else 'moderate'
                    })
                    
                    logger.warning(
                        f"检测到性能差距: {metric}, "
                        f"训练={train_recent:.4f}, 验证={val_recent:.4f}, "
                        f"差距={gap*100:.1f}%"
                    )
        
        return signals
    
    def _detect_early_peaking(
        self,
        val_history: Dict[str, List[float]]
    ) -> List[Dict]:
        """
        检测验证集提前达峰
        
        信号2: 验证集性能在训练早期达到最佳后持续下降
        """
        signals = []
        
        for metric, values in val_history.items():
            if len(values) < 10:
                continue
            
            # 找到最佳值的位置
            best_idx = np.argmax(values) if 'acc' in metric.lower() or 'sharpe' in metric.lower() else np.argmin(values)
            best_value = values[best_idx]
            
            # 如果最佳值出现在前30%的训练过程中
            if best_idx < len(values) * 0.3:
                # 检查后续是否持续下降
                subsequent_values = values[best_idx+1:]
                if len(subsequent_values) >= 5:
                    # 计算后续值与最佳值的平均差距
                    avg_subsequent = np.mean(subsequent_values)
                    degradation = abs(best_value - avg_subsequent) / abs(best_value) if best_value != 0 else 0
                    
                    if degradation > 0.05:  # 下降超过5%
                        signals.append({
                            'type': 'early_peaking',
                            'metric': metric,
                            'peak_epoch': int(best_idx),
                            'peak_value': float(best_value),
                            'final_value': float(values[-1]),
                            'degradation': float(degradation),
                            'severity': 'moderate'
                        })
                        
                        logger.warning(
                            f"检测到提前达峰: {metric}, "
                            f"峰值在epoch {best_idx}, "
                            f"后续下降{degradation*100:.1f}%"
                        )
        
        return signals
    
    def _detect_validation_stagnation(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]]
    ) -> List[Dict]:
        """
        检测训练提升但验证停滞
        
        信号3: 训练持续提升但验证集停滞不前
        """
        signals = []
        
        for metric in train_history.keys():
            if metric not in val_history:
                continue
            
            train_values = train_history[metric]
            val_values = val_history[metric]
            
            if len(train_values) < self.stagnation_patience * 2:
                continue
            
            # 检查最近N个epoch
            n = self.stagnation_patience
            train_recent = train_values[-n:]
            val_recent = val_values[-n:]
            
            # 训练是否持续提升
            train_improving = self._is_improving(train_recent, metric)
            
            # 验证是否停滞
            val_stagnant = self._is_stagnant(val_recent)
            
            if train_improving and val_stagnant:
                signals.append({
                    'type': 'validation_stagnation',
                    'metric': metric,
                    'train_trend': 'improving',
                    'val_trend': 'stagnant',
                    'stagnation_epochs': n,
                    'severity': 'moderate'
                })
                
                logger.warning(
                    f"检测到验证停滞: {metric}, "
                    f"训练持续提升但验证停滞{n}个epoch"
                )
        
        return signals
    
    def _detect_variance_increase(
        self,
        val_history: Dict[str, List[float]]
    ) -> List[Dict]:
        """
        检测验证集方差增大
        
        信号4: 验证集性能方差显著增大
        """
        signals = []
        
        for metric, values in val_history.items():
            if len(values) < 20:
                continue
            
            # 分为前半和后半
            mid = len(values) // 2
            first_half = values[:mid]
            second_half = values[mid:]
            
            # 计算方差
            var_first = np.var(first_half)
            var_second = np.var(second_half)
            
            # 检查方差是否显著增大
            if var_first > 0:
                variance_increase = (var_second - var_first) / var_first
                
                if variance_increase > self.variance_threshold:
                    signals.append({
                        'type': 'variance_increase',
                        'metric': metric,
                        'var_first_half': float(var_first),
                        'var_second_half': float(var_second),
                        'increase': float(variance_increase),
                        'threshold': self.variance_threshold,
                        'severity': 'mild'
                    })
                    
                    logger.warning(
                        f"检测到方差增大: {metric}, "
                        f"增大{variance_increase*100:.1f}%"
                    )
        
        return signals
    
    def _is_improving(self, values: List[float], metric: str) -> bool:
        """判断指标是否持续改善"""
        if len(values) < 3:
            return False
        
        # 对于准确率、夏普率等，越大越好
        if 'acc' in metric.lower() or 'sharpe' in metric.lower():
            return np.mean(np.diff(values)) > 0
        # 对于损失等，越小越好
        else:
            return np.mean(np.diff(values)) < 0
    
    def _is_stagnant(self, values: List[float]) -> bool:
        """判断指标是否停滞"""
        if len(values) < 3:
            return False
        
        # 计算变化率
        changes = np.abs(np.diff(values))
        mean_change = np.mean(changes)
        mean_value = np.mean(np.abs(values))
        
        # 如果变化率很小，认为停滞
        if mean_value > 0:
            relative_change = mean_change / mean_value
            return relative_change < 0.01  # 变化小于1%
        
        return False
    
    def _assess_severity(self) -> str:
        """评估过拟合严重程度"""
        if not self.signals:
            return 'none'
        
        severity_scores = {
            'severe': 3,
            'moderate': 2,
            'mild': 1
        }
        
        total_score = sum(severity_scores.get(s.get('severity', 'mild'), 1) 
                         for s in self.signals)
        avg_score = total_score / len(self.signals)
        
        if avg_score >= 2.5:
            return 'severe'
        elif avg_score >= 1.5:
            return 'moderate'
        else:
            return 'mild'
    
    def _generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        signal_types = set(s['type'] for s in self.signals)
        
        if 'performance_gap' in signal_types:
            recommendations.append("增加正则化强度（L2正则化、Dropout）")
            recommendations.append("减少模型复杂度（减少层数或神经元数量）")
            recommendations.append("增加训练数据量或使用数据增强")
        
        if 'early_peaking' in signal_types:
            recommendations.append("使用早停机制（Early Stopping）")
            recommendations.append("降低学习率")
            recommendations.append("增加Dropout比例")
        
        if 'validation_stagnation' in signal_types:
            recommendations.append("停止训练，模型已经过拟合")
            recommendations.append("使用更强的正则化")
            recommendations.append("考虑使用集成学习方法")
        
        if 'variance_increase' in signal_types:
            recommendations.append("增加批次大小以稳定训练")
            recommendations.append("使用学习率衰减")
            recommendations.append("增加训练数据的多样性")
        
        # 去重
        recommendations = list(dict.fromkeys(recommendations))
        
        return recommendations
    
    def plot_training_curves(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        filename: str = 'training_curves.png',
        output_dir: str = 'results/overfitting'
    ) -> str:
        """
        绘制训练曲线
        
        Args:
            train_history: 训练历史
            val_history: 验证历史
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        n_metrics = len(train_history)
        if n_metrics == 0:
            logger.warning("没有指标可绘制")
            return ""
        
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, train_values) in enumerate(train_history.items()):
            ax = axes[idx]
            
            epochs = range(1, len(train_values) + 1)
            ax.plot(epochs, train_values, 'b-', label='训练', linewidth=2)
            
            if metric in val_history:
                val_values = val_history[metric]
                ax.plot(range(1, len(val_values) + 1), val_values, 
                       'r-', label='验证', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} 训练曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存: {filepath}")
        
        return str(filepath)
    
    def save_report(
        self,
        detection_result: Dict,
        filename: str = 'overfitting_report.txt',
        output_dir: str = 'results/overfitting'
    ) -> str:
        """
        保存检测报告
        
        Args:
            detection_result: 检测结果
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("过拟合检测报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 总体结果
        report_lines.append(f"是否过拟合: {'是' if detection_result['has_overfitting'] else '否'}")
        report_lines.append(f"严重程度: {detection_result['severity']}")
        report_lines.append(f"检测到信号数: {len(detection_result['signals'])}")
        report_lines.append("")
        
        # 详细信号
        if detection_result['signals']:
            report_lines.append("检测到的信号:")
            report_lines.append("-" * 80)
            for i, signal in enumerate(detection_result['signals'], 1):
                report_lines.append(f"\n信号 {i}:")
                report_lines.append(f"  类型: {signal['type']}")
                report_lines.append(f"  严重程度: {signal.get('severity', 'unknown')}")
                for key, value in signal.items():
                    if key not in ['type', 'severity']:
                        report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # 修复建议
        if detection_result['recommendations']:
            report_lines.append("修复建议:")
            report_lines.append("-" * 80)
            for i, rec in enumerate(detection_result['recommendations'], 1):
                report_lines.append(f"  {i}. {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"过拟合检测报告已保存: {filepath}")
        
        return str(filepath)


class MultiSeedStabilityTester:
    """
    任务6.3.2: 多Seed稳定性测试
    
    使用多个随机种子测试模型稳定性
    """
    
    def __init__(
        self,
        n_seeds: int = 10,
        stability_threshold: float = 0.3
    ):
        """
        初始化多Seed稳定性测试器
        
        Args:
            n_seeds: 测试的随机种子数量
            stability_threshold: 稳定性阈值（CV < 0.3为稳定）
        """
        self.n_seeds = n_seeds
        self.stability_threshold = stability_threshold
        
        self.results: List[Dict] = []
        
        logger.info(f"多Seed稳定性测试器初始化")
        logger.info(f"  测试种子数: {n_seeds}")
        logger.info(f"  稳定性阈值: {stability_threshold}")
    
    def run_stability_test(
        self,
        train_func: Callable,
        eval_func: Callable,
        seeds: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行稳定性测试
        
        Args:
            train_func: 训练函数，接收seed和**kwargs，返回模型
            eval_func: 评估函数，接收模型和**kwargs，返回指标字典
            seeds: 随机种子列表（如果为None则自动生成）
            **kwargs: 传递给训练和评估函数的参数
            
        Returns:
            稳定性测试结果
        """
        logger.info(f"开始多Seed稳定性测试（{self.n_seeds}个种子）...")
        
        # 生成种子
        if seeds is None:
            seeds = list(range(42, 42 + self.n_seeds))
        else:
            seeds = seeds[:self.n_seeds]
        
        self.results = []
        
        # 对每个种子进行训练和评估
        for i, seed in enumerate(seeds, 1):
            logger.info(f"\n运行种子 {i}/{len(seeds)}: {seed}")
            
            try:
                # 训练模型
                model = train_func(seed=seed, **kwargs)
                
                # 评估模型
                metrics = eval_func(model, **kwargs)
                
                self.results.append({
                    'seed': seed,
                    'metrics': metrics
                })
                
                logger.info(f"种子 {seed} 完成: {metrics}")
                
            except Exception as e:
                logger.error(f"种子 {seed} 失败: {e}")
                continue
        
        if not self.results:
            logger.error("所有种子都失败了")
            return {}
        
        # 分析稳定性
        stability_analysis = self._analyze_stability()
        
        logger.info(f"\n稳定性测试完成，成功运行 {len(self.results)}/{len(seeds)} 个种子")
        
        return stability_analysis
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """分析稳定性"""
        if not self.results:
            return {}
        
        # 收集所有指标
        all_metrics = {}
        for result in self.results:
            for metric, value in result['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # 计算统计量
        statistics = {}
        for metric, values in all_metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
            
            statistics[metric] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'cv': float(cv),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'values': [float(v) for v in values]
            }
        
        # 判断稳定性
        is_stable = all(
            stats['cv'] < self.stability_threshold 
            for stats in statistics.values()
        )
        
        # 计算平均CV
        avg_cv = np.mean([stats['cv'] for stats in statistics.values()])
        
        analysis = {
            'n_seeds': len(self.results),
            'is_stable': is_stable,
            'avg_cv': float(avg_cv),
            'threshold': self.stability_threshold,
            'statistics': statistics,
            'individual_results': self.results
        }
        
        logger.info(f"稳定性: {'稳定' if is_stable else '不稳定'}")
        logger.info(f"平均CV: {avg_cv:.4f}")
        
        return analysis
    
    def plot_stability(
        self,
        stability_analysis: Dict,
        filename: str = 'stability_analysis.png',
        output_dir: str = 'results/stability'
    ) -> str:
        """
        绘制稳定性分析图
        
        Args:
            stability_analysis: 稳定性分析结果
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        statistics = stability_analysis.get('statistics', {})
        if not statistics:
            logger.warning("没有统计数据可绘制")
            return ""
        
        n_metrics = len(statistics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, (metric, stats) in enumerate(statistics.items()):
            ax = axes[idx]
            
            values = stats['values']
            
            # 绘制箱线图
            bp = ax.boxplot([values], labels=[metric], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            
            # 添加散点
            x = np.random.normal(1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.5, color='red', s=50)
            
            # 添加统计信息
            ax.text(0.02, 0.98, 
                   f"Mean: {stats['mean']:.4f}\n"
                   f"Std: {stats['std']:.4f}\n"
                   f"CV: {stats['cv']:.4f}",
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_ylabel('值')
            ax.set_title(f'{metric} 稳定性分析')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"稳定性分析图已保存: {filepath}")
        
        return str(filepath)
    
    def save_report(
        self,
        stability_analysis: Dict,
        filename: str = 'stability_report.txt',
        output_dir: str = 'results/stability'
    ) -> str:
        """
        保存稳定性测试报告
        
        Args:
            stability_analysis: 稳定性分析结果
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("多Seed稳定性测试报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 总体结果
        report_lines.append(f"测试种子数: {stability_analysis['n_seeds']}")
        report_lines.append(f"稳定性: {'稳定' if stability_analysis['is_stable'] else '不稳定'}")
        report_lines.append(f"平均CV: {stability_analysis['avg_cv']:.4f}")
        report_lines.append(f"稳定性阈值: {stability_analysis['threshold']}")
        report_lines.append("")
        
        # 各指标统计
        report_lines.append("各指标统计:")
        report_lines.append("-" * 80)
        
        statistics = stability_analysis.get('statistics', {})
        for metric, stats in statistics.items():
            report_lines.append(f"\n{metric}:")
            report_lines.append(f"  均值: {stats['mean']:.4f}")
            report_lines.append(f"  标准差: {stats['std']:.4f}")
            report_lines.append(f"  变异系数(CV): {stats['cv']:.4f}")
            report_lines.append(f"  最小值: {stats['min']:.4f}")
            report_lines.append(f"  最大值: {stats['max']:.4f}")
            report_lines.append(f"  中位数: {stats['median']:.4f}")
            report_lines.append(f"  稳定性: {'稳定' if stats['cv'] < self.stability_threshold else '不稳定'}")
        
        report_lines.append("")
        
        # 建议
        report_lines.append("建议:")
        report_lines.append("-" * 80)
        if stability_analysis['is_stable']:
            report_lines.append("  模型训练稳定，可以投入使用")
        else:
            report_lines.append("  模型训练不稳定，建议:")
            report_lines.append("  1. 检查数据质量和预处理流程")
            report_lines.append("  2. 调整模型超参数")
            report_lines.append("  3. 增加训练数据量")
            report_lines.append("  4. 使用更强的正则化")
            report_lines.append("  5. 考虑使用集成学习方法")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"稳定性测试报告已保存: {filepath}")
        
        return str(filepath)