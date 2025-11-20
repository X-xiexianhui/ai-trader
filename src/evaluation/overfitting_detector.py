"""
过拟合检测器

检测模型是否存在过拟合问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class OverfittingDetector:
    """
    过拟合检测器
    
    通过多种指标检测模型过拟合
    """
    
    def __init__(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        test_metrics: Optional[Dict[str, List[float]]] = None
    ):
        """
        初始化过拟合检测器
        
        Args:
            train_metrics: 训练集指标历史，{metric_name: [values]}
            val_metrics: 验证集指标历史
            test_metrics: 测试集指标（可选）
        """
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        
        self.detection_results = {}
        
        logger.info("过拟合检测器初始化")
    
    def detect_all(self) -> Dict:
        """
        执行所有过拟合检测
        
        Returns:
            Dict: 检测结果
        """
        self.detection_results = {
            'performance_gap': self._detect_performance_gap(),
            'validation_degradation': self._detect_validation_degradation(),
            'variance_increase': self._detect_variance_increase(),
            'learning_curve': self._analyze_learning_curve(),
            'overall_assessment': {}
        }
        
        # 综合评估
        self.detection_results['overall_assessment'] = self._overall_assessment()
        
        return self.detection_results
    
    def _detect_performance_gap(self) -> Dict:
        """
        检测训练集和验证集性能差距
        
        Returns:
            Dict: 性能差距分析结果
        """
        results = {}
        
        for metric_name in self.train_metrics.keys():
            if metric_name not in self.val_metrics:
                continue
            
            train_values = np.array(self.train_metrics[metric_name])
            val_values = np.array(self.val_metrics[metric_name])
            
            # 使用最后几个epoch的平均值
            window = min(10, len(train_values))
            train_final = train_values[-window:].mean()
            val_final = val_values[-window:].mean()
            
            # 计算差距
            gap = abs(train_final - val_final)
            gap_pct = (gap / abs(train_final) * 100) if train_final != 0 else 0
            
            # 判断是否过拟合（差距超过10%）
            is_overfitting = gap_pct > 10
            
            results[metric_name] = {
                'train_final': float(train_final),
                'val_final': float(val_final),
                'gap': float(gap),
                'gap_pct': float(gap_pct),
                'is_overfitting': is_overfitting
            }
        
        return results
    
    def _detect_validation_degradation(self) -> Dict:
        """
        检测验证集性能下降
        
        Returns:
            Dict: 验证集性能退化分析
        """
        results = {}
        
        for metric_name, values in self.val_metrics.items():
            values = np.array(values)
            
            if len(values) < 10:
                continue
            
            # 找到最佳性能点
            best_idx = np.argmax(values) if 'loss' not in metric_name.lower() else np.argmin(values)
            best_value = values[best_idx]
            final_value = values[-1]
            
            # 计算退化程度
            degradation = abs(final_value - best_value)
            degradation_pct = (degradation / abs(best_value) * 100) if best_value != 0 else 0
            
            # 判断是否显著退化（超过5%）
            is_degraded = degradation_pct > 5 and best_idx < len(values) - 5
            
            results[metric_name] = {
                'best_value': float(best_value),
                'best_epoch': int(best_idx),
                'final_value': float(final_value),
                'degradation': float(degradation),
                'degradation_pct': float(degradation_pct),
                'is_degraded': is_degraded
            }
        
        return results
    
    def _detect_variance_increase(self) -> Dict:
        """
        检测方差增大
        
        Returns:
            Dict: 方差分析结果
        """
        results = {}
        
        for metric_name in self.train_metrics.keys():
            if metric_name not in self.val_metrics:
                continue
            
            train_values = np.array(self.train_metrics[metric_name])
            val_values = np.array(self.val_metrics[metric_name])
            
            if len(train_values) < 20:
                continue
            
            # 分为前半部分和后半部分
            mid = len(train_values) // 2
            
            train_var_early = np.var(train_values[:mid])
            train_var_late = np.var(train_values[mid:])
            val_var_early = np.var(val_values[:mid])
            val_var_late = np.var(val_values[mid:])
            
            # 计算方差变化
            train_var_change = (train_var_late - train_var_early) / train_var_early if train_var_early != 0 else 0
            val_var_change = (val_var_late - val_var_early) / val_var_early if val_var_early != 0 else 0
            
            # 验证集方差增大超过50%可能表示过拟合
            is_variance_increased = val_var_change > 0.5
            
            results[metric_name] = {
                'train_var_early': float(train_var_early),
                'train_var_late': float(train_var_late),
                'train_var_change_pct': float(train_var_change * 100),
                'val_var_early': float(val_var_early),
                'val_var_late': float(val_var_late),
                'val_var_change_pct': float(val_var_change * 100),
                'is_variance_increased': is_variance_increased
            }
        
        return results
    
    def _analyze_learning_curve(self) -> Dict:
        """
        分析学习曲线
        
        Returns:
            Dict: 学习曲线分析结果
        """
        results = {}
        
        for metric_name in self.train_metrics.keys():
            if metric_name not in self.val_metrics:
                continue
            
            train_values = np.array(self.train_metrics[metric_name])
            val_values = np.array(self.val_metrics[metric_name])
            
            # 计算趋势（使用线性回归）
            x = np.arange(len(train_values))
            
            train_slope, train_intercept, train_r, _, _ = stats.linregress(x, train_values)
            val_slope, val_intercept, val_r, _, _ = stats.linregress(x, val_values)
            
            # 判断学习曲线形态
            # 理想情况：训练和验证都在改善且趋势相近
            # 过拟合：训练持续改善但验证停滞或退化
            is_healthy = (
                abs(train_slope - val_slope) / abs(train_slope) < 0.5 if train_slope != 0 else True
            )
            
            results[metric_name] = {
                'train_slope': float(train_slope),
                'train_r_squared': float(train_r ** 2),
                'val_slope': float(val_slope),
                'val_r_squared': float(val_r ** 2),
                'slope_difference': float(abs(train_slope - val_slope)),
                'is_healthy_curve': is_healthy
            }
        
        return results
    
    def _overall_assessment(self) -> Dict:
        """
        综合评估过拟合程度
        
        Returns:
            Dict: 综合评估结果
        """
        # 收集所有过拟合信号
        overfitting_signals = []
        
        # 1. 性能差距
        for metric, result in self.detection_results['performance_gap'].items():
            if result['is_overfitting']:
                overfitting_signals.append(f"性能差距过大: {metric} ({result['gap_pct']:.1f}%)")
        
        # 2. 验证集退化
        for metric, result in self.detection_results['validation_degradation'].items():
            if result['is_degraded']:
                overfitting_signals.append(f"验证集性能退化: {metric} ({result['degradation_pct']:.1f}%)")
        
        # 3. 方差增大
        for metric, result in self.detection_results['variance_increase'].items():
            if result['is_variance_increased']:
                overfitting_signals.append(f"验证集方差增大: {metric} ({result['val_var_change_pct']:.1f}%)")
        
        # 4. 学习曲线异常
        for metric, result in self.detection_results['learning_curve'].items():
            if not result['is_healthy_curve']:
                overfitting_signals.append(f"学习曲线异常: {metric}")
        
        # 综合判断
        num_signals = len(overfitting_signals)
        
        if num_signals == 0:
            severity = "无过拟合"
            recommendation = "模型泛化能力良好，可以继续使用"
        elif num_signals <= 2:
            severity = "轻微过拟合"
            recommendation = "建议增加正则化或使用早停"
        elif num_signals <= 4:
            severity = "中度过拟合"
            recommendation = "需要调整模型复杂度或增加训练数据"
        else:
            severity = "严重过拟合"
            recommendation = "模型需要重新设计，考虑简化模型或大幅增加正则化"
        
        return {
            'num_signals': num_signals,
            'signals': overfitting_signals,
            'severity': severity,
            'recommendation': recommendation,
            'is_overfitting': num_signals > 0
        }
    
    def print_report(self):
        """打印过拟合检测报告"""
        if not self.detection_results:
            self.detect_all()
        
        print("\n" + "="*80)
        print("过拟合检测报告")
        print("="*80)
        
        # 综合评估
        overall = self.detection_results['overall_assessment']
        print(f"\n综合评估:")
        print(f"  严重程度: {overall['severity']}")
        print(f"  检测到的信号数: {overall['num_signals']}")
        print(f"  建议: {overall['recommendation']}")
        
        if overall['signals']:
            print(f"\n检测到的过拟合信号:")
            for i, signal in enumerate(overall['signals'], 1):
                print(f"  {i}. {signal}")
        
        # 性能差距
        print(f"\n1. 训练-验证性能差距:")
        print("-" * 80)
        for metric, result in self.detection_results['performance_gap'].items():
            status = "⚠️ 过拟合" if result['is_overfitting'] else "✓ 正常"
            print(f"  {metric}:")
            print(f"    训练集: {result['train_final']:.4f}")
            print(f"    验证集: {result['val_final']:.4f}")
            print(f"    差距: {result['gap']:.4f} ({result['gap_pct']:.2f}%)")
            print(f"    状态: {status}")
        
        # 验证集退化
        print(f"\n2. 验证集性能退化:")
        print("-" * 80)
        for metric, result in self.detection_results['validation_degradation'].items():
            status = "⚠️ 退化" if result['is_degraded'] else "✓ 正常"
            print(f"  {metric}:")
            print(f"    最佳值: {result['best_value']:.4f} (epoch {result['best_epoch']})")
            print(f"    最终值: {result['final_value']:.4f}")
            print(f"    退化: {result['degradation']:.4f} ({result['degradation_pct']:.2f}%)")
            print(f"    状态: {status}")
        
        # 方差分析
        print(f"\n3. 方差变化分析:")
        print("-" * 80)
        for metric, result in self.detection_results['variance_increase'].items():
            status = "⚠️ 方差增大" if result['is_variance_increased'] else "✓ 正常"
            print(f"  {metric}:")
            print(f"    验证集方差变化: {result['val_var_change_pct']:+.2f}%")
            print(f"    状态: {status}")
        
        # 学习曲线
        print(f"\n4. 学习曲线分析:")
        print("-" * 80)
        for metric, result in self.detection_results['learning_curve'].items():
            status = "✓ 健康" if result['is_healthy_curve'] else "⚠️ 异常"
            print(f"  {metric}:")
            print(f"    训练集趋势: {result['train_slope']:.6f} (R²={result['train_r_squared']:.4f})")
            print(f"    验证集趋势: {result['val_slope']:.6f} (R²={result['val_r_squared']:.4f})")
            print(f"    状态: {status}")
        
        print("="*80 + "\n")
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """
        绘制学习曲线
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            n_metrics = len(self.train_metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
            
            if n_metrics == 1:
                axes = [axes]
            
            for idx, (metric_name, train_values) in enumerate(self.train_metrics.items()):
                ax = axes[idx]
                
                epochs = range(len(train_values))
                ax.plot(epochs, train_values, label='Train', marker='o', markersize=3)
                
                if metric_name in self.val_metrics:
                    val_values = self.val_metrics[metric_name]
                    ax.plot(epochs, val_values, label='Validation', marker='s', markersize=3)
                
                if self.test_metrics and metric_name in self.test_metrics:
                    test_values = self.test_metrics[metric_name]
                    ax.plot(epochs, test_values, label='Test', marker='^', markersize=3)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'Learning Curve: {metric_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"学习曲线图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟的训练历史（过拟合场景）
    epochs = 50
    
    # 训练集持续改善
    train_loss = 1.0 - np.linspace(0, 0.9, epochs) + np.random.randn(epochs) * 0.02
    train_acc = np.linspace(0.5, 0.95, epochs) + np.random.randn(epochs) * 0.01
    
    # 验证集先改善后退化（过拟合）
    val_loss = 1.0 - np.linspace(0, 0.6, epochs) + np.random.randn(epochs) * 0.05
    val_loss[30:] += np.linspace(0, 0.3, 20)  # 后期退化
    val_acc = np.linspace(0.5, 0.85, epochs) + np.random.randn(epochs) * 0.02
    val_acc[30:] -= np.linspace(0, 0.15, 20)  # 后期退化
    
    train_metrics = {
        'loss': train_loss.tolist(),
        'accuracy': train_acc.tolist()
    }
    
    val_metrics = {
        'loss': val_loss.tolist(),
        'accuracy': val_acc.tolist()
    }
    
    # 创建检测器
    detector = OverfittingDetector(train_metrics, val_metrics)
    
    # 执行检测
    results = detector.detect_all()
    
    # 打印报告
    detector.print_report()
    
    # 绘制学习曲线
    # detector.plot_learning_curves()