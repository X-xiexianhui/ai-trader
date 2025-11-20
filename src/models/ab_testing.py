"""
A/B测试框架

实现模型A/B测试、流量分配、统计显著性检验
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import stats
import json

logger = logging.getLogger(__name__)


@dataclass
class ModelVariant:
    """模型变体"""
    name: str
    model: nn.Module
    traffic_weight: float = 0.5
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    sample_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def record_metric(self, metric_name: str, value: float):
        """记录指标"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        self.sample_count += 1
    
    def get_metric_stats(self, metric_name: str) -> Dict[str, float]:
        """获取指标统计"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }


class ABTest:
    """A/B测试管理器"""
    
    def __init__(
        self,
        test_name: str,
        variants: List[ModelVariant],
        primary_metric: str = "accuracy",
        min_samples: int = 100,
        confidence_level: float = 0.95
    ):
        """
        初始化A/B测试
        
        Args:
            test_name: 测试名称
            variants: 模型变体列表
            primary_metric: 主要评估指标
            min_samples: 最小样本数
            confidence_level: 置信水平
        """
        self.test_name = test_name
        self.variants = {v.name: v for v in variants}
        self.primary_metric = primary_metric
        self.min_samples = min_samples
        self.confidence_level = confidence_level
        self.started_at = datetime.now()
        self.is_active = True
        
        # 归一化流量权重
        total_weight = sum(v.traffic_weight for v in variants)
        for variant in variants:
            variant.traffic_weight /= total_weight
        
        logger.info(f"A/B test '{test_name}' initialized with {len(variants)} variants")
    
    def select_variant(self, user_id: Optional[str] = None) -> ModelVariant:
        """
        选择模型变体
        
        Args:
            user_id: 用户ID（用于一致性哈希）
            
        Returns:
            variant: 选中的模型变体
        """
        if user_id is not None:
            # 使用一致性哈希确保同一用户总是看到相同变体
            hash_value = hash(user_id) % 1000 / 1000
        else:
            # 随机选择
            hash_value = np.random.random()
        
        cumulative_weight = 0
        for variant in self.variants.values():
            cumulative_weight += variant.traffic_weight
            if hash_value <= cumulative_weight:
                return variant
        
        # 默认返回第一个变体
        return list(self.variants.values())[0]
    
    def predict(
        self,
        data: torch.Tensor,
        user_id: Optional[str] = None,
        variant_name: Optional[str] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        使用选定的变体进行预测
        
        Args:
            data: 输入数据
            user_id: 用户ID
            variant_name: 指定变体名称（可选）
            
        Returns:
            prediction: 预测结果
            variant_name: 使用的变体名称
        """
        if variant_name is not None:
            variant = self.variants[variant_name]
        else:
            variant = self.select_variant(user_id)
        
        variant.model.eval()
        with torch.no_grad():
            prediction = variant.model(data)
        
        return prediction, variant.name
    
    def record_result(
        self,
        variant_name: str,
        metrics: Dict[str, float]
    ):
        """
        记录测试结果
        
        Args:
            variant_name: 变体名称
            metrics: 指标字典
        """
        variant = self.variants[variant_name]
        for metric_name, value in metrics.items():
            variant.record_metric(metric_name, value)
    
    def get_variant_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有变体的统计信息"""
        stats = {}
        for name, variant in self.variants.items():
            stats[name] = {
                "sample_count": variant.sample_count,
                "traffic_weight": variant.traffic_weight,
                "metrics": {}
            }
            
            for metric_name in variant.metrics.keys():
                stats[name]["metrics"][metric_name] = variant.get_metric_stats(metric_name)
        
        return stats
    
    def compare_variants(
        self,
        variant_a: str,
        variant_b: str,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        比较两个变体
        
        Args:
            variant_a: 变体A名称
            variant_b: 变体B名称
            metric: 比较的指标（默认使用primary_metric）
            
        Returns:
            comparison: 比较结果
        """
        if metric is None:
            metric = self.primary_metric
        
        var_a = self.variants[variant_a]
        var_b = self.variants[variant_b]
        
        if metric not in var_a.metrics or metric not in var_b.metrics:
            return {"error": f"Metric '{metric}' not found in one or both variants"}
        
        values_a = var_a.metrics[metric]
        values_b = var_b.metrics[metric]
        
        # 检查样本数
        if len(values_a) < self.min_samples or len(values_b) < self.min_samples:
            return {
                "error": "Insufficient samples",
                "samples_a": len(values_a),
                "samples_b": len(values_b),
                "min_required": self.min_samples
            }
        
        # 计算统计量
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a, ddof=1)
        std_b = np.std(values_b, ddof=1)
        
        # t检验
        t_stat, p_value = stats.ttest_ind(values_a, values_b)
        
        # 效应量（Cohen's d）
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + 
                              (len(values_b) - 1) * std_b**2) / 
                             (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # 置信区间
        se = pooled_std * np.sqrt(1/len(values_a) + 1/len(values_b))
        ci_margin = stats.t.ppf((1 + self.confidence_level) / 2, 
                                len(values_a) + len(values_b) - 2) * se
        
        # 判断显著性
        is_significant = p_value < (1 - self.confidence_level)
        
        # 相对提升
        relative_improvement = ((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else 0
        
        comparison = {
            "metric": metric,
            "variant_a": {
                "name": variant_a,
                "mean": mean_a,
                "std": std_a,
                "samples": len(values_a)
            },
            "variant_b": {
                "name": variant_b,
                "mean": mean_b,
                "std": std_b,
                "samples": len(values_b)
            },
            "statistics": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "cohens_d": cohens_d,
                "confidence_level": self.confidence_level,
                "confidence_interval": {
                    "lower": mean_a - mean_b - ci_margin,
                    "upper": mean_a - mean_b + ci_margin
                }
            },
            "result": {
                "is_significant": is_significant,
                "relative_improvement_percent": relative_improvement,
                "winner": variant_a if mean_a > mean_b else variant_b,
                "recommendation": self._get_recommendation(
                    is_significant, relative_improvement, cohens_d
                )
            }
        }
        
        return comparison
    
    def _get_recommendation(
        self,
        is_significant: bool,
        relative_improvement: float,
        cohens_d: float
    ) -> str:
        """获取推荐建议"""
        if not is_significant:
            return "No significant difference detected. Continue testing or keep current variant."
        
        if abs(cohens_d) < 0.2:
            return "Statistically significant but small effect size. Consider practical significance."
        elif abs(cohens_d) < 0.5:
            return "Moderate effect size detected. Consider switching to better variant."
        else:
            return "Large effect size detected. Strong recommendation to switch to better variant."
    
    def get_winner(self, metric: Optional[str] = None) -> Optional[str]:
        """
        获取获胜变体
        
        Args:
            metric: 评估指标（默认使用primary_metric）
            
        Returns:
            winner: 获胜变体名称
        """
        if metric is None:
            metric = self.primary_metric
        
        # 找到样本数最多的两个变体进行比较
        sorted_variants = sorted(
            self.variants.items(),
            key=lambda x: x[1].sample_count,
            reverse=True
        )
        
        if len(sorted_variants) < 2:
            return None
        
        var_a_name = sorted_variants[0][0]
        var_b_name = sorted_variants[1][0]
        
        comparison = self.compare_variants(var_a_name, var_b_name, metric)
        
        if "error" in comparison:
            return None
        
        if comparison["result"]["is_significant"]:
            return comparison["result"]["winner"]
        
        return None
    
    def update_traffic_weights(self, weights: Dict[str, float]):
        """
        更新流量权重
        
        Args:
            weights: 变体名称到权重的映射
        """
        total_weight = sum(weights.values())
        for name, weight in weights.items():
            if name in self.variants:
                self.variants[name].traffic_weight = weight / total_weight
        
        logger.info(f"Traffic weights updated: {weights}")
    
    def stop_test(self):
        """停止测试"""
        self.is_active = False
        logger.info(f"A/B test '{self.test_name}' stopped")
    
    def export_results(self, filepath: str):
        """
        导出测试结果
        
        Args:
            filepath: 导出文件路径
        """
        results = {
            "test_name": self.test_name,
            "started_at": self.started_at.isoformat(),
            "primary_metric": self.primary_metric,
            "min_samples": self.min_samples,
            "confidence_level": self.confidence_level,
            "is_active": self.is_active,
            "variants": {}
        }
        
        for name, variant in self.variants.items():
            results["variants"][name] = {
                "traffic_weight": variant.traffic_weight,
                "sample_count": variant.sample_count,
                "created_at": variant.created_at.isoformat(),
                "metrics": {
                    metric_name: variant.get_metric_stats(metric_name)
                    for metric_name in variant.metrics.keys()
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")


class MultiArmedBandit:
    """多臂老虎机算法（用于动态流量分配）"""
    
    def __init__(
        self,
        variants: List[str],
        algorithm: str = "epsilon_greedy",
        epsilon: float = 0.1,
        temperature: float = 1.0
    ):
        """
        初始化多臂老虎机
        
        Args:
            variants: 变体名称列表
            algorithm: 算法类型 (epsilon_greedy/ucb/thompson_sampling)
            epsilon: ε-贪心算法的探索率
            temperature: Softmax温度参数
        """
        self.variants = variants
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.temperature = temperature
        
        # 初始化统计
        self.counts = {v: 0 for v in variants}
        self.values = {v: 0.0 for v in variants}
        self.rewards = {v: [] for v in variants}
        
        logger.info(f"Multi-armed bandit initialized with {algorithm} algorithm")
    
    def select_variant(self) -> str:
        """选择变体"""
        if self.algorithm == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.algorithm == "ucb":
            return self._ucb()
        elif self.algorithm == "thompson_sampling":
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _epsilon_greedy(self) -> str:
        """ε-贪心算法"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.choice(self.variants)
        else:
            # 利用：选择最佳
            return max(self.values.items(), key=lambda x: x[1])[0]
    
    def _ucb(self) -> str:
        """UCB算法（Upper Confidence Bound）"""
        total_counts = sum(self.counts.values())
        
        if total_counts == 0:
            return np.random.choice(self.variants)
        
        ucb_values = {}
        for variant in self.variants:
            if self.counts[variant] == 0:
                return variant  # 优先选择未尝试的
            
            avg_reward = self.values[variant]
            exploration_bonus = np.sqrt(2 * np.log(total_counts) / self.counts[variant])
            ucb_values[variant] = avg_reward + exploration_bonus
        
        return max(ucb_values.items(), key=lambda x: x[1])[0]
    
    def _thompson_sampling(self) -> str:
        """Thompson采样"""
        samples = {}
        for variant in self.variants:
            if len(self.rewards[variant]) == 0:
                samples[variant] = np.random.beta(1, 1)
            else:
                rewards = self.rewards[variant]
                successes = sum(rewards)
                failures = len(rewards) - successes
                samples[variant] = np.random.beta(successes + 1, failures + 1)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def update(self, variant: str, reward: float):
        """
        更新统计
        
        Args:
            variant: 变体名称
            reward: 奖励值
        """
        self.counts[variant] += 1
        n = self.counts[variant]
        value = self.values[variant]
        
        # 增量更新平均值
        self.values[variant] = ((n - 1) / n) * value + (1 / n) * reward
        self.rewards[variant].append(reward)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "counts": self.counts.copy(),
            "values": self.values.copy(),
            "total_trials": sum(self.counts.values())
        }


def example_usage():
    """使用示例"""
    # 创建示例模型
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc = nn.Linear(10, hidden_size)
            self.out = nn.Linear(hidden_size, 2)
        
        def forward(self, x):
            x = torch.relu(self.fc(x))
            return self.out(x)
    
    # 创建两个变体
    model_a = SimpleModel(64)
    model_b = SimpleModel(128)
    
    variant_a = ModelVariant(name="model_a", model=model_a, traffic_weight=0.5)
    variant_b = ModelVariant(name="model_b", model=model_b, traffic_weight=0.5)
    
    # 初始化A/B测试
    ab_test = ABTest(
        test_name="model_size_test",
        variants=[variant_a, variant_b],
        primary_metric="accuracy",
        min_samples=100
    )
    
    # 模拟测试
    print("Running A/B test simulation...")
    for i in range(200):
        # 选择变体
        variant = ab_test.select_variant(user_id=f"user_{i}")
        
        # 模拟预测和评估
        data = torch.randn(1, 10)
        prediction, variant_name = ab_test.predict(data)
        
        # 模拟准确率（model_b稍好）
        if variant_name == "model_a":
            accuracy = np.random.normal(0.75, 0.1)
        else:
            accuracy = np.random.normal(0.80, 0.1)
        
        ab_test.record_result(variant_name, {"accuracy": accuracy})
    
    # 获取统计
    stats = ab_test.get_variant_stats()
    print("\nVariant Statistics:")
    for name, stat in stats.items():
        print(f"\n{name}:")
        print(f"  Samples: {stat['sample_count']}")
        print(f"  Accuracy: {stat['metrics']['accuracy']['mean']:.4f} ± {stat['metrics']['accuracy']['std']:.4f}")
    
    # 比较变体
    comparison = ab_test.compare_variants("model_a", "model_b")
    print(f"\nComparison Results:")
    print(f"  P-value: {comparison['statistics']['p_value']:.4f}")
    print(f"  Significant: {comparison['result']['is_significant']}")
    print(f"  Winner: {comparison['result']['winner']}")
    print(f"  Improvement: {comparison['result']['relative_improvement_percent']:.2f}%")
    print(f"  Recommendation: {comparison['result']['recommendation']}")


if __name__ == "__main__":
    example_usage()