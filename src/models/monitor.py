"""
模型监控模块

实时监控模型性能、预测分布和模型退化
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMonitor:
    """模型监控器"""
    
    def __init__(
        self,
        model_name: str,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
        log_dir: str = "logs/monitoring"
    ):
        """
        初始化模型监控器
        
        Args:
            model_name: 模型名称
            window_size: 滑动窗口大小
            alert_thresholds: 告警阈值
            log_dir: 日志目录
        """
        self.model_name = model_name
        self.window_size = window_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认告警阈值
        self.alert_thresholds = alert_thresholds or {
            "latency_p99_ms": 100,
            "error_rate": 0.05,
            "prediction_drift": 0.3,
            "feature_drift": 0.3
        }
        
        # 监控数据
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.feature_stats = {}
        
        # 基线统计
        self.baseline_stats = None
        
        # 告警历史
        self.alerts = []
        
        # 性能计数器
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
        
        logger.info(f"Initialized ModelMonitor for {model_name}")
    
    def record_prediction(
        self,
        prediction: np.ndarray,
        latency: float,
        features: Optional[np.ndarray] = None,
        error: bool = False
    ):
        """
        记录预测
        
        Args:
            prediction: 预测结果
            latency: 推理延迟（秒）
            features: 输入特征
            error: 是否发生错误
        """
        self.total_requests += 1
        
        # 记录延迟
        self.latencies.append(latency)
        
        # 记录预测
        if isinstance(prediction, np.ndarray):
            self.predictions.append(prediction.flatten())
        else:
            self.predictions.append([prediction])
        
        # 记录错误
        if error:
            self.total_errors += 1
            self.errors.append(1)
        else:
            self.errors.append(0)
        
        # 记录特征统计
        if features is not None:
            self._update_feature_stats(features)
        
        # 检查告警
        self._check_alerts()
    
    def _update_feature_stats(self, features: np.ndarray):
        """更新特征统计"""
        features = features.flatten()
        
        for i, value in enumerate(features):
            feature_name = f"feature_{i}"
            
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = deque(maxlen=self.window_size)
            
            self.feature_stats[feature_name].append(value)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        if not self.latencies:
            return {}
        
        latencies_ms = np.array(self.latencies) * 1000
        
        return {
            "mean_ms": float(np.mean(latencies_ms)),
            "median_ms": float(np.median(latencies_ms)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "min_ms": float(np.min(latencies_ms)),
            "max_ms": float(np.max(latencies_ms))
        }
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """获取预测统计"""
        if not self.predictions:
            return {}
        
        predictions = np.array([p[0] if len(p) > 0 else 0 for p in self.predictions])
        
        return {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions)),
            "q25": float(np.percentile(predictions, 25)),
            "q75": float(np.percentile(predictions, 75))
        }
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if not self.errors:
            return 0.0
        return float(np.mean(self.errors))
    
    def get_throughput(self) -> float:
        """获取吞吐量（QPS）"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.total_requests / elapsed
    
    def detect_prediction_drift(self) -> Optional[Dict[str, float]]:
        """
        检测预测分布漂移
        
        Returns:
            drift_info: 漂移信息，如果无基线则返回None
        """
        if self.baseline_stats is None:
            return None
        
        current_stats = self.get_prediction_stats()
        
        if not current_stats:
            return None
        
        # 计算KL散度或其他漂移指标
        # 这里使用简单的均值和标准差比较
        mean_drift = abs(
            current_stats["mean"] - self.baseline_stats["mean"]
        ) / (abs(self.baseline_stats["mean"]) + 1e-8)
        
        std_drift = abs(
            current_stats["std"] - self.baseline_stats["std"]
        ) / (self.baseline_stats["std"] + 1e-8)
        
        drift_score = (mean_drift + std_drift) / 2
        
        return {
            "drift_score": float(drift_score),
            "mean_drift": float(mean_drift),
            "std_drift": float(std_drift),
            "is_drifted": drift_score > self.alert_thresholds["prediction_drift"]
        }
    
    def detect_feature_drift(self) -> Dict[str, Any]:
        """
        检测特征分布漂移
        
        Returns:
            drift_info: 各特征的漂移信息
        """
        if self.baseline_stats is None or "features" not in self.baseline_stats:
            return {}
        
        drift_info = {}
        
        for feature_name, values in self.feature_stats.items():
            if len(values) < 10:
                continue
            
            baseline = self.baseline_stats["features"].get(feature_name)
            if baseline is None:
                continue
            
            current_mean = np.mean(values)
            current_std = np.std(values)
            
            mean_drift = abs(current_mean - baseline["mean"]) / (abs(baseline["mean"]) + 1e-8)
            std_drift = abs(current_std - baseline["std"]) / (baseline["std"] + 1e-8)
            
            drift_score = (mean_drift + std_drift) / 2
            
            drift_info[feature_name] = {
                "drift_score": float(drift_score),
                "mean_drift": float(mean_drift),
                "std_drift": float(std_drift),
                "is_drifted": drift_score > self.alert_thresholds["feature_drift"]
            }
        
        return drift_info
    
    def set_baseline(self):
        """设置当前统计为基线"""
        self.baseline_stats = {
            "timestamp": datetime.now().isoformat(),
            "predictions": self.get_prediction_stats(),
            "latency": self.get_latency_stats(),
            "features": {}
        }
        
        # 保存特征基线
        for feature_name, values in self.feature_stats.items():
            if len(values) > 0:
                self.baseline_stats["features"][feature_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values))
                }
        
        logger.info("Baseline statistics set")
    
    def _check_alerts(self):
        """检查告警条件"""
        alerts = []
        
        # 检查延迟
        latency_stats = self.get_latency_stats()
        if latency_stats and latency_stats["p99_ms"] > self.alert_thresholds["latency_p99_ms"]:
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "message": f"P99 latency {latency_stats['p99_ms']:.2f}ms exceeds threshold",
                "value": latency_stats["p99_ms"],
                "threshold": self.alert_thresholds["latency_p99_ms"]
            })
        
        # 检查错误率
        error_rate = self.get_error_rate()
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "severity": "critical",
                "message": f"Error rate {error_rate:.2%} exceeds threshold",
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate"]
            })
        
        # 检查预测漂移
        drift_info = self.detect_prediction_drift()
        if drift_info and drift_info["is_drifted"]:
            alerts.append({
                "type": "prediction_drift",
                "severity": "warning",
                "message": f"Prediction distribution drift detected",
                "value": drift_info["drift_score"],
                "threshold": self.alert_thresholds["prediction_drift"]
            })
        
        # 记录告警
        for alert in alerts:
            alert["timestamp"] = datetime.now().isoformat()
            alert["model_name"] = self.model_name
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[Dict]:
        """
        获取告警
        
        Args:
            severity: 筛选严重程度
            since: 筛选时间
            
        Returns:
            alerts: 告警列表
        """
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        if since:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a["timestamp"]) >= since
            ]
        
        return alerts
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        uptime = time.time() - self.start_time
        
        summary = {
            "model_name": self.model_name,
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.get_error_rate(),
            "throughput_qps": self.get_throughput(),
            "latency": self.get_latency_stats(),
            "predictions": self.get_prediction_stats(),
            "alerts": {
                "total": len(self.alerts),
                "critical": len([a for a in self.alerts if a["severity"] == "critical"]),
                "warning": len([a for a in self.alerts if a["severity"] == "warning"])
            }
        }
        
        # 添加漂移信息
        drift_info = self.detect_prediction_drift()
        if drift_info:
            summary["prediction_drift"] = drift_info
        
        return summary
    
    def export_metrics(self, output_path: Optional[str] = None) -> str:
        """
        导出监控指标
        
        Args:
            output_path: 输出路径
            
        Returns:
            output_path: 实际输出路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"{self.model_name}_metrics_{timestamp}.json"
        
        metrics = {
            "summary": self.get_summary(),
            "baseline": self.baseline_stats,
            "alerts": self.alerts,
            "export_time": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to: {output_path}")
        return str(output_path)
    
    def reset(self):
        """重置监控器"""
        self.latencies.clear()
        self.predictions.clear()
        self.errors.clear()
        self.feature_stats.clear()
        self.alerts.clear()
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = time.time()
        logger.info("Monitor reset")


class PerformanceDashboard:
    """性能仪表板"""
    
    def __init__(self, monitors: Dict[str, ModelMonitor]):
        """
        初始化仪表板
        
        Args:
            monitors: 模型监控器字典
        """
        self.monitors = monitors
    
    def get_overview(self) -> Dict[str, Any]:
        """获取总览"""
        overview = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }
        
        for model_name, monitor in self.monitors.items():
            overview["models"][model_name] = monitor.get_summary()
        
        return overview
    
    def get_comparison(self) -> pd.DataFrame:
        """获取模型对比"""
        data = []
        
        for model_name, monitor in self.monitors.items():
            summary = monitor.get_summary()
            
            row = {
                "model": model_name,
                "requests": summary["total_requests"],
                "error_rate": summary["error_rate"],
                "throughput_qps": summary["throughput_qps"],
                "p99_latency_ms": summary["latency"].get("p99_ms", 0),
                "alerts": summary["alerts"]["total"]
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_report(self, output_path: str):
        """生成监控报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "overview": self.get_overview(),
            "comparison": self.get_comparison().to_dict(orient="records")
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated: {output_path}")


def example_usage():
    """使用示例"""
    # 创建监控器
    monitor = ModelMonitor(
        model_name="ppo_v1",
        window_size=1000,
        alert_thresholds={
            "latency_p99_ms": 100,
            "error_rate": 0.05,
            "prediction_drift": 0.3
        }
    )
    
    # 模拟预测记录
    np.random.seed(42)
    
    for i in range(100):
        prediction = np.random.randn(1)
        latency = np.random.uniform(0.01, 0.05)
        features = np.random.randn(10)
        error = np.random.random() < 0.01
        
        monitor.record_prediction(
            prediction=prediction,
            latency=latency,
            features=features,
            error=error
        )
    
    # 设置基线
    monitor.set_baseline()
    
    # 继续记录（模拟漂移）
    for i in range(100):
        prediction = np.random.randn(1) + 0.5  # 漂移
        latency = np.random.uniform(0.01, 0.05)
        features = np.random.randn(10)
        
        monitor.record_prediction(
            prediction=prediction,
            latency=latency,
            features=features
        )
    
    # 获取摘要
    summary = monitor.get_summary()
    print("\nMonitoring Summary:")
    print(f"Total requests: {summary['total_requests']}")
    print(f"Error rate: {summary['error_rate']:.2%}")
    print(f"Throughput: {summary['throughput_qps']:.2f} QPS")
    print(f"P99 latency: {summary['latency']['p99_ms']:.2f}ms")
    
    # 检查漂移
    drift_info = monitor.detect_prediction_drift()
    if drift_info:
        print(f"\nPrediction Drift:")
        print(f"Drift score: {drift_info['drift_score']:.3f}")
        print(f"Is drifted: {drift_info['is_drifted']}")
    
    # 获取告警
    alerts = monitor.get_alerts()
    print(f"\nTotal alerts: {len(alerts)}")
    for alert in alerts[:3]:
        print(f"- {alert['type']}: {alert['message']}")
    
    # 导出指标
    output_path = monitor.export_metrics()
    print(f"\nMetrics exported to: {output_path}")


if __name__ == "__main__":
    example_usage()