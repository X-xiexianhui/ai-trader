"""
监控告警模块

实现服务监控和告警功能
"""

import time
import psutil
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """告警信息"""
    level: str  # info/warning/error/critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold
        }


@dataclass
class MonitoringConfig:
    """监控配置"""
    # 性能阈值
    max_latency_ms: float = 100.0
    max_memory_percent: float = 80.0
    max_cpu_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0
    
    # 错误率阈值
    max_error_rate: float = 0.05
    
    # 监控窗口
    window_size: int = 1000
    
    # 告警冷却期（秒）
    alert_cooldown: int = 300


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, config: MonitoringConfig):
        """
        初始化系统监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.alerts: deque = deque(maxlen=1000)
        self.last_alert_time: Dict[str, float] = {}
        
        # 性能指标
        self.latencies: deque = deque(maxlen=config.window_size)
        self.error_count = 0
        self.total_requests = 0
        
        logger.info("System monitor initialized")
    
    def record_request(self, latency: float, success: bool = True):
        """
        记录请求
        
        Args:
            latency: 延迟（秒）
            success: 是否成功
        """
        self.latencies.append(latency)
        self.total_requests += 1
        
        if not success:
            self.error_count += 1
    
    def check_latency(self) -> Optional[Alert]:
        """检查延迟"""
        if not self.latencies:
            return None
        
        avg_latency_ms = sum(self.latencies) / len(self.latencies) * 1000
        
        if avg_latency_ms > self.config.max_latency_ms:
            return Alert(
                level="warning",
                message=f"High latency detected: {avg_latency_ms:.2f}ms",
                metric_name="latency",
                metric_value=avg_latency_ms,
                threshold=self.config.max_latency_ms
            )
        
        return None
    
    def check_error_rate(self) -> Optional[Alert]:
        """检查错误率"""
        if self.total_requests == 0:
            return None
        
        error_rate = self.error_count / self.total_requests
        
        if error_rate > self.config.max_error_rate:
            return Alert(
                level="error",
                message=f"High error rate: {error_rate:.2%}",
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=self.config.max_error_rate
            )
        
        return None
    
    def check_memory(self) -> Optional[Alert]:
        """检查内存使用"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > self.config.max_memory_percent:
            return Alert(
                level="warning",
                message=f"High memory usage: {memory_percent:.1f}%",
                metric_name="memory",
                metric_value=memory_percent,
                threshold=self.config.max_memory_percent
            )
        
        return None
    
    def check_cpu(self) -> Optional[Alert]:
        """检查CPU使用"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > self.config.max_cpu_percent:
            return Alert(
                level="warning",
                message=f"High CPU usage: {cpu_percent:.1f}%",
                metric_name="cpu",
                metric_value=cpu_percent,
                threshold=self.config.max_cpu_percent
            )
        
        return None
    
    def check_gpu(self) -> Optional[Alert]:
        """检查GPU使用"""
        if not torch.cuda.is_available():
            return None
        
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            
            if gpu_memory_allocated > self.config.max_gpu_memory_percent:
                return Alert(
                    level="warning",
                    message=f"High GPU memory usage: {gpu_memory_allocated:.1f}%",
                    metric_name="gpu_memory",
                    metric_value=gpu_memory_allocated,
                    threshold=self.config.max_gpu_memory_percent
                )
        except:
            pass
        
        return None
    
    def check_all(self) -> List[Alert]:
        """执行所有检查"""
        alerts = []
        
        # 检查各项指标
        checks = [
            self.check_latency(),
            self.check_error_rate(),
            self.check_memory(),
            self.check_cpu(),
            self.check_gpu()
        ]
        
        for alert in checks:
            if alert is not None:
                # 检查冷却期
                if self._should_alert(alert.metric_name):
                    alerts.append(alert)
                    self.alerts.append(alert)
                    self.last_alert_time[alert.metric_name] = time.time()
                    logger.warning(f"Alert: {alert.message}")
        
        return alerts
    
    def _should_alert(self, metric_name: str) -> bool:
        """检查是否应该发送告警（考虑冷却期）"""
        if metric_name not in self.last_alert_time:
            return True
        
        elapsed = time.time() - self.last_alert_time[metric_name]
        return elapsed > self.config.alert_cooldown
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }
        
        # 延迟统计
        if self.latencies:
            import numpy as np
            latencies_ms = [l * 1000 for l in self.latencies]
            metrics["latency"] = {
                "avg_ms": np.mean(latencies_ms),
                "p50_ms": np.percentile(latencies_ms, 50),
                "p95_ms": np.percentile(latencies_ms, 95),
                "p99_ms": np.percentile(latencies_ms, 99),
                "max_ms": np.max(latencies_ms)
            }
        
        # GPU信息
        if torch.cuda.is_available():
            metrics["gpu"] = {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2
            }
        else:
            metrics["gpu"] = {"available": False}
        
        return metrics
    
    def get_alerts(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取告警列表
        
        Args:
            level: 告警级别过滤
            
        Returns:
            alerts: 告警列表
        """
        alerts = [alert.to_dict() for alert in self.alerts]
        
        if level:
            alerts = [a for a in alerts if a["level"] == level]
        
        return alerts
    
    def export_metrics(self, output_path: str):
        """
        导出监控指标
        
        Args:
            output_path: 输出文件路径
        """
        metrics = self.get_metrics()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {output_path}")


class PerformanceDashboard:
    """性能仪表板"""
    
    def __init__(self, monitor: SystemMonitor):
        """
        初始化仪表板
        
        Args:
            monitor: 系统监控器
        """
        self.monitor = monitor
    
    def print_dashboard(self):
        """打印仪表板"""
        metrics = self.monitor.get_metrics()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE DASHBOARD")
        print("=" * 60)
        
        # 请求统计
        print(f"\n📊 Request Statistics:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Error Count: {metrics['error_count']}")
        print(f"  Error Rate: {metrics['error_rate']:.2%}")
        
        # 延迟统计
        if "latency" in metrics:
            print(f"\n⏱️  Latency:")
            print(f"  Average: {metrics['latency']['avg_ms']:.2f}ms")
            print(f"  P95: {metrics['latency']['p95_ms']:.2f}ms")
            print(f"  P99: {metrics['latency']['p99_ms']:.2f}ms")
            print(f"  Max: {metrics['latency']['max_ms']:.2f}ms")
        
        # 系统资源
        print(f"\n💻 System Resources:")
        print(f"  CPU: {metrics['system']['cpu_percent']:.1f}%")
        print(f"  Memory: {metrics['system']['memory_percent']:.1f}%")
        print(f"  Disk: {metrics['system']['disk_percent']:.1f}%")
        
        # GPU信息
        if metrics['gpu']['available']:
            print(f"\n🎮 GPU:")
            print(f"  Device Count: {metrics['gpu']['device_count']}")
            print(f"  Memory Allocated: {metrics['gpu']['memory_allocated_mb']:.1f}MB")
            print(f"  Memory Reserved: {metrics['gpu']['memory_reserved_mb']:.1f}MB")
        
        # 告警
        alerts = self.monitor.get_alerts()
        if alerts:
            print(f"\n⚠️  Recent Alerts ({len(alerts)}):")
            for alert in alerts[-5:]:  # 显示最近5条
                print(f"  [{alert['level'].upper()}] {alert['message']}")
        
        print("\n" + "=" * 60)


def example_usage():
    """使用示例"""
    # 创建监控器
    config = MonitoringConfig(
        max_latency_ms=100.0,
        max_memory_percent=80.0,
        max_error_rate=0.05
    )
    monitor = SystemMonitor(config)
    
    # 模拟请求
    import random
    for i in range(100):
        latency = random.uniform(0.01, 0.15)
        success = random.random() > 0.02
        monitor.record_request(latency, success)
    
    # 检查告警
    alerts = monitor.check_all()
    print(f"Alerts generated: {len(alerts)}")
    
    # 显示仪表板
    dashboard = PerformanceDashboard(monitor)
    dashboard.print_dashboard()
    
    # 导出指标
    monitor.export_metrics("logs/monitoring/metrics.json")


if __name__ == "__main__":
    example_usage()