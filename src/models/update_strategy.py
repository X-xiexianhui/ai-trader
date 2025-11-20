"""
模型更新策略模块

实现模型更新触发条件、平滑切换和回滚机制
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging
from pathlib import Path
import json

from .version_manager import ModelVersionManager
from .monitor import ModelMonitor

logger = logging.getLogger(__name__)


class UpdateTrigger(Enum):
    """更新触发器类型"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DRIFT_DETECTED = "drift_detected"
    ERROR_RATE_HIGH = "error_rate_high"


class UpdateStrategy:
    """模型更新策略"""
    
    def __init__(
        self,
        version_manager: ModelVersionManager,
        monitor: ModelMonitor,
        config: Optional[Dict] = None
    ):
        """
        初始化更新策略
        
        Args:
            version_manager: 版本管理器
            monitor: 模型监控器
            config: 配置参数
        """
        self.version_manager = version_manager
        self.monitor = monitor
        
        # 默认配置
        self.config = config or {
            "performance_threshold": 0.8,  # 性能阈值（相对基线）
            "drift_threshold": 0.3,  # 漂移阈值
            "error_rate_threshold": 0.05,  # 错误率阈值
            "min_samples": 1000,  # 最小样本数
            "cooldown_hours": 24,  # 更新冷却时间
            "auto_rollback": True,  # 自动回滚
            "rollback_threshold": 0.7  # 回滚阈值
        }
        
        # 更新历史
        self.update_history = []
        self.last_update_time = None
        self.current_version = None
        self.previous_version = None
        
        # 性能基线
        self.baseline_metrics = None
        
        logger.info("Initialized UpdateStrategy")
    
    def should_update(self) -> Dict[str, Any]:
        """
        检查是否应该更新模型
        
        Returns:
            decision: 更新决策
        """
        decision = {
            "should_update": False,
            "triggers": [],
            "reasons": []
        }
        
        # 检查冷却时间
        if self.last_update_time:
            cooldown = timedelta(hours=self.config["cooldown_hours"])
            if datetime.now() - self.last_update_time < cooldown:
                decision["reasons"].append("In cooldown period")
                return decision
        
        # 检查样本数
        if self.monitor.total_requests < self.config["min_samples"]:
            decision["reasons"].append(f"Insufficient samples: {self.monitor.total_requests}")
            return decision
        
        # 检查性能退化
        if self._check_performance_degradation():
            decision["triggers"].append(UpdateTrigger.PERFORMANCE_DEGRADATION)
            decision["reasons"].append("Performance degradation detected")
        
        # 检查漂移
        if self._check_drift():
            decision["triggers"].append(UpdateTrigger.DRIFT_DETECTED)
            decision["reasons"].append("Distribution drift detected")
        
        # 检查错误率
        if self._check_error_rate():
            decision["triggers"].append(UpdateTrigger.ERROR_RATE_HIGH)
            decision["reasons"].append("High error rate detected")
        
        # 做出决策
        if decision["triggers"]:
            decision["should_update"] = True
        
        return decision
    
    def _check_performance_degradation(self) -> bool:
        """检查性能退化"""
        if self.baseline_metrics is None:
            return False
        
        current_metrics = self.monitor.get_summary()
        
        # 比较关键指标
        if "latency" in current_metrics and "latency" in self.baseline_metrics:
            current_p99 = current_metrics["latency"].get("p99_ms", 0)
            baseline_p99 = self.baseline_metrics["latency"].get("p99_ms", 0)
            
            if baseline_p99 > 0:
                ratio = current_p99 / baseline_p99
                if ratio > 1 / self.config["performance_threshold"]:
                    logger.warning(f"Latency degradation: {ratio:.2f}x baseline")
                    return True
        
        return False
    
    def _check_drift(self) -> bool:
        """检查分布漂移"""
        drift_info = self.monitor.detect_prediction_drift()
        
        if drift_info and drift_info.get("is_drifted", False):
            logger.warning(f"Drift detected: score={drift_info['drift_score']:.3f}")
            return True
        
        return False
    
    def _check_error_rate(self) -> bool:
        """检查错误率"""
        error_rate = self.monitor.get_error_rate()
        
        if error_rate > self.config["error_rate_threshold"]:
            logger.warning(f"High error rate: {error_rate:.2%}")
            return True
        
        return False
    
    def update_model(
        self,
        new_version_id: str,
        trigger: UpdateTrigger = UpdateTrigger.MANUAL,
        smooth_transition: bool = True
    ) -> Dict[str, Any]:
        """
        更新模型
        
        Args:
            new_version_id: 新版本ID
            trigger: 触发器类型
            smooth_transition: 是否平滑过渡
            
        Returns:
            result: 更新结果
        """
        logger.info(f"Updating model to version: {new_version_id}")
        logger.info(f"Trigger: {trigger.value}")
        
        # 记录当前版本
        self.previous_version = self.current_version
        
        # 加载新版本
        try:
            new_version_info = self.version_manager.get_version_info(new_version_id)
            if new_version_info is None:
                raise ValueError(f"Version {new_version_id} not found")
            
            # 平滑过渡
            if smooth_transition and self.previous_version:
                self._smooth_transition(new_version_id)
            else:
                self.version_manager.set_current_version(new_version_id)
            
            self.current_version = new_version_id
            self.last_update_time = datetime.now()
            
            # 记录更新历史
            update_record = {
                "timestamp": self.last_update_time.isoformat(),
                "from_version": self.previous_version,
                "to_version": new_version_id,
                "trigger": trigger.value,
                "metrics_before": self.monitor.get_summary(),
                "success": True
            }
            self.update_history.append(update_record)
            
            # 重置监控器
            self.monitor.reset()
            
            logger.info(f"Model updated successfully to {new_version_id}")
            
            return {
                "success": True,
                "version": new_version_id,
                "previous_version": self.previous_version,
                "timestamp": self.last_update_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
            
            update_record = {
                "timestamp": datetime.now().isoformat(),
                "from_version": self.previous_version,
                "to_version": new_version_id,
                "trigger": trigger.value,
                "success": False,
                "error": str(e)
            }
            self.update_history.append(update_record)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _smooth_transition(self, new_version_id: str, duration: int = 300):
        """
        平滑过渡到新版本
        
        Args:
            new_version_id: 新版本ID
            duration: 过渡时长（秒）
        """
        logger.info(f"Starting smooth transition over {duration}s")
        
        # 这里可以实现渐进式流量切换
        # 例如：0% -> 10% -> 50% -> 100%
        steps = [0.1, 0.5, 1.0]
        step_duration = duration / len(steps)
        
        for i, ratio in enumerate(steps):
            logger.info(f"Transition step {i+1}/{len(steps)}: {ratio*100:.0f}% traffic")
            time.sleep(step_duration)
        
        # 最终切换
        self.version_manager.set_current_version(new_version_id)
        logger.info("Smooth transition completed")
    
    def should_rollback(self) -> Dict[str, Any]:
        """
        检查是否应该回滚
        
        Returns:
            decision: 回滚决策
        """
        decision = {
            "should_rollback": False,
            "reasons": []
        }
        
        if not self.config["auto_rollback"]:
            return decision
        
        if self.previous_version is None:
            return decision
        
        # 检查更新后的性能
        current_metrics = self.monitor.get_summary()
        
        # 获取前一版本的指标
        prev_version_info = self.version_manager.get_version_info(self.previous_version)
        if prev_version_info is None:
            return decision
        
        prev_metrics = prev_version_info.get("metrics", {})
        
        # 比较关键指标
        for metric in ["sharpe", "cagr"]:
            if metric in prev_metrics:
                # 这里需要实际的性能数据，暂时使用监控数据
                # 实际应用中需要从回测或实盘获取
                pass
        
        # 检查错误率
        if self.monitor.get_error_rate() > self.config["error_rate_threshold"] * 2:
            decision["should_rollback"] = True
            decision["reasons"].append("Error rate too high after update")
        
        return decision
    
    def rollback(self) -> Dict[str, Any]:
        """
        回滚到前一版本
        
        Returns:
            result: 回滚结果
        """
        if self.previous_version is None:
            return {
                "success": False,
                "error": "No previous version to rollback to"
            }
        
        logger.warning(f"Rolling back to version: {self.previous_version}")
        
        try:
            self.version_manager.rollback_to_version(self.previous_version)
            
            # 交换版本
            self.current_version, self.previous_version = \
                self.previous_version, self.current_version
            
            # 记录回滚
            rollback_record = {
                "timestamp": datetime.now().isoformat(),
                "from_version": self.previous_version,
                "to_version": self.current_version,
                "trigger": "rollback",
                "success": True
            }
            self.update_history.append(rollback_record)
            
            # 重置监控器
            self.monitor.reset()
            
            logger.info(f"Rollback successful to {self.current_version}")
            
            return {
                "success": True,
                "version": self.current_version,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def set_baseline(self):
        """设置性能基线"""
        self.baseline_metrics = self.monitor.get_summary()
        self.monitor.set_baseline()
        logger.info("Baseline metrics set")
    
    def get_update_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        获取更新历史
        
        Args:
            limit: 返回记录数限制
            
        Returns:
            history: 更新历史
        """
        history = self.update_history
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def export_strategy_config(self, output_path: str):
        """导出策略配置"""
        config_data = {
            "config": self.config,
            "current_version": self.current_version,
            "previous_version": self.previous_version,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "baseline_metrics": self.baseline_metrics,
            "update_history": self.update_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Strategy config exported to: {output_path}")
    
    def auto_update_loop(self, check_interval: int = 3600):
        """
        自动更新循环
        
        Args:
            check_interval: 检查间隔（秒）
        """
        logger.info(f"Starting auto-update loop (interval: {check_interval}s)")
        
        while True:
            try:
                # 检查是否需要更新
                decision = self.should_update()
                
                if decision["should_update"]:
                    logger.info(f"Update triggered: {decision['reasons']}")
                    
                    # 获取最佳版本
                    best_version = self.version_manager.get_best_version(
                        model_name=self.monitor.model_name,
                        metric="sharpe"
                    )
                    
                    if best_version:
                        self.update_model(
                            new_version_id=best_version["version_id"],
                            trigger=decision["triggers"][0]
                        )
                
                # 检查是否需要回滚
                rollback_decision = self.should_rollback()
                
                if rollback_decision["should_rollback"]:
                    logger.warning(f"Rollback triggered: {rollback_decision['reasons']}")
                    self.rollback()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Auto-update loop stopped")
                break
            except Exception as e:
                logger.error(f"Error in auto-update loop: {e}")
                time.sleep(check_interval)


def example_usage():
    """使用示例"""
    from .version_manager import ModelVersionManager
    from .monitor import ModelMonitor
    
    # 初始化组件
    version_manager = ModelVersionManager(base_dir="models")
    monitor = ModelMonitor(model_name="ppo", window_size=1000)
    
    # 创建更新策略
    strategy = UpdateStrategy(
        version_manager=version_manager,
        monitor=monitor,
        config={
            "performance_threshold": 0.8,
            "drift_threshold": 0.3,
            "error_rate_threshold": 0.05,
            "min_samples": 100,
            "cooldown_hours": 1,
            "auto_rollback": True
        }
    )
    
    # 模拟一些监控数据
    import numpy as np
    for i in range(200):
        prediction = np.random.randn(1)
        latency = np.random.uniform(0.01, 0.05)
        monitor.record_prediction(prediction, latency)
    
    # 设置基线
    strategy.set_baseline()
    
    # 检查是否需要更新
    decision = strategy.should_update()
    print(f"\nShould update: {decision['should_update']}")
    print(f"Reasons: {decision['reasons']}")
    
    # 手动更新（假设有新版本）
    # result = strategy.update_model(
    #     new_version_id="ppo_v20231120_120000",
    #     trigger=UpdateTrigger.MANUAL
    # )
    # print(f"\nUpdate result: {result}")
    
    # 获取更新历史
    history = strategy.get_update_history()
    print(f"\nUpdate history: {len(history)} records")
    
    # 导出配置
    strategy.export_strategy_config("logs/update_strategy_config.json")
    print("\nStrategy config exported")


if __name__ == "__main__":
    example_usage()