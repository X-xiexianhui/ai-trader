"""
模型版本管理模块

实现模型版本控制、性能跟踪和版本回退功能
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """模型版本管理器"""
    
    def __init__(self, base_dir: str = "models"):
        """
        初始化版本管理器
        
        Args:
            base_dir: 模型存储根目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_dir / "versions.json"
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict:
        """加载版本信息"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"versions": [], "current_version": None}
    
    def _save_versions(self):
        """保存版本信息"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def save_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        保存模型新版本
        
        Args:
            model: PyTorch模型
            model_name: 模型名称 (ts2vec/transformer/ppo)
            metrics: 性能指标字典
            config: 模型配置
            description: 版本描述
            tags: 版本标签
            
        Returns:
            version_id: 版本ID
        """
        # 生成版本ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_name}_v{timestamp}"
        
        # 创建版本目录
        version_dir = self.base_dir / model_name / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        model_path = version_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存完整模型（用于推理）
        full_model_path = version_dir / "model_full.pth"
        torch.save(model, full_model_path)
        
        # 保存配置
        if config:
            config_path = version_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        # 创建版本元数据
        version_info = {
            "version_id": version_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "datetime": datetime.now().isoformat(),
            "metrics": metrics,
            "config": config,
            "description": description,
            "tags": tags or [],
            "path": str(version_dir),
            "model_path": str(model_path),
            "full_model_path": str(full_model_path)
        }
        
        # 添加到版本列表
        self.versions["versions"].append(version_info)
        self.versions["current_version"] = version_id
        self._save_versions()
        
        logger.info(f"Saved model version: {version_id}")
        logger.info(f"Metrics: {metrics}")
        
        return version_id
    
    def load_model(
        self,
        version_id: Optional[str] = None,
        model_class: Optional[torch.nn.Module] = None,
        load_full: bool = False
    ) -> torch.nn.Module:
        """
        加载指定版本的模型
        
        Args:
            version_id: 版本ID，None则加载当前版本
            model_class: 模型类（如果load_full=False需要提供）
            load_full: 是否加载完整模型
            
        Returns:
            model: 加载的模型
        """
        if version_id is None:
            version_id = self.versions["current_version"]
            if version_id is None:
                raise ValueError("No current version set")
        
        # 查找版本信息
        version_info = self.get_version_info(version_id)
        if version_info is None:
            raise ValueError(f"Version {version_id} not found")
        
        if load_full:
            # 加载完整模型
            model = torch.load(version_info["full_model_path"])
        else:
            # 加载权重
            if model_class is None:
                raise ValueError("model_class required when load_full=False")
            model = model_class
            model.load_state_dict(torch.load(version_info["model_path"]))
        
        logger.info(f"Loaded model version: {version_id}")
        return model
    
    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """获取版本信息"""
        for version in self.versions["versions"]:
            if version["version_id"] == version_id:
                return version
        return None
    
    def list_versions(
        self,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_by: str = "timestamp",
        ascending: bool = False
    ) -> List[Dict]:
        """
        列出所有版本
        
        Args:
            model_name: 筛选模型名称
            tags: 筛选标签
            sort_by: 排序字段
            ascending: 是否升序
            
        Returns:
            versions: 版本列表
        """
        versions = self.versions["versions"]
        
        # 筛选模型名称
        if model_name:
            versions = [v for v in versions if v["model_name"] == model_name]
        
        # 筛选标签
        if tags:
            versions = [
                v for v in versions
                if any(tag in v.get("tags", []) for tag in tags)
            ]
        
        # 排序
        if sort_by in ["timestamp", "datetime"]:
            versions = sorted(
                versions,
                key=lambda x: x[sort_by],
                reverse=not ascending
            )
        elif sort_by in versions[0].get("metrics", {}):
            versions = sorted(
                versions,
                key=lambda x: x["metrics"].get(sort_by, float('-inf')),
                reverse=not ascending
            )
        
        return versions
    
    def get_best_version(
        self,
        model_name: str,
        metric: str = "sharpe",
        maximize: bool = True
    ) -> Optional[Dict]:
        """
        获取最佳版本
        
        Args:
            model_name: 模型名称
            metric: 评估指标
            maximize: 是否最大化指标
            
        Returns:
            version_info: 最佳版本信息
        """
        versions = self.list_versions(model_name=model_name)
        if not versions:
            return None
        
        # 筛选有该指标的版本
        valid_versions = [
            v for v in versions
            if metric in v.get("metrics", {})
        ]
        
        if not valid_versions:
            return None
        
        # 找到最佳版本
        best_version = max(
            valid_versions,
            key=lambda x: x["metrics"][metric] if maximize
            else -x["metrics"][metric]
        )
        
        return best_version
    
    def set_current_version(self, version_id: str):
        """设置当前版本"""
        version_info = self.get_version_info(version_id)
        if version_info is None:
            raise ValueError(f"Version {version_id} not found")
        
        self.versions["current_version"] = version_id
        self._save_versions()
        logger.info(f"Set current version to: {version_id}")
    
    def delete_version(self, version_id: str, keep_files: bool = False):
        """
        删除版本
        
        Args:
            version_id: 版本ID
            keep_files: 是否保留文件
        """
        version_info = self.get_version_info(version_id)
        if version_info is None:
            raise ValueError(f"Version {version_id} not found")
        
        # 删除文件
        if not keep_files:
            version_dir = Path(version_info["path"])
            if version_dir.exists():
                shutil.rmtree(version_dir)
        
        # 从列表中移除
        self.versions["versions"] = [
            v for v in self.versions["versions"]
            if v["version_id"] != version_id
        ]
        
        # 如果删除的是当前版本，清除当前版本
        if self.versions["current_version"] == version_id:
            self.versions["current_version"] = None
        
        self._save_versions()
        logger.info(f"Deleted version: {version_id}")
    
    def rollback_to_version(self, version_id: str) -> Dict:
        """
        回退到指定版本
        
        Args:
            version_id: 目标版本ID
            
        Returns:
            version_info: 版本信息
        """
        version_info = self.get_version_info(version_id)
        if version_info is None:
            raise ValueError(f"Version {version_id} not found")
        
        self.set_current_version(version_id)
        logger.info(f"Rolled back to version: {version_id}")
        
        return version_info
    
    def compare_versions(
        self,
        version_id1: str,
        version_id2: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        比较两个版本
        
        Args:
            version_id1: 版本1 ID
            version_id2: 版本2 ID
            metrics: 要比较的指标列表
            
        Returns:
            comparison: 比较结果
        """
        v1 = self.get_version_info(version_id1)
        v2 = self.get_version_info(version_id2)
        
        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")
        
        # 如果未指定指标，使用所有共同指标
        if metrics is None:
            metrics = list(set(v1["metrics"].keys()) & set(v2["metrics"].keys()))
        
        comparison = {
            "version1": version_id1,
            "version2": version_id2,
            "metrics": {}
        }
        
        for metric in metrics:
            val1 = v1["metrics"].get(metric)
            val2 = v2["metrics"].get(metric)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                
                comparison["metrics"][metric] = {
                    "version1": val1,
                    "version2": val2,
                    "difference": diff,
                    "percent_change": pct_change,
                    "better": version_id2 if val2 > val1 else version_id1
                }
        
        return comparison
    
    def export_version_history(self, output_path: str):
        """导出版本历史"""
        with open(output_path, 'w') as f:
            json.dump(self.versions, f, indent=2)
        logger.info(f"Exported version history to: {output_path}")
    
    def get_version_statistics(self, model_name: Optional[str] = None) -> Dict:
        """
        获取版本统计信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            stats: 统计信息
        """
        versions = self.list_versions(model_name=model_name)
        
        if not versions:
            return {"total_versions": 0}
        
        # 收集所有指标
        all_metrics = set()
        for v in versions:
            all_metrics.update(v.get("metrics", {}).keys())
        
        # 计算统计
        stats = {
            "total_versions": len(versions),
            "model_names": list(set(v["model_name"] for v in versions)),
            "date_range": {
                "earliest": min(v["datetime"] for v in versions),
                "latest": max(v["datetime"] for v in versions)
            },
            "metrics": {}
        }
        
        # 每个指标的统计
        for metric in all_metrics:
            values = [
                v["metrics"][metric]
                for v in versions
                if metric in v.get("metrics", {})
            ]
            
            if values:
                import numpy as np
                stats["metrics"][metric] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "best_version": max(
                        [v for v in versions if metric in v.get("metrics", {})],
                        key=lambda x: x["metrics"][metric]
                    )["version_id"]
                }
        
        return stats


def example_usage():
    """使用示例"""
    # 初始化版本管理器
    manager = ModelVersionManager(base_dir="models")
    
    # 假设我们有一个训练好的模型
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
    
    model = DummyModel()
    
    # 保存模型版本
    version_id = manager.save_model(
        model=model,
        model_name="ppo",
        metrics={
            "sharpe": 1.8,
            "cagr": 0.25,
            "max_dd": 0.15,
            "win_rate": 0.58
        },
        config={"hidden_dim": 512, "learning_rate": 1e-4},
        description="Initial PPO model with improved reward function",
        tags=["baseline", "production"]
    )
    
    print(f"Saved version: {version_id}")
    
    # 列出所有版本
    versions = manager.list_versions(model_name="ppo")
    print(f"\nTotal versions: {len(versions)}")
    
    # 获取最佳版本
    best = manager.get_best_version(model_name="ppo", metric="sharpe")
    if best:
        print(f"\nBest version by Sharpe: {best['version_id']}")
        print(f"Sharpe: {best['metrics']['sharpe']}")
    
    # 获取统计信息
    stats = manager.get_version_statistics(model_name="ppo")
    print(f"\nVersion statistics:")
    print(f"Total versions: {stats['total_versions']}")
    if "sharpe" in stats.get("metrics", {}):
        print(f"Sharpe - Mean: {stats['metrics']['sharpe']['mean']:.2f}, "
              f"Best: {stats['metrics']['sharpe']['max']:.2f}")


if __name__ == "__main__":
    example_usage()