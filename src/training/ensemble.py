"""
模型集成

实现多模型集成和预测不确定性估计
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Callable
import logging
from pathlib import Path
import copy

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """模型集成器"""
    
    def __init__(self,
                 models: Optional[List[nn.Module]] = None,
                 weights: Optional[List[float]] = None,
                 method: str = 'average'):
        """
        初始化模型集成器
        
        Args:
            models: 模型列表
            weights: 模型权重
            method: 集成方法 ('average', 'weighted', 'voting')
        """
        self.models = models or []
        self.method = method
        
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models) if self.models else []
        else:
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"模型集成器初始化: {len(self.models)}个模型, 方法={method}")
    
    def add_model(self, model: nn.Module, weight: float = 1.0):
        """
        添加模型
        
        Args:
            model: 模型
            weight: 权重
        """
        self.models.append(model)
        self.weights.append(weight)
        
        # 重新归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        logger.info(f"添加模型，当前共{len(self.models)}个模型")
    
    def predict(self, x: torch.Tensor, return_std: bool = False):
        """
        集成预测
        
        Args:
            x: 输入数据
            return_std: 是否返回标准差（不确定性）
            
        Returns:
            预测结果，如果return_std=True则返回(mean, std)
        """
        if not self.models:
            raise ValueError("没有可用的模型")
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_models, batch_size, ...]
        
        if self.method == 'average':
            mean_pred = torch.mean(predictions, dim=0)
        elif self.method == 'weighted':
            weights_tensor = torch.tensor(self.weights, device=x.device).view(-1, 1, 1)
            mean_pred = torch.sum(predictions * weights_tensor, dim=0)
        elif self.method == 'voting':
            # 对于分类任务的投票
            mean_pred = torch.mode(predictions, dim=0)[0]
        else:
            raise ValueError(f"未知的集成方法: {self.method}")
        
        if return_std:
            std_pred = torch.std(predictions, dim=0)
            return mean_pred, std_pred
        
        return mean_pred
    
    def save(self, save_dir: str):
        """
        保存所有模型
        
        Args:
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_path / f"model_{i}.pt"
            torch.save(model.state_dict(), model_path)
        
        # 保存权重
        weights_path = save_path / "weights.npy"
        np.save(weights_path, np.array(self.weights))
        
        logger.info(f"集成模型已保存到: {save_dir}")
    
    def load(self, load_dir: str, model_class: Callable, model_kwargs: Dict):
        """
        加载所有模型
        
        Args:
            load_dir: 加载目录
            model_class: 模型类
            model_kwargs: 模型参数
        """
        load_path = Path(load_dir)
        
        # 加载权重
        weights_path = load_path / "weights.npy"
        if weights_path.exists():
            self.weights = np.load(weights_path).tolist()
        
        # 加载模型
        self.models = []
        i = 0
        while True:
            model_path = load_path / f"model_{i}.pt"
            if not model_path.exists():
                break
            
            model = model_class(**model_kwargs)
            model.load_state_dict(torch.load(model_path))
            self.models.append(model)
            i += 1
        
        logger.info(f"从{load_dir}加载了{len(self.models)}个模型")


class BootstrapEnsemble:
    """Bootstrap集成"""
    
    def __init__(self,
                 base_model_class: Callable,
                 model_kwargs: Dict,
                 n_models: int = 5,
                 sample_ratio: float = 0.8):
        """
        初始化Bootstrap集成
        
        Args:
            base_model_class: 基础模型类
            model_kwargs: 模型参数
            n_models: 模型数量
            sample_ratio: 采样比例
        """
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.n_models = n_models
        self.sample_ratio = sample_ratio
        
        self.ensemble = ModelEnsemble(method='average')
        
        logger.info(f"Bootstrap集成初始化: {n_models}个模型")
    
    def fit(self,
            train_data: torch.utils.data.Dataset,
            train_fn: Callable,
            **train_kwargs):
        """
        训练Bootstrap集成
        
        Args:
            train_data: 训练数据集
            train_fn: 训练函数
            **train_kwargs: 训练参数
        """
        n_samples = len(train_data)
        sample_size = int(n_samples * self.sample_ratio)
        
        for i in range(self.n_models):
            logger.info(f"训练第{i+1}/{self.n_models}个模型...")
            
            # Bootstrap采样
            indices = np.random.choice(n_samples, sample_size, replace=True)
            subset = torch.utils.data.Subset(train_data, indices)
            
            # 创建并训练模型
            model = self.base_model_class(**self.model_kwargs)
            trained_model = train_fn(model, subset, **train_kwargs)
            
            # 添加到集成
            self.ensemble.add_model(trained_model)
        
        logger.info("Bootstrap集成训练完成")
    
    def predict(self, x: torch.Tensor, return_std: bool = False):
        """预测"""
        return self.ensemble.predict(x, return_std)


class SnapshotEnsemble:
    """快照集成"""
    
    def __init__(self,
                 model: nn.Module,
                 n_snapshots: int = 5,
                 cycle_length: int = 10):
        """
        初始化快照集成
        
        Args:
            model: 基础模型
            n_snapshots: 快照数量
            cycle_length: 循环长度（epoch）
        """
        self.base_model = model
        self.n_snapshots = n_snapshots
        self.cycle_length = cycle_length
        
        self.ensemble = ModelEnsemble(method='average')
        self.snapshots = []
        
        logger.info(f"快照集成初始化: {n_snapshots}个快照")
    
    def should_save_snapshot(self, epoch: int) -> bool:
        """
        判断是否应该保存快照
        
        Args:
            epoch: 当前epoch
            
        Returns:
            bool: 是否保存
        """
        # 在每个循环的末尾保存快照
        return (epoch + 1) % self.cycle_length == 0
    
    def save_snapshot(self):
        """保存当前模型快照"""
        snapshot = copy.deepcopy(self.base_model)
        self.snapshots.append(snapshot)
        self.ensemble.add_model(snapshot)
        
        logger.info(f"保存快照，当前共{len(self.snapshots)}个快照")
    
    def predict(self, x: torch.Tensor, return_std: bool = False):
        """预测"""
        return self.ensemble.predict(x, return_std)


class BaggingEnsemble:
    """Bagging集成"""
    
    def __init__(self,
                 base_model_class: Callable,
                 model_kwargs: Dict,
                 n_models: int = 5,
                 max_samples: float = 0.8,
                 max_features: float = 1.0):
        """
        初始化Bagging集成
        
        Args:
            base_model_class: 基础模型类
            model_kwargs: 模型参数
            n_models: 模型数量
            max_samples: 最大样本比例
            max_features: 最大特征比例
        """
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.n_models = n_models
        self.max_samples = max_samples
        self.max_features = max_features
        
        self.ensemble = ModelEnsemble(method='average')
        self.feature_indices = []
        
        logger.info(f"Bagging集成初始化: {n_models}个模型")
    
    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            train_fn: Callable,
            **train_kwargs):
        """
        训练Bagging集成
        
        Args:
            X: 特征
            y: 标签
            train_fn: 训练函数
            **train_kwargs: 训练参数
        """
        n_samples, n_features = X.shape
        sample_size = int(n_samples * self.max_samples)
        feature_size = int(n_features * self.max_features)
        
        for i in range(self.n_models):
            logger.info(f"训练第{i+1}/{self.n_models}个模型...")
            
            # 采样样本
            sample_indices = np.random.choice(n_samples, sample_size, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # 采样特征
            feature_indices = np.random.choice(n_features, feature_size, replace=False)
            X_sample = X_sample[:, feature_indices]
            self.feature_indices.append(feature_indices)
            
            # 训练模型
            model = self.base_model_class(**self.model_kwargs)
            trained_model = train_fn(model, X_sample, y_sample, **train_kwargs)
            
            self.ensemble.add_model(trained_model)
        
        logger.info("Bagging集成训练完成")
    
    def predict(self, X: torch.Tensor, return_std: bool = False):
        """
        预测
        
        Args:
            X: 输入特征
            return_std: 是否返回标准差
            
        Returns:
            预测结果
        """
        predictions = []
        
        for i, model in enumerate(self.ensemble.models):
            # 使用对应的特征子集
            X_subset = X[:, self.feature_indices[i]]
            
            model.eval()
            with torch.no_grad():
                pred = model(X_subset)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = torch.mean(predictions, dim=0)
        
        if return_std:
            std_pred = torch.std(predictions, dim=0)
            return mean_pred, std_pred
        
        return mean_pred


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.fc(x)
    
    # 测试模型集成
    print("测试模型集成...")
    ensemble = ModelEnsemble(method='average')
    
    for i in range(3):
        model = SimpleModel()
        ensemble.add_model(model)
    
    x = torch.randn(32, 10)
    pred_mean, pred_std = ensemble.predict(x, return_std=True)
    
    print(f"预测形状: {pred_mean.shape}")
    print(f"不确定性形状: {pred_std.shape}")
    print(f"平均不确定性: {pred_std.mean().item():.4f}")
    
    # 测试快照集成
    print("\n测试快照集成...")
    model = SimpleModel()
    snapshot_ensemble = SnapshotEnsemble(model, n_snapshots=3, cycle_length=5)
    
    for epoch in range(15):
        if snapshot_ensemble.should_save_snapshot(epoch):
            snapshot_ensemble.save_snapshot()
            print(f"Epoch {epoch}: 保存快照")
    
    pred = snapshot_ensemble.predict(x)
    print(f"快照集成预测形状: {pred.shape}")