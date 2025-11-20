"""
TS2Vec评估器

实现TS2Vec模型的评估功能，包括:
- Embedding质量评估
- 线性探测(Linear Probing)
- 聚类质量评估
- 下游任务性能评估

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    accuracy_score, f1_score, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split

from .model import TS2Vec

logger = logging.getLogger(__name__)


class TS2VecEvaluator:
    """
    TS2Vec评估器
    
    评估学习到的embedding质量。
    """
    
    def __init__(
        self,
        model: TS2Vec,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化评估器
        
        Args:
            model: TS2Vec模型
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        logger.info(f"TS2VecEvaluator initialized on device: {device}")
    
    def extract_embeddings(
        self,
        data: np.ndarray,
        batch_size: int = 128
    ) -> np.ndarray:
        """
        提取embedding
        
        Args:
            data: 输入数据 (N, seq_len, input_dim)
            batch_size: 批次大小
            
        Returns:
            embeddings: (N, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # 提取embedding
                emb = self.model(batch_tensor)
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def linear_probing_classification(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        线性探测 - 分类任务
        
        在冻结的embedding上训练线性分类器。
        
        Args:
            embeddings: Embedding向量
            labels: 分类标签
            test_size: 测试集比例
            
        Returns:
            评估指标
        """
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=test_size, random_state=42
        )
        
        # 训练逻辑回归
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        logger.info(f"Linear Probing (Classification) - "
                   f"Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def linear_probing_regression(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        线性探测 - 回归任务
        
        在冻结的embedding上训练线性回归器。
        
        Args:
            embeddings: Embedding向量
            targets: 回归目标
            test_size: 测试集比例
            
        Returns:
            评估指标
        """
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, targets, test_size=test_size, random_state=42
        )
        
        # 训练岭回归
        reg = Ridge(alpha=1.0, random_state=42)
        reg.fit(X_train, y_train)
        
        # 预测
        y_pred = reg.predict(X_test)
        
        # 计算指标
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Linear Probing (Regression) - "
                   f"MSE: {metrics['mse']:.4f}, "
                   f"R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def clustering_quality(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> Dict[str, float]:
        """
        聚类质量评估
        
        使用K-Means聚类评估embedding的可分性。
        
        Args:
            embeddings: Embedding向量
            n_clusters: 聚类数量
            
        Returns:
            聚类质量指标
        """
        # K-Means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 计算指标
        metrics = {
            'silhouette_score': silhouette_score(embeddings, cluster_labels),
            'davies_bouldin_score': davies_bouldin_score(embeddings, cluster_labels),
            'inertia': kmeans.inertia_
        }
        
        logger.info(f"Clustering Quality - "
                   f"Silhouette: {metrics['silhouette_score']:.4f}, "
                   f"Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}")
        
        return metrics
    
    def embedding_statistics(
        self,
        embeddings: np.ndarray
    ) -> Dict[str, float]:
        """
        Embedding统计信息
        
        Args:
            embeddings: Embedding向量
            
        Returns:
            统计指标
        """
        metrics = {
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'l2_norm_mean': float(np.mean(np.linalg.norm(embeddings, axis=1))),
            'l2_norm_std': float(np.std(np.linalg.norm(embeddings, axis=1)))
        }
        
        logger.info(f"Embedding Statistics - "
                   f"Mean: {metrics['mean']:.4f}, "
                   f"Std: {metrics['std']:.4f}, "
                   f"L2 Norm: {metrics['l2_norm_mean']:.4f}")
        
        return metrics
    
    def evaluate_comprehensive(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        n_clusters: int = 5
    ) -> Dict:
        """
        综合评估
        
        Args:
            data: 输入数据
            labels: 分类标签（可选）
            targets: 回归目标（可选）
            n_clusters: 聚类数量
            
        Returns:
            完整评估结果
        """
        logger.info("Starting comprehensive evaluation...")
        
        # 提取embedding
        embeddings = self.extract_embeddings(data)
        
        results = {}
        
        # 1. Embedding统计
        results['statistics'] = self.embedding_statistics(embeddings)
        
        # 2. 聚类质量
        results['clustering'] = self.clustering_quality(embeddings, n_clusters)
        
        # 3. 线性探测 - 分类
        if labels is not None:
            results['linear_probing_classification'] = \
                self.linear_probing_classification(embeddings, labels)
        
        # 4. 线性探测 - 回归
        if targets is not None:
            results['linear_probing_regression'] = \
                self.linear_probing_regression(embeddings, targets)
        
        logger.info("Comprehensive evaluation completed")
        
        return results


class EmbeddingExtractor:
    """
    Embedding提取器
    
    批量提取和缓存embedding。
    """
    
    def __init__(
        self,
        model: TS2Vec,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化提取器
        
        Args:
            model: TS2Vec模型
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        logger.info(f"EmbeddingExtractor initialized on device: {device}")
    
    def extract(
        self,
        data: np.ndarray,
        batch_size: int = 128,
        return_projection: bool = False
    ) -> np.ndarray:
        """
        提取embedding
        
        Args:
            data: 输入数据 (N, seq_len, input_dim)
            batch_size: 批次大小
            return_projection: 是否返回投影后的向量
            
        Returns:
            embeddings: (N, embedding_dim)
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                # 提取embedding
                if return_projection:
                    emb, proj = self.model(batch_tensor, return_projection=True)
                    embeddings.append(proj.cpu().numpy())
                else:
                    emb = self.model(batch_tensor)
                    embeddings.append(emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        logger.info(f"Extracted {len(embeddings)} embeddings")
        
        return embeddings
    
    def extract_and_save(
        self,
        data: np.ndarray,
        save_path: str,
        batch_size: int = 128
    ):
        """
        提取并保存embedding
        
        Args:
            data: 输入数据
            save_path: 保存路径
            batch_size: 批次大小
        """
        embeddings = self.extract(data, batch_size)
        np.save(save_path, embeddings)
        logger.info(f"Embeddings saved to: {save_path}")
    
    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        """
        加载保存的embedding
        
        Args:
            path: 文件路径
            
        Returns:
            embeddings
        """
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from: {path}, shape: {embeddings.shape}")
        return embeddings


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== TS2Vec评估器示例 ===")
    
    # 创建模拟数据
    np.random.seed(42)
    data = np.random.randn(1000, 256, 27)
    labels = np.random.randint(0, 3, 1000)  # 3类分类
    targets = np.random.randn(1000)  # 回归目标
    
    # 创建模型
    model = TS2Vec(input_dim=27)
    
    # 创建评估器
    evaluator = TS2VecEvaluator(model, device='cpu')
    
    # 综合评估
    print("\n执行综合评估...")
    results = evaluator.evaluate_comprehensive(
        data=data,
        labels=labels,
        targets=targets,
        n_clusters=3
    )
    
    print("\n评估结果:")
    print(f"Embedding统计: {results['statistics']}")
    print(f"聚类质量: {results['clustering']}")
    print(f"分类性能: {results['linear_probing_classification']}")
    print(f"回归性能: {results['linear_probing_regression']}")
    
    # Embedding提取器
    print("\n=== Embedding提取器示例 ===")
    extractor = EmbeddingExtractor(model, device='cpu')
    
    embeddings = extractor.extract(data, batch_size=128)
    print(f"提取的embedding形状: {embeddings.shape}")
    
    # 保存和加载
    save_path = 'embeddings_test.npy'
    extractor.extract_and_save(data, save_path)
    loaded_embeddings = EmbeddingExtractor.load_embeddings(save_path)
    print(f"加载的embedding形状: {loaded_embeddings.shape}")
    
    # 清理
    import os
    os.remove(save_path)
    print("\n示例完成!")