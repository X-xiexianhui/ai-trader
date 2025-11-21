"""
TS2Vec评估模块

任务2.4.1-2.4.5实现:
1. 对比损失监控
2. Embedding质量评估
3. 线性探测评估
4. 聚类质量评估
5. t-SNE可视化
"""

import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TS2VecEvaluator:
    """
    TS2Vec评估器
    
    评估训练好的TS2Vec模型的质量
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model: 训练好的TS2Vec模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.evaluation_results = {}
        
    def monitor_contrastive_loss(self,
                                 train_losses: list,
                                 val_losses: list,
                                 save_path: str = 'loss_curves.png') -> Dict:
        """
        任务2.4.1: 对比损失监控
        
        监控和可视化训练/验证损失
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            save_path: 保存路径
            
        Returns:
            损失统计信息
        """
        logger.info("监控对比损失...")
        
        # 计算统计信息
        stats = {
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'min_train_loss': min(train_losses) if train_losses else None,
            'min_val_loss': min(val_losses) if val_losses else None,
            'converged': val_losses[-1] < 0.5 if val_losses else False
        }
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss')
        plt.title('TS2Vec Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"损失曲线已保存: {save_path}")
        logger.info(f"最终训练损失: {stats['final_train_loss']:.4f}")
        logger.info(f"最终验证损失: {stats['final_val_loss']:.4f}")
        
        self.evaluation_results['loss_stats'] = stats
        
        return stats
    
    def evaluate_embedding_quality(self,
                                   dataloader,
                                   target_pos_sim: float = 0.8,
                                   target_neg_sim: float = 0.3) -> Dict:
        """
        任务2.4.2: Embedding质量评估
        
        评估正负样本对的相似度
        
        Args:
            dataloader: 数据加载器
            target_pos_sim: 目标正样本相似度
            target_neg_sim: 目标负样本相似度
            
        Returns:
            质量评估报告
        """
        logger.info("评估embedding质量...")
        
        pos_similarities = []
        neg_similarities = []
        
        with torch.no_grad():
            for batch_idx, (x_i, x_j) in enumerate(dataloader):
                if batch_idx >= 100:  # 限制评估样本数
                    break
                
                x_i = x_i.to(self.device)
                x_j = x_j.to(self.device)
                
                # 获取embeddings
                z_i, z_j = self.model(x_i, x_j, return_loss=False)
                
                # 计算正样本相似度
                pos_sim = torch.cosine_similarity(
                    z_i.reshape(-1, z_i.shape[-1]),
                    z_j.reshape(-1, z_j.shape[-1]),
                    dim=1
                )
                pos_similarities.extend(pos_sim.cpu().numpy())
                
                # 计算负样本相似度（随机配对）
                batch_size = z_i.shape[0]
                if batch_size > 1:
                    # 随机打乱z_j
                    perm = torch.randperm(batch_size)
                    z_j_shuffled = z_j[perm]
                    
                    neg_sim = torch.cosine_similarity(
                        z_i.reshape(-1, z_i.shape[-1]),
                        z_j_shuffled.reshape(-1, z_j_shuffled.shape[-1]),
                        dim=1
                    )
                    neg_similarities.extend(neg_sim.cpu().numpy())
        
        # 计算统计量
        pos_similarities = np.array(pos_similarities)
        neg_similarities = np.array(neg_similarities)
        
        results = {
            'pos_sim_mean': float(pos_similarities.mean()),
            'pos_sim_std': float(pos_similarities.std()),
            'neg_sim_mean': float(neg_similarities.mean()),
            'neg_sim_std': float(neg_similarities.std()),
            'separation': float(pos_similarities.mean() - neg_similarities.mean()),
            'meets_target': (pos_similarities.mean() > target_pos_sim and 
                           neg_similarities.mean() < target_neg_sim)
        }
        
        logger.info(f"正样本相似度: {results['pos_sim_mean']:.4f} ± {results['pos_sim_std']:.4f}")
        logger.info(f"负样本相似度: {results['neg_sim_mean']:.4f} ± {results['neg_sim_std']:.4f}")
        logger.info(f"分离度: {results['separation']:.4f}")
        
        self.evaluation_results['embedding_quality'] = results
        
        return results
    
    def linear_probing(self,
                      train_data: torch.Tensor,
                      train_labels: torch.Tensor,
                      test_data: torch.Tensor,
                      test_labels: torch.Tensor,
                      target_accuracy: float = 0.55) -> Dict:
        """
        任务2.4.3: 线性探测评估
        
        冻结TS2Vec权重,训练线性分类器评估embedding质量
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签（涨跌方向）
            test_data: 测试数据
            test_labels: 测试标签
            target_accuracy: 目标准确率
            
        Returns:
            线性探测性能报告
        """
        logger.info("执行线性探测评估...")
        
        # 生成embeddings
        with torch.no_grad():
            train_data = train_data.to(self.device)
            test_data = test_data.to(self.device)
            
            train_embeddings = self.model.encode(train_data, return_projection=True)
            test_embeddings = self.model.encode(test_data, return_projection=True)
            
            # 平均池化
            train_embeddings = train_embeddings.mean(dim=1).cpu().numpy()
            test_embeddings = test_embeddings.mean(dim=1).cpu().numpy()
        
        # 训练线性分类器
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_embeddings, train_labels.numpy())
        
        # 预测
        train_pred = clf.predict(train_embeddings)
        test_pred = clf.predict(test_embeddings)
        
        # 计算指标
        train_acc = accuracy_score(train_labels.numpy(), train_pred)
        test_acc = accuracy_score(test_labels.numpy(), test_pred)
        
        # 计算AUC（如果是二分类）
        if len(np.unique(train_labels.numpy())) == 2:
            train_proba = clf.predict_proba(train_embeddings)[:, 1]
            test_proba = clf.predict_proba(test_embeddings)[:, 1]
            train_auc = roc_auc_score(train_labels.numpy(), train_proba)
            test_auc = roc_auc_score(test_labels.numpy(), test_proba)
        else:
            train_auc = None
            test_auc = None
        
        results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_auc': float(train_auc) if train_auc else None,
            'test_auc': float(test_auc) if test_auc else None,
            'meets_target': test_acc > target_accuracy
        }
        
        logger.info(f"训练准确率: {train_acc:.4f}")
        logger.info(f"测试准确率: {test_acc:.4f}")
        if test_auc:
            logger.info(f"测试AUC: {test_auc:.4f}")
        
        self.evaluation_results['linear_probing'] = results
        
        return results
    
    def clustering_quality(self,
                          data: torch.Tensor,
                          n_clusters: int = 5,
                          target_silhouette: float = 0.3) -> Dict:
        """
        任务2.4.4: 聚类质量评估
        
        评估embedding的聚类质量
        
        Args:
            data: 输入数据
            n_clusters: 聚类数量
            target_silhouette: 目标轮廓系数
            
        Returns:
            聚类质量报告
        """
        logger.info("评估聚类质量...")
        
        # 生成embeddings
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.model.encode(data, return_projection=True)
            
            # 平均池化
            embeddings = embeddings.mean(dim=1).cpu().numpy()
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 计算轮廓系数
        silhouette = silhouette_score(embeddings, cluster_labels)
        
        # 计算簇内距离和簇间距离
        inertia = kmeans.inertia_
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'inertia': float(inertia),
            'cluster_sizes': [int((cluster_labels == i).sum()) for i in range(n_clusters)],
            'meets_target': silhouette > target_silhouette
        }
        
        logger.info(f"聚类数量: {n_clusters}")
        logger.info(f"轮廓系数: {silhouette:.4f}")
        logger.info(f"簇大小: {results['cluster_sizes']}")
        
        self.evaluation_results['clustering'] = results
        
        return results
    
    def tsne_visualization(self,
                          data: torch.Tensor,
                          labels: Optional[np.ndarray] = None,
                          save_path: str = 'tsne_visualization.png',
                          n_samples: int = 1000) -> None:
        """
        任务2.4.5: t-SNE可视化
        
        可视化embedding的分布
        
        Args:
            data: 输入数据
            labels: 标签（用于着色）
            save_path: 保存路径
            n_samples: 采样数量
        """
        logger.info("生成t-SNE可视化...")
        
        # 生成embeddings
        with torch.no_grad():
            data = data.to(self.device)
            embeddings = self.model.encode(data, return_projection=True)
            
            # 平均池化
            embeddings = embeddings.mean(dim=1).cpu().numpy()
        
        # 采样（如果数据太多）
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=labels, cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, label='Label')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       alpha=0.6, s=20)
        
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('TS2Vec Embedding Visualization (t-SNE)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"t-SNE可视化已保存: {save_path}")
    
    def generate_evaluation_report(self,
                                  output_path: str = 'ts2vec_evaluation_report.txt') -> Dict:
        """
        生成完整的评估报告
        
        Returns:
            评估结果字典
        """
        logger.info("生成TS2Vec评估报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TS2Vec模型评估报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 损失统计
        if 'loss_stats' in self.evaluation_results:
            report_lines.append("1. 训练损失统计")
            report_lines.append("-" * 80)
            stats = self.evaluation_results['loss_stats']
            for key, value in stats.items():
                if value is not None:
                    if isinstance(value, bool):
                        report_lines.append(f"  {key}: {value}")
                    else:
                        report_lines.append(f"  {key}: {value:.4f}")
            report_lines.append("")
        
        # 2. Embedding质量
        if 'embedding_quality' in self.evaluation_results:
            report_lines.append("2. Embedding质量评估")
            report_lines.append("-" * 80)
            quality = self.evaluation_results['embedding_quality']
            for key, value in quality.items():
                if isinstance(value, bool):
                    report_lines.append(f"  {key}: {value}")
                else:
                    report_lines.append(f"  {key}: {value:.4f}")
            report_lines.append("")
        
        # 3. 线性探测
        if 'linear_probing' in self.evaluation_results:
            report_lines.append("3. 线性探测评估")
            report_lines.append("-" * 80)
            probing = self.evaluation_results['linear_probing']
            for key, value in probing.items():
                if value is not None:
                    if isinstance(value, bool):
                        report_lines.append(f"  {key}: {value}")
                    else:
                        report_lines.append(f"  {key}: {value:.4f}")
            report_lines.append("")
        
        # 4. 聚类质量
        if 'clustering' in self.evaluation_results:
            report_lines.append("4. 聚类质量评估")
            report_lines.append("-" * 80)
            clustering = self.evaluation_results['clustering']
            for key, value in clustering.items():
                if isinstance(value, (bool, int)):
                    report_lines.append(f"  {key}: {value}")
                elif isinstance(value, list):
                    report_lines.append(f"  {key}: {value}")
                else:
                    report_lines.append(f"  {key}: {value:.4f}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"评估报告已保存: {output_path}")
        
        return self.evaluation_results