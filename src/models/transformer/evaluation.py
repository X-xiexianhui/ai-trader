"""
Transformer评估指标实现

任务3.5.1-3.5.3实现:
1. 监督学习指标计算
2. 状态表征质量评估
3. 注意力权重可视化
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import r2_score, roc_auc_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class SupervisedMetrics:
    """
    任务3.5.1: 监督学习指标计算
    
    计算回归和分类任务的评估指标
    """
    
    @staticmethod
    def compute_regression_metrics(predictions: np.ndarray,
                                   targets: np.ndarray) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            predictions: 预测值 [N]
            targets: 目标值 [N]
            
        Returns:
            指标字典
        """
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # R²
        r2 = r2_score(targets, predictions)
        
        # 方向准确率
        pred_sign = np.sign(predictions)
        target_sign = np.sign(targets)
        direction_acc = np.mean(pred_sign == target_sign)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_acc)
        }
    
    @staticmethod
    def compute_classification_metrics(predictions: np.ndarray,
                                       targets: np.ndarray,
                                       probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            predictions: 预测类别 [N]
            targets: 目标类别 [N]
            probabilities: 预测概率 [N, num_classes] (可选)
            
        Returns:
            指标字典
        """
        # 准确率
        accuracy = np.mean(predictions == targets)
        
        # F1分数(宏平均)
        f1_macro = f1_score(targets, predictions, average='macro')
        
        # F1分数(加权平均)
        f1_weighted = f1_score(targets, predictions, average='weighted')
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted)
        }
        
        # AUC-ROC(如果提供概率)
        if probabilities is not None and len(np.unique(targets)) > 1:
            try:
                # 多分类AUC(OvR)
                auc = roc_auc_score(
                    targets,
                    probabilities,
                    multi_class='ovr',
                    average='macro'
                )
                metrics['auc_roc'] = float(auc)
            except Exception as e:
                logger.warning(f"AUC计算失败: {e}")
        
        return metrics
    
    @staticmethod
    def evaluate_model(model: nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: str = 'cuda') -> Dict[str, float]:
        """
        完整评估模型
        
        Args:
            model: Transformer模型
            dataloader: 数据加载器
            device: 设备
            
        Returns:
            所有指标
        """
        model.eval()
        
        all_reg_preds = []
        all_reg_targets = []
        all_cls_preds = []
        all_cls_targets = []
        all_cls_probs = []
        
        with torch.no_grad():
            for sequences, reg_targets in dataloader:
                sequences = sequences.to(device)
                reg_targets = reg_targets.to(device)
                
                # 前向传播
                results = model.predict(sequences)
                
                # 收集预测
                all_reg_preds.append(results['return_pred'].cpu().numpy())
                all_reg_targets.append(reg_targets.cpu().numpy())
                all_cls_preds.append(results['direction_label'].cpu().numpy())
                all_cls_probs.append(results['direction_prob'].cpu().numpy())
                
                # 创建分类目标
                from .auxiliary_tasks import create_labels_from_returns
                cls_targets = create_labels_from_returns(reg_targets.squeeze())
                all_cls_targets.append(cls_targets.cpu().numpy())
        
        # 合并所有批次
        reg_preds = np.concatenate(all_reg_preds).flatten()
        reg_targets = np.concatenate(all_reg_targets).flatten()
        cls_preds = np.concatenate(all_cls_preds)
        cls_targets = np.concatenate(all_cls_targets)
        cls_probs = np.concatenate(all_cls_probs)
        
        # 计算指标
        reg_metrics = SupervisedMetrics.compute_regression_metrics(
            reg_preds, reg_targets
        )
        cls_metrics = SupervisedMetrics.compute_classification_metrics(
            cls_preds, cls_targets, cls_probs
        )
        
        # 合并指标
        all_metrics = {
            **{f'reg_{k}': v for k, v in reg_metrics.items()},
            **{f'cls_{k}': v for k, v in cls_metrics.items()}
        }
        
        return all_metrics


class StateRepresentationQuality:
    """
    任务3.5.2: 状态表征质量评估
    
    评估状态向量的质量
    """
    
    @staticmethod
    def compute_variance(states: np.ndarray) -> Dict[str, float]:
        """
        计算状态向量的方差
        
        Args:
            states: 状态向量 [N, d_model]
            
        Returns:
            方差统计
        """
        # 每个维度的方差
        dim_variance = np.var(states, axis=0)
        
        # 总体方差
        total_variance = np.mean(dim_variance)
        
        # 最小和最大方差
        min_variance = np.min(dim_variance)
        max_variance = np.max(dim_variance)
        
        return {
            'mean_variance': float(total_variance),
            'min_variance': float(min_variance),
            'max_variance': float(max_variance),
            'variance_std': float(np.std(dim_variance))
        }
    
    @staticmethod
    def compute_norm_distribution(states: np.ndarray) -> Dict[str, float]:
        """
        计算状态向量范数的分布
        
        Args:
            states: 状态向量 [N, d_model]
            
        Returns:
            范数统计
        """
        # L2范数
        norms = np.linalg.norm(states, axis=1)
        
        return {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms))
        }
    
    @staticmethod
    def compute_separability(states: np.ndarray,
                            labels: np.ndarray) -> Dict[str, float]:
        """
        计算不同类别状态的可分离性
        
        Args:
            states: 状态向量 [N, d_model]
            labels: 类别标签 [N]
            
        Returns:
            可分离性指标
        """
        unique_labels = np.unique(labels)
        
        # 类内距离
        intra_distances = []
        for label in unique_labels:
            mask = labels == label
            class_states = states[mask]
            if len(class_states) > 1:
                centroid = np.mean(class_states, axis=0)
                distances = np.linalg.norm(class_states - centroid, axis=1)
                intra_distances.extend(distances)
        
        # 类间距离
        inter_distances = []
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(states[mask], axis=0)
            centroids.append(centroid)
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[j])
                inter_distances.append(distance)
        
        # 可分离性比率
        mean_intra = np.mean(intra_distances) if intra_distances else 0
        mean_inter = np.mean(inter_distances) if inter_distances else 0
        separability = mean_inter / (mean_intra + 1e-8)
        
        return {
            'mean_intra_distance': float(mean_intra),
            'mean_inter_distance': float(mean_inter),
            'separability_ratio': float(separability)
        }
    
    @staticmethod
    def evaluate_states(model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: str = 'cuda') -> Dict[str, float]:
        """
        完整评估状态表征质量
        
        Args:
            model: Transformer模型
            dataloader: 数据加载器
            device: 设备
            
        Returns:
            质量指标
        """
        model.eval()
        
        all_states = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, reg_targets in dataloader:
                sequences = sequences.to(device)
                reg_targets = reg_targets.to(device)
                
                # 获取状态向量
                results = model.predict(sequences)
                all_states.append(results['state'].cpu().numpy())
                
                # 创建标签
                from .auxiliary_tasks import create_labels_from_returns
                labels = create_labels_from_returns(reg_targets.squeeze())
                all_labels.append(labels.cpu().numpy())
        
        # 合并
        states = np.concatenate(all_states)
        labels = np.concatenate(all_labels)
        
        # 计算各项指标
        variance_metrics = StateRepresentationQuality.compute_variance(states)
        norm_metrics = StateRepresentationQuality.compute_norm_distribution(states)
        separability_metrics = StateRepresentationQuality.compute_separability(
            states, labels
        )
        
        return {
            **variance_metrics,
            **norm_metrics,
            **separability_metrics
        }


class AttentionVisualizer:
    """
    任务3.5.3: 注意力权重可视化
    
    提取和可视化注意力权重
    """
    
    @staticmethod
    def extract_attention_weights(model: nn.Module,
                                  sequence: torch.Tensor,
                                  layer_idx: int = -1) -> np.ndarray:
        """
        提取注意力权重
        
        Args:
            model: Transformer模型
            sequence: 输入序列 [1, seq_len, input_dim]
            layer_idx: 层索引(-1表示最后一层)
            
        Returns:
            注意力权重 [nhead, seq_len, seq_len]
        """
        model.eval()
        
        # 需要修改模型以返回注意力权重
        # 这里提供一个简化的实现框架
        logger.warning("注意力权重提取需要修改模型forward方法")
        
        # 占位符实现
        seq_len = sequence.size(1)
        nhead = 8  # 默认头数
        
        # 返回随机权重作为示例
        return np.random.rand(nhead, seq_len, seq_len)
    
    @staticmethod
    def plot_attention_heatmap(attention_weights: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        绘制注意力热力图
        
        Args:
            attention_weights: 注意力权重 [nhead, seq_len, seq_len]
            save_path: 保存路径
        """
        nhead, seq_len, _ = attention_weights.shape
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(nhead, 8)):
            sns.heatmap(
                attention_weights[i],
                ax=axes[i],
                cmap='viridis',
                cbar=True
            )
            axes[i].set_title(f'Head {i+1}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"注意力热力图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_tsne_visualization(states: np.ndarray,
                               labels: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        绘制状态向量的t-SNE可视化
        
        Args:
            states: 状态向量 [N, d_model]
            labels: 标签 [N]
            save_path: 保存路径
        """
        # t-SNE降维
        logger.info("执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        states_2d = tsne.fit_transform(states)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            states_2d[:, 0],
            states_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        plt.colorbar(scatter, label='Direction Label')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.title('State Vector t-SNE Visualization')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"t-SNE可视化已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()