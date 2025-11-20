"""
Transformer评估器实现

实现Transformer模型的评估功能，包括:
- 回归任务评估
- 分类任务评估
- 状态表征质量评估
- 注意力权重分析

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TransformerEvaluator:
    """
    Transformer评估器
    
    评估模型在回归和分类任务上的性能。
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化评估器
        
        Args:
            model: Transformer模型
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        logger.info(f"TransformerEvaluator initialized on {device}")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
            return_predictions: 是否返回预测结果
            
        Returns:
            评估指标字典
        """
        all_reg_preds = []
        all_reg_targets = []
        all_cls_preds = []
        all_cls_targets = []
        
        for batch in dataloader:
            # 移动数据到设备
            ts2vec_emb = batch['ts2vec_emb'].to(self.device)
            manual_features = batch['manual_features'].to(self.device)
            reg_target = batch['regression_target'].to(self.device)
            cls_target = batch['classification_target'].to(self.device)
            
            # 前向传播
            _, reg_pred, cls_pred = self.model(ts2vec_emb, manual_features)
            
            # 收集预测和目标
            all_reg_preds.append(reg_pred.cpu().numpy())
            all_reg_targets.append(reg_target.cpu().numpy())
            all_cls_preds.append(cls_pred.cpu().numpy())
            all_cls_targets.append(cls_target.cpu().numpy())
        
        # 合并所有批次
        reg_preds = np.concatenate(all_reg_preds, axis=0).reshape(-1)
        reg_targets = np.concatenate(all_reg_targets, axis=0).reshape(-1)
        cls_preds = np.concatenate(all_cls_preds, axis=0)
        cls_targets = np.concatenate(all_cls_targets, axis=0).reshape(-1)
        
        # 计算回归指标
        reg_metrics = self._compute_regression_metrics(reg_preds, reg_targets)
        
        # 计算分类指标
        cls_metrics = self._compute_classification_metrics(cls_preds, cls_targets)
        
        # 合并指标
        metrics = {**reg_metrics, **cls_metrics}
        
        if return_predictions:
            return metrics, {
                'reg_preds': reg_preds,
                'reg_targets': reg_targets,
                'cls_preds': cls_preds,
                'cls_targets': cls_targets
            }
        
        return metrics
    
    def _compute_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        计算回归指标
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            回归指标字典
        """
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 方向准确率
        pred_direction = np.sign(predictions)
        target_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == target_direction)
        
        return {
            'reg_mse': mse,
            'reg_rmse': rmse,
            'reg_mae': mae,
            'reg_r2': r2,
            'reg_direction_acc': direction_accuracy
        }
    
    def _compute_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        计算分类指标
        
        Args:
            predictions: 预测logits (n_samples, n_classes)
            targets: 目标标签 (n_samples,)
            
        Returns:
            分类指标字典
        """
        # 获取预测类别
        pred_classes = np.argmax(predictions, axis=-1).reshape(-1)
        
        # 准确率
        accuracy = accuracy_score(targets, pred_classes)
        
        # 精确率、召回率、F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, pred_classes, average='weighted', zero_division=0
        )
        
        # 尝试计算AUC（多分类）
        try:
            # 使用softmax概率
            probs = np.exp(predictions) / np.exp(predictions).sum(axis=-1, keepdims=True)
            auc = roc_auc_score(targets, probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        return {
            'cls_accuracy': accuracy,
            'cls_precision': precision,
            'cls_recall': recall,
            'cls_f1': f1,
            'cls_auc': auc
        }
    
    @torch.no_grad()
    def extract_attention_weights(
        self,
        ts2vec_emb: torch.Tensor,
        manual_features: torch.Tensor,
        layer_idx: int = -1
    ) -> np.ndarray:
        """
        提取注意力权重
        
        Args:
            ts2vec_emb: TS2Vec embedding
            manual_features: 手工特征
            layer_idx: 层索引（-1表示最后一层）
            
        Returns:
            注意力权重 (batch, num_heads, seq_len, seq_len)
        """
        ts2vec_emb = ts2vec_emb.to(self.device)
        manual_features = manual_features.to(self.device)
        
        # 前向传播并获取注意力权重
        _, _, _, attention_weights_list = self.model(
            ts2vec_emb, manual_features, return_attention=True
        )
        
        # 选择指定层的注意力权重
        attention_weights = attention_weights_list[layer_idx]
        
        return attention_weights.cpu().numpy()
    
    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重 (num_heads, seq_len, seq_len)
            save_path: 保存路径
        """
        num_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(num_heads, 8)):
            sns.heatmap(
                attention_weights[i],
                ax=axes[i],
                cmap='viridis',
                cbar=True
            )
            axes[i].set_title(f'Head {i + 1}')
            axes[i].set_xlabel('Key Position')
            axes[i].set_ylabel('Query Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        plt.close()
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        report = "=" * 60 + "\n"
        report += "Transformer Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        
        # 回归任务指标
        report += "Regression Task Metrics:\n"
        report += "-" * 60 + "\n"
        report += f"  MSE:                {metrics.get('reg_mse', 0):.6f}\n"
        report += f"  RMSE:               {metrics.get('reg_rmse', 0):.6f}\n"
        report += f"  MAE:                {metrics.get('reg_mae', 0):.6f}\n"
        report += f"  R²:                 {metrics.get('reg_r2', 0):.6f}\n"
        report += f"  Direction Accuracy: {metrics.get('reg_direction_acc', 0):.4f}\n\n"
        
        # 分类任务指标
        report += "Classification Task Metrics:\n"
        report += "-" * 60 + "\n"
        report += f"  Accuracy:  {metrics.get('cls_accuracy', 0):.4f}\n"
        report += f"  Precision: {metrics.get('cls_precision', 0):.4f}\n"
        report += f"  Recall:    {metrics.get('cls_recall', 0):.4f}\n"
        report += f"  F1 Score:  {metrics.get('cls_f1', 0):.4f}\n"
        report += f"  AUC:       {metrics.get('cls_auc', 0):.4f}\n\n"
        
        report += "=" * 60 + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report


class StateVectorExtractor:
    """
    状态向量提取器
    
    批量提取Transformer的状态向量表征。
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化提取器
        
        Args:
            model: Transformer模型
            device: 设备
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        logger.info(f"StateVectorExtractor initialized on {device}")
    
    @torch.no_grad()
    def extract(
        self,
        dataloader: DataLoader,
        return_last_only: bool = True
    ) -> np.ndarray:
        """
        批量提取状态向量
        
        Args:
            dataloader: 数据加载器
            return_last_only: 是否只返回序列最后一个时间步的状态
            
        Returns:
            状态向量数组
        """
        all_states = []
        
        for batch in dataloader:
            # 移动数据到设备
            ts2vec_emb = batch['ts2vec_emb'].to(self.device)
            manual_features = batch['manual_features'].to(self.device)
            
            # 前向传播
            state_vector, _, _ = self.model(ts2vec_emb, manual_features)
            
            if return_last_only:
                # 只取最后一个时间步
                state_vector = state_vector[:, -1, :]
            
            all_states.append(state_vector.cpu().numpy())
        
        # 合并所有批次
        states = np.concatenate(all_states, axis=0)
        
        logger.info(f"Extracted state vectors: shape={states.shape}")
        
        return states
    
    def extract_single(
        self,
        ts2vec_emb: np.ndarray,
        manual_features: np.ndarray
    ) -> np.ndarray:
        """
        提取单个样本的状态向量
        
        Args:
            ts2vec_emb: TS2Vec embedding (seq_len, embedding_dim)
            manual_features: 手工特征 (seq_len, feature_dim)
            
        Returns:
            状态向量 (seq_len, d_model)
        """
        # 转换为tensor并添加batch维度
        ts2vec_emb = torch.FloatTensor(ts2vec_emb).unsqueeze(0).to(self.device)
        manual_features = torch.FloatTensor(manual_features).unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            state_vector, _, _ = self.model(ts2vec_emb, manual_features)
        
        # 移除batch维度
        state_vector = state_vector.squeeze(0).cpu().numpy()
        
        return state_vector
    
    def save_states(
        self,
        states: np.ndarray,
        save_path: str
    ):
        """
        保存状态向量
        
        Args:
            states: 状态向量数组
            save_path: 保存路径
        """
        np.save(save_path, states)
        logger.info(f"State vectors saved to {save_path}")
    
    def load_states(self, load_path: str) -> np.ndarray:
        """
        加载状态向量
        
        Args:
            load_path: 加载路径
            
        Returns:
            状态向量数组
        """
        states = np.load(load_path)
        logger.info(f"State vectors loaded from {load_path}: shape={states.shape}")
        return states


# 示例用法
if __name__ == "__main__":
    from .model import TransformerModel
    from .dataset import TransformerDataModule
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Transformer评估器示例 ===")
    
    # 创建模拟数据
    n_samples = 200
    ts2vec_dim = 128
    feature_dim = 27
    
    test_ts2vec = np.random.randn(n_samples, ts2vec_dim)
    test_features = np.random.randn(n_samples, feature_dim)
    test_prices = np.cumsum(np.random.randn(n_samples) * 0.01) + 100
    
    # 创建数据模块
    data_module = TransformerDataModule(
        train_ts2vec=test_ts2vec,
        train_features=test_features,
        train_prices=test_prices,
        seq_len=64,
        batch_size=16
    )
    
    # 创建模型
    model = TransformerModel(
        ts2vec_dim=128,
        manual_dim=27,
        d_model=256,
        num_heads=8,
        num_layers=6
    )
    
    # 创建评估器
    evaluator = TransformerEvaluator(model, device='cpu')
    
    # 评估
    print("\n执行评估...")
    test_loader = data_module.train_dataloader()
    metrics = evaluator.evaluate(test_loader)
    
    # 生成报告
    report = evaluator.generate_report(metrics)
    print("\n" + report)
    
    # 创建状态向量提取器
    print("\n=== 状态向量提取示例 ===")
    extractor = StateVectorExtractor(model, device='cpu')
    
    # 提取状态向量
    states = extractor.extract(test_loader, return_last_only=True)
    print(f"提取的状态向量形状: {states.shape}")
    
    print("\n示例完成!")