"""
Transformer辅助任务实现

任务3.3.1-3.3.3实现:
1. 回归辅助头(预测未来收益率)
2. 分类辅助头(预测涨跌方向)
3. 多任务损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RegressionHead(nn.Module):
    """
    任务3.3.1: 回归辅助头
    
    预测未来收益率
    """
    
    def __init__(self, d_model: int = 256):
        """
        初始化回归头
        
        Args:
            d_model: 输入维度
        """
        super().__init__()
        
        self.fc = nn.Linear(d_model, 1)
        
        logger.info(f"回归头初始化: {d_model}→1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态向量 [batch, d_model]
            
        Returns:
            预测的未来收益率 [batch, 1]
        """
        return self.fc(x)


class ClassificationHead(nn.Module):
    """
    任务3.3.2: 分类辅助头
    
    预测未来涨跌方向(涨/平/跌)
    """
    
    def __init__(self, d_model: int = 256, num_classes: int = 3):
        """
        初始化分类头
        
        Args:
            d_model: 输入维度
            num_classes: 类别数(3: 涨/平/跌)
        """
        super().__init__()
        
        self.fc = nn.Linear(d_model, num_classes)
        
        logger.info(f"分类头初始化: {d_model}→{num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 状态向量 [batch, d_model]
            
        Returns:
            类别logits [batch, num_classes]
        """
        return self.fc(x)


class MultiTaskLoss(nn.Module):
    """
    任务3.3.3: 多任务损失函数
    
    组合回归损失和分类损失
    """
    
    def __init__(self,
                 lambda_reg: float = 0.1,
                 lambda_cls: float = 0.05):
        """
        初始化多任务损失
        
        Args:
            lambda_reg: 回归损失权重
            lambda_cls: 分类损失权重
        """
        super().__init__()
        
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        logger.info(f"多任务损失初始化: λ_reg={lambda_reg}, λ_cls={lambda_cls}")
    
    def forward(self,
                reg_pred: torch.Tensor,
                reg_target: torch.Tensor,
                cls_pred: torch.Tensor,
                cls_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多任务损失
        
        Args:
            reg_pred: 回归预测 [batch, 1]
            reg_target: 回归目标 [batch, 1]
            cls_pred: 分类预测 [batch, num_classes]
            cls_target: 分类目标 [batch] (类别索引)
            
        Returns:
            总损失, 损失字典
        """
        # 回归损失
        loss_reg = self.mse_loss(reg_pred, reg_target)
        
        # 分类损失
        loss_cls = self.ce_loss(cls_pred, cls_target)
        
        # 总损失
        total_loss = self.lambda_reg * loss_reg + self.lambda_cls * loss_cls
        
        # 损失字典
        loss_dict = {
            'total': total_loss.item(),
            'regression': loss_reg.item(),
            'classification': loss_cls.item()
        }
        
        return total_loss, loss_dict


class TransformerWithAuxiliaryTasks(nn.Module):
    """
    带辅助任务的完整Transformer模型
    
    整合主模型和辅助任务头
    """
    
    def __init__(self,
                 transformer_model: nn.Module,
                 d_model: int = 256,
                 num_classes: int = 3):
        """
        初始化模型
        
        Args:
            transformer_model: Transformer主模型
            d_model: 模型维度
            num_classes: 分类类别数
        """
        super().__init__()
        
        self.transformer = transformer_model
        self.regression_head = RegressionHead(d_model)
        self.classification_head = ClassificationHead(d_model, num_classes)
        
        logger.info("带辅助任务的Transformer模型初始化")
    
    def forward(self,
                x: torch.Tensor,
                pooling: str = 'last') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            pooling: 池化方法
            
        Returns:
            状态向量, 回归预测, 分类预测
        """
        # 获取状态向量
        state = self.transformer.get_state_vector(x, pooling)
        
        # 辅助任务预测
        reg_pred = self.regression_head(state)
        cls_pred = self.classification_head(state)
        
        return state, reg_pred, cls_pred
    
    def predict(self,
                x: torch.Tensor,
                pooling: str = 'last') -> Dict[str, torch.Tensor]:
        """
        预测(推理模式)
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            pooling: 池化方法
            
        Returns:
            预测结果字典
        """
        self.eval()
        with torch.no_grad():
            state, reg_pred, cls_pred = self.forward(x, pooling)
            
            # 分类概率
            cls_prob = F.softmax(cls_pred, dim=-1)
            cls_label = torch.argmax(cls_prob, dim=-1)
            
            return {
                'state': state,
                'return_pred': reg_pred,
                'direction_prob': cls_prob,
                'direction_label': cls_label
            }


def create_labels_from_returns(returns: torch.Tensor,
                               threshold: float = 0.001) -> torch.Tensor:
    """
    从收益率创建分类标签
    
    Args:
        returns: 收益率 [batch]
        threshold: 涨跌阈值
        
    Returns:
        分类标签 [batch] (0=跌, 1=平, 2=涨)
    """
    labels = torch.zeros_like(returns, dtype=torch.long)
    labels[returns > threshold] = 2  # 涨
    labels[returns < -threshold] = 0  # 跌
    labels[(returns >= -threshold) & (returns <= threshold)] = 1  # 平
    
    return labels


def compute_auxiliary_metrics(reg_pred: torch.Tensor,
                              reg_target: torch.Tensor,
                              cls_pred: torch.Tensor,
                              cls_target: torch.Tensor) -> Dict[str, float]:
    """
    计算辅助任务的评估指标
    
    Args:
        reg_pred: 回归预测 [batch, 1]
        reg_target: 回归目标 [batch, 1]
        cls_pred: 分类预测 [batch, num_classes]
        cls_target: 分类目标 [batch]
        
    Returns:
        指标字典
    """
    with torch.no_grad():
        # 回归指标
        mse = F.mse_loss(reg_pred, reg_target).item()
        mae = F.l1_loss(reg_pred, reg_target).item()
        
        # 方向准确率(预测和目标的符号是否一致)
        pred_sign = torch.sign(reg_pred.squeeze())
        target_sign = torch.sign(reg_target.squeeze())
        direction_acc = (pred_sign == target_sign).float().mean().item()
        
        # 分类指标
        cls_pred_label = torch.argmax(cls_pred, dim=-1)
        cls_acc = (cls_pred_label == cls_target).float().mean().item()
        
        return {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            'classification_accuracy': cls_acc
        }