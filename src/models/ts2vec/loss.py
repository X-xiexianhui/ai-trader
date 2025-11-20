"""
TS2Vec对比学习损失函数

实现NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失函数，
用于对比学习训练。

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NTXentLoss(nn.Module):
    """
    NT-Xent损失函数
    
    Normalized Temperature-scaled Cross Entropy Loss
    用于对比学习，最大化正样本对的相似度，最小化负样本对的相似度。
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_cosine_similarity: bool = True
    ):
        """
        初始化NT-Xent损失
        
        Args:
            temperature: 温度参数，控制分布的平滑程度
            use_cosine_similarity: 是否使用余弦相似度
        """
        super(NTXentLoss, self).__init__()
        
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity
        
        logger.info(f"NTXentLoss initialized: temperature={temperature}")
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算NT-Xent损失
        
        Args:
            z_i: 第一个视图的表征 [batch_size, seq_len, dim]
            z_j: 第二个视图的表征 [batch_size, seq_len, dim]
            mask: 可选的掩码 [batch_size, seq_len]
            
        Returns:
            损失值
        """
        batch_size, seq_len, dim = z_i.shape
        
        # 展平序列维度
        z_i = z_i.reshape(-1, dim)  # [batch_size * seq_len, dim]
        z_j = z_j.reshape(-1, dim)  # [batch_size * seq_len, dim]
        
        # 如果有掩码，只计算有效位置的损失
        if mask is not None:
            mask = mask.reshape(-1)  # [batch_size * seq_len]
            z_i = z_i[mask]
            z_j = z_j[mask]
        
        N = z_i.shape[0]
        
        # 归一化（用于余弦相似度）
        if self.use_cosine_similarity:
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)
        
        # 拼接两个视图
        z = torch.cat([z_i, z_j], dim=0)  # [2N, dim]
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(z, z.t())  # [2N, 2N]
        
        # 创建正样本掩码
        # 对于每个样本i，其正样本是i+N（另一个视图中的对应位置）
        positive_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=z.device)
        for i in range(N):
            positive_mask[i, i + N] = True
            positive_mask[i + N, i] = True
        
        # 创建负样本掩码（排除自己和正样本）
        negative_mask = ~positive_mask
        for i in range(2 * N):
            negative_mask[i, i] = False
        
        # 应用温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        # 计算损失
        # 对于每个样本，计算其与正样本的相似度和与所有负样本的相似度
        losses = []
        for i in range(2 * N):
            # 正样本相似度
            positive_sim = similarity_matrix[i][positive_mask[i]]
            
            # 负样本相似度
            negative_sim = similarity_matrix[i][negative_mask[i]]
            
            # 计算log-sum-exp
            # loss = -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            #      = -pos + log(exp(pos) + sum(exp(neg)))
            #      = -pos + log_sum_exp([pos, neg])
            all_sim = torch.cat([positive_sim, negative_sim])
            loss = -positive_sim + torch.logsumexp(all_sim, dim=0)
            losses.append(loss)
        
        # 平均损失
        loss = torch.stack(losses).mean()
        
        return loss


class HierarchicalContrastiveLoss(nn.Module):
    """
    层次化对比损失
    
    在不同时间尺度上计算对比损失。
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        temporal_unit: int = 0
    ):
        """
        初始化层次化对比损失
        
        Args:
            temperature: 温度参数
            temporal_unit: 时间单元大小（0表示实例级别）
        """
        super(HierarchicalContrastiveLoss, self).__init__()
        
        self.temperature = temperature
        self.temporal_unit = temporal_unit
        self.ntxent = NTXentLoss(temperature=temperature)
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        计算层次化对比损失
        
        Args:
            z_i: 第一个视图的表征 [batch_size, seq_len, dim]
            z_j: 第二个视图的表征 [batch_size, seq_len, dim]
            
        Returns:
            损失值
        """
        if self.temporal_unit == 0:
            # 实例级别对比
            return self.ntxent(z_i, z_j)
        else:
            # 时间单元级别对比
            batch_size, seq_len, dim = z_i.shape
            
            # 将序列分割成时间单元
            num_units = seq_len // self.temporal_unit
            
            # 重塑为时间单元
            z_i_units = z_i[:, :num_units * self.temporal_unit, :].reshape(
                batch_size, num_units, self.temporal_unit, dim
            )
            z_j_units = z_j[:, :num_units * self.temporal_unit, :].reshape(
                batch_size, num_units, self.temporal_unit, dim
            )
            
            # 对每个时间单元取平均
            z_i_units = z_i_units.mean(dim=2)  # [batch_size, num_units, dim]
            z_j_units = z_j_units.mean(dim=2)  # [batch_size, num_units, dim]
            
            # 计算对比损失
            return self.ntxent(z_i_units, z_j_units)


class TS2VecLoss(nn.Module):
    """
    TS2Vec完整损失函数
    
    结合实例级别和时间级别的对比损失。
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        temporal_unit: int = 0,
        alpha: float = 0.5
    ):
        """
        初始化TS2Vec损失
        
        Args:
            temperature: 温度参数
            temporal_unit: 时间单元大小
            alpha: 实例级别损失的权重
        """
        super(TS2VecLoss, self).__init__()
        
        self.instance_loss = NTXentLoss(temperature=temperature)
        self.temporal_loss = HierarchicalContrastiveLoss(
            temperature=temperature,
            temporal_unit=temporal_unit
        ) if temporal_unit > 0 else None
        self.alpha = alpha
        
        logger.info(f"TS2VecLoss initialized: "
                   f"temperature={temperature}, "
                   f"temporal_unit={temporal_unit}, "
                   f"alpha={alpha}")
    
    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            z_i: 第一个视图的表征 [batch_size, seq_len, dim]
            z_j: 第二个视图的表征 [batch_size, seq_len, dim]
            
        Returns:
            损失值
        """
        # 实例级别损失
        loss_instance = self.instance_loss(z_i, z_j)
        
        if self.temporal_loss is not None:
            # 时间级别损失
            loss_temporal = self.temporal_loss(z_i, z_j)
            
            # 组合损失
            total_loss = self.alpha * loss_instance + (1 - self.alpha) * loss_temporal
        else:
            total_loss = loss_instance
        
        return total_loss


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建测试数据
    batch_size = 4
    seq_len = 256
    dim = 128
    
    z_i = torch.randn(batch_size, seq_len, dim)
    z_j = torch.randn(batch_size, seq_len, dim)
    
    print("\n=== 对比学习损失函数示例 ===")
    print(f"输入形状: z_i={z_i.shape}, z_j={z_j.shape}")
    
    # 测试NT-Xent损失
    print("\n1. NT-Xent损失:")
    ntxent_loss = NTXentLoss(temperature=0.07)
    loss = ntxent_loss(z_i, z_j)
    print(f"   损失值: {loss.item():.4f}")
    
    # 测试层次化对比损失
    print("\n2. 层次化对比损失:")
    hierarchical_loss = HierarchicalContrastiveLoss(
        temperature=0.07,
        temporal_unit=16
    )
    loss = hierarchical_loss(z_i, z_j)
    print(f"   损失值: {loss.item():.4f}")
    
    # 测试TS2Vec完整损失
    print("\n3. TS2Vec完整损失:")
    ts2vec_loss = TS2VecLoss(
        temperature=0.07,
        temporal_unit=16,
        alpha=0.5
    )
    loss = ts2vec_loss(z_i, z_j)
    print(f"   损失值: {loss.item():.4f}")
    
    # 测试梯度
    print("\n4. 梯度测试:")
    z_i.requires_grad = True
    z_j.requires_grad = True
    loss = ts2vec_loss(z_i, z_j)
    loss.backward()
    print(f"   z_i梯度范数: {z_i.grad.norm().item():.4f}")
    print(f"   z_j梯度范数: {z_j.grad.norm().item():.4f}")
    
    # 测试带掩码的损失
    print("\n5. 带掩码的损失:")
    mask = torch.rand(batch_size, seq_len) > 0.2  # 80%的位置有效
    ntxent_loss_masked = NTXentLoss(temperature=0.07)
    loss = ntxent_loss_masked(z_i, z_j, mask=mask)
    print(f"   损失值: {loss.item():.4f}")
    print(f"   有效位置比例: {mask.float().mean():.2%}")