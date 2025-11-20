"""
Transformer模型实现

实现用于时间序列预测的Transformer编码器，包括:
- 多头自注意力机制
- 前馈网络
- 位置编码
- 特征融合层
- 辅助任务头

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    使用sin和cos函数为序列中的每个位置生成唯一的编码。
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 注册为buffer，不参与梯度更新
        self.register_buffer('pe', pe)
        
        logger.info(f"PositionalEncoding initialized: d_model={d_model}, max_len={max_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    实现scaled dot-product attention with multiple heads。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"MultiHeadAttention initialized: d_model={d_model}, num_heads={num_heads}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: Query张量 (batch, seq_len, d_model)
            key: Key张量 (batch, seq_len, d_model)
            value: Value张量 (batch, seq_len, d_model)
            mask: 注意力掩码 (batch, seq_len, seq_len) 或 (seq_len, seq_len)
            
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            attention_weights: 注意力权重 (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # 线性投影并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = self.W_o(context)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """
    前馈网络
    
    两层全连接网络with GELU activation。
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"FeedForward initialized: d_model={d_model}, d_ff={d_ff}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            
        Returns:
            输出张量 (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    包含多头注意力和前馈网络，with residual connections and layer normalization。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈网络隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            mask: 注意力掩码
            
        Returns:
            output: 输出张量 (batch, seq_len, d_model)
            attention_weights: 注意力权重
        """
        # 多头自注意力 + 残差连接 + LayerNorm
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x, attention_weights


class FeatureFusion(nn.Module):
    """
    特征融合层
    
    融合TS2Vec embedding和手工特征。
    """
    
    def __init__(
        self,
        ts2vec_dim: int = 128,
        manual_dim: int = 27,
        d_model: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化特征融合层
        
        Args:
            ts2vec_dim: TS2Vec embedding维度
            manual_dim: 手工特征维度
            d_model: 输出模型维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.ts2vec_dim = ts2vec_dim
        self.manual_dim = manual_dim
        self.input_dim = ts2vec_dim + manual_dim
        
        # 输入嵌入层
        self.input_projection = nn.Linear(self.input_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"FeatureFusion initialized: "
                   f"ts2vec_dim={ts2vec_dim}, manual_dim={manual_dim}, "
                   f"d_model={d_model}")
    
    def forward(
        self,
        ts2vec_emb: torch.Tensor,
        manual_features: torch.Tensor
    ) -> torch.Tensor:
        """
        融合特征
        
        Args:
            ts2vec_emb: TS2Vec embedding (batch, seq_len, ts2vec_dim)
            manual_features: 手工特征 (batch, seq_len, manual_dim)
            
        Returns:
            融合后的特征 (batch, seq_len, d_model)
        """
        # 拼接特征
        x = torch.cat([ts2vec_emb, manual_features], dim=-1)
        
        # 投影到模型维度
        x = self.input_projection(x)
        x = self.dropout(x)
        
        return x


class AuxiliaryHeads(nn.Module):
    """
    辅助任务头
    
    包含回归头（预测收益率）和分类头（预测方向）。
    """
    
    def __init__(
        self,
        d_model: int = 256,
        dropout: float = 0.1
    ):
        """
        初始化辅助任务头
        
        Args:
            d_model: 输入维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 回归头：预测未来收益率
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 分类头：预测价格方向（上涨/下跌/持平）
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3类：上涨、下跌、持平
        )
        
        logger.info(f"AuxiliaryHeads initialized: d_model={d_model}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, seq_len, d_model)
            
        Returns:
            regression_output: 回归预测 (batch, seq_len, 1)
            classification_output: 分类预测 (batch, seq_len, 3)
        """
        regression_output = self.regression_head(x)
        classification_output = self.classification_head(x)
        
        return regression_output, classification_output


class TransformerModel(nn.Module):
    """
    完整的Transformer模型
    
    用于时间序列状态表征学习，支持监督学习预训练。
    """
    
    def __init__(
        self,
        ts2vec_dim: int = 128,
        manual_dim: int = 27,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_auxiliary: bool = True
    ):
        """
        初始化Transformer模型
        
        Args:
            ts2vec_dim: TS2Vec embedding维度
            manual_dim: 手工特征维度
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络隐藏层维度
            max_seq_len: 最大序列长度
            dropout: Dropout率
            use_auxiliary: 是否使用辅助任务
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_auxiliary = use_auxiliary
        
        # 特征融合层
        self.feature_fusion = FeatureFusion(
            ts2vec_dim, manual_dim, d_model, dropout
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 辅助任务头
        if use_auxiliary:
            self.auxiliary_heads = AuxiliaryHeads(d_model, dropout)
        
        self._init_parameters()
        
        logger.info(f"TransformerModel initialized: "
                   f"d_model={d_model}, num_heads={num_heads}, "
                   f"num_layers={num_layers}")
    
    def _init_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（上三角掩码）
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            掩码张量 (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask == 0  # 转换为bool，True表示可见
        return mask
    
    def forward(
        self,
        ts2vec_emb: torch.Tensor,
        manual_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            ts2vec_emb: TS2Vec embedding (batch, seq_len, ts2vec_dim)
            manual_features: 手工特征 (batch, seq_len, manual_dim)
            mask: 注意力掩码（可选）
            return_attention: 是否返回注意力权重
            
        Returns:
            state_vector: 状态向量 (batch, seq_len, d_model)
            regression_output: 回归预测（如果use_auxiliary=True）
            classification_output: 分类预测（如果use_auxiliary=True）
        """
        # 特征融合
        x = self.feature_fusion(ts2vec_emb, manual_features)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 创建因果掩码（如果未提供）
        if mask is None:
            seq_len = x.size(1)
            mask = self.create_causal_mask(seq_len, x.device)
        
        # 通过编码器层
        attention_weights_list = []
        for layer in self.encoder_layers:
            x, attention_weights = layer(x, mask)
            if return_attention:
                attention_weights_list.append(attention_weights)
        
        # 状态向量
        state_vector = x
        
        # 辅助任务
        regression_output = None
        classification_output = None
        if self.use_auxiliary:
            regression_output, classification_output = self.auxiliary_heads(x)
        
        if return_attention:
            return state_vector, regression_output, classification_output, attention_weights_list
        
        return state_vector, regression_output, classification_output
    
    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Transformer模型示例 ===")
    
    # 创建模型
    model = TransformerModel(
        ts2vec_dim=128,
        manual_dim=27,
        d_model=256,
        num_heads=8,
        num_layers=6,
        use_auxiliary=True
    )
    
    print(f"\n模型参数量: {model.count_parameters():,}")
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 64
    ts2vec_emb = torch.randn(batch_size, seq_len, 128)
    manual_features = torch.randn(batch_size, seq_len, 27)
    
    # 前向传播
    print("\n执行前向传播...")
    state_vector, reg_out, cls_out = model(ts2vec_emb, manual_features)
    
    print(f"状态向量形状: {state_vector.shape}")
    print(f"回归输出形状: {reg_out.shape}")
    print(f"分类输出形状: {cls_out.shape}")
    
    print("\n示例完成!")