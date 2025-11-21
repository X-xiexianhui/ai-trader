"""
Transformer模型实现

任务3.2.1-3.2.5实现:
1. 正弦位置编码
2. 多头自注意力层
3. 前馈网络层
4. Transformer编码器层
5. 完整Transformer模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    任务3.2.1: 正弦位置编码
    
    为序列添加位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 应用正弦和余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # 注册为buffer(不参与训练)
        self.register_buffer('pe', pe)
        
        logger.info(f"位置编码初始化: d_model={d_model}, max_len={max_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiHeadAttention(nn.Module):
    """
    任务3.2.2: 多头自注意力层
    
    实现多头注意力机制
    """
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"多头注意力初始化: d_model={d_model}, nhead={nhead}")
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            mask: 注意力掩码(因果掩码) [seq_len, seq_len]
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性投影并分割为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, nhead, seq_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, nhead, seq_len, seq_len]
        
        # 应用因果掩码(防止看到未来)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        # context: [batch, nhead, seq_len, d_k]
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    任务3.2.3: 前馈网络层
    
    Position-wise Feed-Forward Network
    """
    
    def __init__(self,
                 d_model: int = 256,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            dim_feedforward: 前馈网络隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"前馈网络初始化: {d_model}→{dim_feedforward}→{d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    任务3.2.4: Transformer编码器层
    
    单个Transformer编码器层
    """
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            nhead: 注意力头数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
        """
        super().__init__()
        
        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info("Transformer编码器层初始化")
    
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        # 多头自注意力子层 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerStateModel(nn.Module):
    """
    任务3.2.5: 完整Transformer模型
    
    用于状态建模的完整Transformer
    """
    
    def __init__(self,
                 input_dim: int = 155,  # 128 (TS2Vec) + 27 (手工特征)
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        """
        初始化Transformer模型
        
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Transformer模型初始化: {num_layers}层, d_model={d_model}")
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码(防止看到未来)
        
        Args:
            seq_len: 序列长度
            device: 设备
            
        Returns:
            因果掩码 [seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask == 0  # 转换为布尔掩码
        return mask
    
    def forward(self,
                x: torch.Tensor,
                use_causal_mask: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            use_causal_mask: 是否使用因果掩码
            
        Returns:
            状态向量 [batch, d_model] 或 [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入
        x = self.input_embedding(x)  # [batch, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 生成因果掩码
        mask = None
        if use_causal_mask:
            mask = self._generate_causal_mask(seq_len, x.device)
        
        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 输出: [batch, seq_len, d_model]
        return x
    
    def get_state_vector(self,
                        x: torch.Tensor,
                        pooling: str = 'last') -> torch.Tensor:
        """
        获取状态向量
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            pooling: 池化方法 ('last', 'mean', 'max')
            
        Returns:
            状态向量 [batch, d_model]
        """
        # 前向传播
        output = self.forward(x)  # [batch, seq_len, d_model]
        
        # 池化
        if pooling == 'last':
            state = output[:, -1, :]  # 取最后一个时间步
        elif pooling == 'mean':
            state = output.mean(dim=1)  # 平均池化
        elif pooling == 'max':
            state = output.max(dim=1)[0]  # 最大池化
        else:
            raise ValueError(f"不支持的池化方法: {pooling}")
        
        return state
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
            }
        }, filepath)
        logger.info(f"Transformer模型已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'TransformerStateModel':
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 合并配置
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        # 创建模型
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Transformer模型已加载: {filepath}")
        
        return model