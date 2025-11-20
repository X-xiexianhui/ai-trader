"""
TS2Vec模型架构实现

实现TS2Vec (Time Series to Vector) 模型的核心架构:
- Dilated CNN编码器 (10层)
- 投影头 (256→128→128)
- 对比学习头

参考论文: TS2Vec: Towards Universal Representation of Time Series
https://arxiv.org/abs/2106.10466

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DilatedConvBlock(nn.Module):
    """
    扩张卷积块
    
    使用扩张卷积来增大感受野，同时保持参数效率。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        """
        初始化扩张卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 扩张率
            dropout: Dropout比率
        """
        super(DilatedConvBlock, self).__init__()
        
        # 计算padding以保持序列长度
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接（如果维度匹配）
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, in_channels, seq_len]
            
        Returns:
            输出张量 [batch_size, out_channels, seq_len]
        """
        identity = x
        
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.residual is not None:
            identity = self.residual(identity)
        
        out = out + identity
        
        return out


class DilatedCNNEncoder(nn.Module):
    """
    扩张CNN编码器
    
    使用10层扩张卷积构建深度编码器，逐步增大感受野。
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 64,
        output_dim: int = 256,
        num_layers: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        初始化扩张CNN编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出embedding维度
            num_layers: 卷积层数
            kernel_size: 卷积核大小
            dropout: Dropout比率
        """
        super(DilatedCNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 构建扩张卷积层
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # 指数增长的扩张率: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
            dilation = 2 ** i
            
            self.conv_layers.append(
                DilatedConvBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # 输出投影层
        self.output_projection = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        
        logger.info(f"DilatedCNNEncoder initialized: "
                   f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"output_dim={output_dim}, num_layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            编码后的张量 [batch_size, seq_len, output_dim]
        """
        # 转换维度: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 通过扩张卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 输出投影
        x = self.output_projection(x)
        
        # 转换回: [batch, features, seq_len] -> [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        return x


class ProjectionHead(nn.Module):
    """
    投影头
    
    将编码器输出投影到对比学习空间 (256→128→128)
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 128
    ):
        """
        初始化投影头
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(ProjectionHead, self).__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            投影后的张量 [batch_size, seq_len, output_dim]
        """
        return self.projection(x)


class TS2Vec(nn.Module):
    """
    TS2Vec模型
    
    完整的TS2Vec模型，包括编码器和投影头。
    """
    
    def __init__(
        self,
        input_dim: int = 27,
        encoder_hidden_dim: int = 64,
        encoder_output_dim: int = 256,
        projection_hidden_dim: int = 128,
        projection_output_dim: int = 128,
        num_layers: int = 10,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        初始化TS2Vec模型
        
        Args:
            input_dim: 输入特征维度 (27维手工特征)
            encoder_hidden_dim: 编码器隐藏层维度
            encoder_output_dim: 编码器输出维度
            projection_hidden_dim: 投影头隐藏层维度
            projection_output_dim: 投影头输出维度
            num_layers: 编码器层数
            kernel_size: 卷积核大小
            dropout: Dropout比率
        """
        super(TS2Vec, self).__init__()
        
        self.input_dim = input_dim
        self.encoder_output_dim = encoder_output_dim
        self.projection_output_dim = projection_output_dim
        
        # 编码器
        self.encoder = DilatedCNNEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 投影头（用于对比学习）
        self.projection_head = ProjectionHead(
            input_dim=encoder_output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim
        )
        
        logger.info(f"TS2Vec model initialized with {self.count_parameters()} parameters")
    
    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            return_projection: 是否返回投影
            
        Returns:
            (编码, 投影) 或 仅编码
        """
        # 编码
        encoding = self.encoder(x)
        
        if return_projection:
            # 投影（用于对比学习）
            projection = self.projection_head(encoding)
            return encoding, projection
        else:
            return encoding, None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅编码，不投影（用于提取embedding）
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            编码张量 [batch_size, seq_len, encoder_output_dim]
        """
        encoding, _ = self.forward(x, return_projection=False)
        return encoding
    
    def count_parameters(self) -> int:
        """
        计算模型参数数量
        
        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embedding_dim(self) -> int:
        """
        获取embedding维度
        
        Returns:
            Embedding维度
        """
        return self.encoder_output_dim


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建模型
    model = TS2Vec(
        input_dim=27,
        encoder_hidden_dim=64,
        encoder_output_dim=256,
        projection_hidden_dim=128,
        projection_output_dim=128,
        num_layers=10,
        kernel_size=3,
        dropout=0.1
    )
    
    print("\n=== TS2Vec模型架构 ===")
    print(f"模型参数数量: {model.count_parameters():,}")
    print(f"Embedding维度: {model.get_embedding_dim()}")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 256
    input_dim = 27
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    encoding, projection = model(x, return_projection=True)
    
    print(f"编码形状: {encoding.shape}")
    print(f"投影形状: {projection.shape}")
    
    # 测试仅编码
    embedding = model.encode(x)
    print(f"Embedding形状: {embedding.shape}")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)