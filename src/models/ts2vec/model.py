"""
TS2Vec模型实现

任务2.2.1-2.2.4实现:
1. 膨胀卷积编码器
2. 投影头网络
3. NT-Xent对比损失函数
4. TS2Vec完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DilatedConvEncoder(nn.Module):
    """
    任务2.2.1: 膨胀卷积编码器
    
    TS2Vec的核心组件,使用膨胀卷积捕获多尺度时间模式
    """
    
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 num_layers: int = 10,
                 kernel_size: int = 3,
                 dilation_rates: Optional[List[int]] = None):
        """
        初始化膨胀卷积编码器
        
        Args:
            input_dim: 输入特征维度(OHLC=4)
            hidden_dim: 隐藏层维度
            num_layers: 卷积层数量
            kernel_size: 卷积核大小
            dilation_rates: 膨胀率列表
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 默认膨胀率
        if dilation_rates is None:
            dilation_rates = [2**i for i in range(num_layers)]
        
        self.dilation_rates = dilation_rates
        
        # 输入投影层
        self.input_projection = nn.Conv1d(
            input_dim, hidden_dim, kernel_size=1
        )
        
        # 膨胀卷积层
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = dilation_rates[i]
            
            # 膨胀卷积
            conv = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2  # 保持序列长度
            )
            
            # 层归一化
            norm = nn.LayerNorm(hidden_dim)
            
            self.conv_layers.append(conv)
            self.norm_layers.append(norm)
        
        logger.info(f"膨胀卷积编码器初始化: {num_layers}层, 隐藏维度{hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, input_dim]
            
        Returns:
            编码特征 [batch, seq_len, hidden_dim]
        """
        # 转换为卷积格式 [batch, channels, seq_len]
        x = x.transpose(1, 2)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 膨胀卷积层
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            # 残差连接
            residual = x
            
            # 卷积 + ReLU
            x = F.relu(conv(x))
            
            # 转换为 [batch, seq_len, channels] 进行LayerNorm
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            
            # 残差连接
            x = x + residual
        
        # 转换回 [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        return x


class ProjectionHead(nn.Module):
    """
    任务2.2.2: 投影头网络
    
    将编码特征映射到对比学习空间
    """
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 128):
        """
        初始化投影头
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"投影头初始化: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch, seq_len, input_dim]
            
        Returns:
            投影后的embedding [batch, seq_len, output_dim]
        """
        # FC1 + ReLU
        x = F.relu(self.fc1(x))
        
        # FC2
        x = self.fc2(x)
        
        # L2归一化
        x = F.normalize(x, p=2, dim=-1)
        
        return x


class NTXentLoss(nn.Module):
    """
    任务2.2.3: NT-Xent对比损失函数
    
    归一化温度交叉熵损失
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        初始化NT-Xent损失
        
        Args:
            temperature: 温度参数τ
        """
        super().__init__()
        self.temperature = temperature
        
        logger.info(f"NT-Xent损失初始化: 温度={temperature}")
    
    def forward(self,
                z_i: torch.Tensor,
                z_j: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            z_i: 第一个视图的embedding [batch, seq_len, dim]
            z_j: 第二个视图的embedding [batch, seq_len, dim]
            
        Returns:
            对比损失标量
        """
        batch_size, seq_len, dim = z_i.shape
        
        # 展平为 [batch*seq_len, dim]
        z_i = z_i.reshape(-1, dim)
        z_j = z_j.reshape(-1, dim)
        
        # 拼接正样本对
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch*seq_len, dim]
        
        # 计算余弦相似度矩阵
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # 创建正样本mask
        N = z_i.shape[0]
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        
        # 正样本对的索引
        pos_mask = torch.zeros(2 * N, 2 * N, dtype=torch.bool, device=z.device)
        pos_mask[:N, N:] = torch.eye(N, dtype=torch.bool, device=z.device)
        pos_mask[N:, :N] = torch.eye(N, dtype=torch.bool, device=z.device)
        
        # 移除对角线(自己与自己的相似度)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # 计算损失
        # 对于每个样本,正样本的相似度 / 所有样本的相似度之和
        exp_sim = torch.exp(sim_matrix)
        
        # 正样本的相似度
        pos_sim = exp_sim.masked_select(pos_mask).view(2 * N, -1)
        
        # 所有样本的相似度之和
        neg_sim = exp_sim.sum(dim=1, keepdim=True)
        
        # 对比损失
        loss = -torch.log(pos_sim / neg_sim).mean()
        
        return loss


class TS2VecModel(nn.Module):
    """
    任务2.2.4: TS2Vec完整模型
    
    整合编码器、投影头和损失函数
    """
    
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_layers: int = 10,
                 kernel_size: int = 3,
                 dilation_rates: Optional[List[int]] = None,
                 temperature: float = 0.1):
        """
        初始化TS2Vec模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 编码器隐藏维度
            output_dim: 投影头输出维度
            num_layers: 编码器层数
            kernel_size: 卷积核大小
            dilation_rates: 膨胀率列表
            temperature: 对比学习温度
        """
        super().__init__()
        
        # 膨胀卷积编码器
        self.encoder = DilatedConvEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_rates=dilation_rates
        )
        
        # 投影头
        self.projection_head = ProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=output_dim,
            output_dim=output_dim
        )
        
        # 对比损失
        self.criterion = NTXentLoss(temperature=temperature)
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        logger.info("TS2Vec模型初始化完成")
    
    def forward(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                return_loss: bool = True):
        """
        前向传播
        
        Args:
            x_i: 第一个增强视图 [batch, seq_len, input_dim]
            x_j: 第二个增强视图 [batch, seq_len, input_dim]
            return_loss: 是否返回损失
            
        Returns:
            如果return_loss=True: 损失值
            否则: (z_i, z_j) embedding对
        """
        # 编码
        h_i = self.encoder(x_i)  # [batch, seq_len, hidden_dim]
        h_j = self.encoder(x_j)
        
        # 投影
        z_i = self.projection_head(h_i)  # [batch, seq_len, output_dim]
        z_j = self.projection_head(h_j)
        
        if return_loss:
            # 计算对比损失
            loss = self.criterion(z_i, z_j)
            return loss
        else:
            return z_i, z_j
    
    def encode(self, x: torch.Tensor, return_projection: bool = False) -> torch.Tensor:
        """
        推理模式:生成embedding
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            return_projection: 是否返回投影后的embedding
            
        Returns:
            embedding [batch, seq_len, dim]
        """
        with torch.no_grad():
            # 编码
            h = self.encoder(x)
            
            if return_projection:
                # 投影
                z = self.projection_head(h)
                return z
            else:
                return h
    
    def get_embedding_dim(self, use_projection: bool = False) -> int:
        """获取embedding维度"""
        return self.output_dim if use_projection else self.hidden_dim
    
    def save(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
            }
        }, filepath)
        logger.info(f"TS2Vec模型已保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'TS2VecModel':
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # 合并配置
        config = checkpoint.get('config', {})
        config.update(kwargs)
        
        # 创建模型
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"TS2Vec模型已加载: {filepath}")
        
        return model