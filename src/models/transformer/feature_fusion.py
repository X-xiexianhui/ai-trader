"""
Transformer特征融合模块

任务3.1.1-3.1.3实现:
1. TS2Vec embedding生成器
2. 特征融合模块
3. 时序窗口序列构建器
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TS2VecEmbeddingGenerator:
    """
    任务3.1.1: TS2Vec embedding生成器
    
    使用预训练的TS2Vec模型批量生成embeddings
    """
    
    def __init__(self,
                 ts2vec_model,
                 device: str = 'cpu',
                 use_projection: bool = False):
        """
        初始化embedding生成器
        
        Args:
            ts2vec_model: 预训练的TS2Vec模型
            device: 计算设备
            use_projection: 是否使用投影头的输出
        """
        self.model = ts2vec_model
        self.device = device
        self.use_projection = use_projection
        
        self.model.to(device)
        self.model.eval()
        
        self.embedding_dim = self.model.get_embedding_dim(use_projection)
        
        logger.info(f"TS2Vec Embedding生成器初始化: dim={self.embedding_dim}")
    
    def generate(self,
                data: torch.Tensor,
                batch_size: int = 32) -> torch.Tensor:
        """
        批量生成embeddings
        
        Args:
            data: 输入数据 [N, seq_len, input_dim]
            batch_size: 批次大小
            
        Returns:
            embeddings [N, seq_len, embedding_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size].to(self.device)
                
                # 生成embedding
                emb = self.model.encode(batch, return_projection=self.use_projection)
                embeddings.append(emb.cpu())
        
        embeddings = torch.cat(embeddings, dim=0)
        
        logger.info(f"生成embeddings: {embeddings.shape}")
        
        return embeddings
    
    def generate_from_dataframe(self,
                                df: pd.DataFrame,
                                window_length: int = 256,
                                stride: int = 1,
                                columns: Optional[list] = None) -> torch.Tensor:
        """
        从DataFrame生成embeddings
        
        Args:
            df: 输入DataFrame
            window_length: 窗口长度
            stride: 滑动步长
            columns: 要使用的列
            
        Returns:
            embeddings
        """
        from ..ts2vec.data_preparation import SlidingWindowGenerator
        
        # 生成窗口
        window_gen = SlidingWindowGenerator(window_length, stride)
        windows = window_gen.generate_from_dataframe(df, columns)
        
        # 转换为tensor
        windows_tensor = torch.FloatTensor(windows)
        
        # 生成embeddings
        embeddings = self.generate(windows_tensor)
        
        return embeddings


class FeatureFusion:
    """
    任务3.1.2: 特征融合模块
    
    融合TS2Vec embeddings和手工特征
    """
    
    def __init__(self,
                 embedding_dim: int = 128,
                 feature_dim: int = 27,
                 fusion_method: str = 'concat'):
        """
        初始化特征融合模块
        
        Args:
            embedding_dim: TS2Vec embedding维度
            feature_dim: 手工特征维度
            fusion_method: 融合方法 ('concat' 或 'weighted')
        """
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.output_dim = embedding_dim + feature_dim
        elif fusion_method == 'weighted':
            self.output_dim = embedding_dim + feature_dim
            # 可学习的权重(如果需要)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
        
        logger.info(f"特征融合初始化: {embedding_dim}+{feature_dim}→{self.output_dim}")
    
    def fuse(self,
             embeddings: torch.Tensor,
             features: torch.Tensor) -> torch.Tensor:
        """
        融合embeddings和特征
        
        Args:
            embeddings: TS2Vec embeddings [N, embedding_dim]
            features: 手工特征 [N, feature_dim]
            
        Returns:
            融合特征 [N, output_dim]
        """
        if self.fusion_method == 'concat':
            # 简单拼接
            fused = torch.cat([embeddings, features], dim=-1)
        
        elif self.fusion_method == 'weighted':
            # 加权融合(这里简化为拼接,实际可以更复杂)
            fused = torch.cat([
                embeddings * self.alpha,
                features * (1 - self.alpha)
            ], dim=-1)
        
        return fused
    
    def fuse_sequences(self,
                      embedding_seq: torch.Tensor,
                      feature_seq: torch.Tensor) -> torch.Tensor:
        """
        融合序列数据
        
        Args:
            embedding_seq: [N, seq_len, embedding_dim]
            feature_seq: [N, seq_len, feature_dim]
            
        Returns:
            融合序列 [N, seq_len, output_dim]
        """
        # 确保时间对齐
        assert embedding_seq.shape[:2] == feature_seq.shape[:2], \
            "Embedding和特征的序列长度必须一致"
        
        # 逐时间步融合
        batch_size, seq_len = embedding_seq.shape[:2]
        
        fused_seq = []
        for t in range(seq_len):
            fused_t = self.fuse(embedding_seq[:, t, :], feature_seq[:, t, :])
            fused_seq.append(fused_t)
        
        fused_seq = torch.stack(fused_seq, dim=1)
        
        return fused_seq


class SequenceBuilder:
    """
    任务3.1.3: 时序窗口序列构建器
    
    为Transformer构建输入序列
    """
    
    def __init__(self,
                 sequence_length: int = 64,
                 stride: int = 1):
        """
        初始化序列构建器
        
        Args:
            sequence_length: 序列长度T
            stride: 滑动步长
        """
        self.sequence_length = sequence_length
        self.stride = stride
        
        logger.info(f"序列构建器初始化: length={sequence_length}, stride={stride}")
    
    def build_sequences(self,
                       features: torch.Tensor,
                       labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        构建序列
        
        Args:
            features: 融合特征 [N, feature_dim]
            labels: 标签(可选) [N]
            
        Returns:
            sequences: [M, seq_len, feature_dim]
            sequence_labels: [M] (如果提供了labels)
        """
        N, feature_dim = features.shape
        
        if N < self.sequence_length:
            raise ValueError(f"数据长度{N}小于序列长度{self.sequence_length}")
        
        # 计算序列数量
        num_sequences = (N - self.sequence_length) // self.stride + 1
        
        sequences = []
        sequence_labels = [] if labels is not None else None
        
        for i in range(num_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.sequence_length
            
            # 提取序列
            seq = features[start_idx:end_idx]
            sequences.append(seq)
            
            # 提取标签(使用序列最后一个时间步的标签)
            if labels is not None:
                label = labels[end_idx - 1]
                sequence_labels.append(label)
        
        sequences = torch.stack(sequences)
        
        if sequence_labels is not None:
            sequence_labels = torch.stack(sequence_labels)
        
        logger.info(f"构建序列: {num_sequences}个序列, 形状{sequences.shape}")
        
        return sequences, sequence_labels
    
    def build_from_dataframe(self,
                            df: pd.DataFrame,
                            feature_columns: list,
                            label_column: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        从DataFrame构建序列
        
        Args:
            df: 输入DataFrame
            feature_columns: 特征列名
            label_column: 标签列名(可选)
            
        Returns:
            sequences, labels
        """
        # 提取特征
        features = torch.FloatTensor(df[feature_columns].values)
        
        # 提取标签
        labels = None
        if label_column is not None:
            labels = torch.FloatTensor(df[label_column].values)
        
        return self.build_sequences(features, labels)


class TransformerDataPipeline:
    """
    Transformer完整数据管道
    
    整合TS2Vec embedding生成、特征融合和序列构建
    """
    
    def __init__(self,
                 ts2vec_model,
                 embedding_dim: int = 128,
                 feature_dim: int = 27,
                 sequence_length: int = 64,
                 device: str = 'cpu'):
        """
        初始化数据管道
        
        Args:
            ts2vec_model: 预训练的TS2Vec模型
            embedding_dim: Embedding维度
            feature_dim: 手工特征维度
            sequence_length: 序列长度
            device: 计算设备
        """
        # TS2Vec embedding生成器
        self.embedding_generator = TS2VecEmbeddingGenerator(
            ts2vec_model, device, use_projection=True
        )
        
        # 特征融合
        self.feature_fusion = FeatureFusion(
            embedding_dim, feature_dim, fusion_method='concat'
        )
        
        # 序列构建器
        self.sequence_builder = SequenceBuilder(sequence_length)
        
        self.output_dim = self.feature_fusion.output_dim
        
        logger.info("Transformer数据管道初始化完成")
    
    def process(self,
                ohlc_data: pd.DataFrame,
                hand_features: pd.DataFrame,
                label_column: Optional[str] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        完整的数据处理流程
        
        Args:
            ohlc_data: OHLC数据
            hand_features: 手工特征
            label_column: 标签列名
            
        Returns:
            sequences, labels
        """
        # 1. 生成TS2Vec embeddings
        embeddings = self.embedding_generator.generate_from_dataframe(
            ohlc_data, columns=['Open', 'High', 'Low', 'Close']
        )
        
        # 2. 对齐时间(embeddings可能比原始数据短)
        # 简化处理:假设已对齐,实际需要更仔细的对齐逻辑
        min_len = min(len(embeddings), len(hand_features))
        embeddings = embeddings[:min_len]
        hand_features_aligned = hand_features.iloc[:min_len]
        
        # 3. 特征融合
        hand_features_tensor = torch.FloatTensor(hand_features_aligned.values)
        
        # 如果embeddings是3D(有seq_len维度),需要平均池化
        if len(embeddings.shape) == 3:
            embeddings = embeddings.mean(dim=1)  # [N, embedding_dim]
        
        fused_features = self.feature_fusion.fuse(embeddings, hand_features_tensor)
        
        # 4. 构建序列
        labels = None
        if label_column is not None:
            labels = torch.FloatTensor(hand_features_aligned[label_column].values)
        
        sequences, sequence_labels = self.sequence_builder.build_sequences(
            fused_features, labels
        )
        
        return sequences, sequence_labels