"""
任务7.1.2: 推理数据管道

实现实时推理的数据管道，用于在线预测
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Optional, Deque
from collections import deque
import logging
from pathlib import Path

from ..features.data_cleaner import DataCleaner
from ..features.feature_calculator import FeatureCalculator
from ..features.feature_scaler import FeatureScaler
from ..models.transformer.feature_fusion import (
    TS2VecEmbeddingGenerator,
    FeatureFusion
)

logger = logging.getLogger(__name__)


class InferenceDataPipeline:
    """
    推理数据管道
    
    实时处理流程：
    1. 接收新K线数据
    2. 计算特征
    3. 归一化（使用训练scaler）
    4. 生成embedding
    5. 维护滑动窗口
    6. 输出模型输入
    
    特点：
    - 低延迟（<50ms）
    - 使用训练时的scaler
    - 维护历史窗口
    - 无未来信息泄露
    """
    
    def __init__(self,
                 ts2vec_model=None,
                 scaler_path: Optional[str] = None,
                 config: Optional[Dict] = None,
                 device: str = 'cpu'):
        """
        初始化推理数据管道
        
        Args:
            ts2vec_model: 预训练的TS2Vec模型
            scaler_path: 训练时保存的scaler路径
            config: 配置字典
            device: 计算设备
        """
        self.config = config or {}
        self.device = device
        
        # 初始化组件
        self.data_cleaner = DataCleaner()
        self.feature_calculator = FeatureCalculator()
        
        # 加载训练时的scaler
        self.feature_scaler = FeatureScaler()
        if scaler_path:
            self.feature_scaler.load(scaler_path)
            logger.info(f"已加载scaler: {scaler_path}")
        else:
            logger.warning("未提供scaler路径，将使用未拟合的scaler")
        
        # TS2Vec相关
        self.ts2vec_model = ts2vec_model
        self.embedding_generator = None
        if ts2vec_model is not None:
            self.embedding_generator = TS2VecEmbeddingGenerator(
                ts2vec_model, device, use_projection=True
            )
        
        # 特征融合
        embedding_dim = self.config.get('embedding_dim', 128)
        feature_dim = self.config.get('feature_dim', 27)
        self.feature_fusion = FeatureFusion(embedding_dim, feature_dim)
        
        # 滑动窗口配置
        self.sequence_length = self.config.get('sequence_length', 64)
        self.ts2vec_window_length = self.config.get('ts2vec_window_length', 256)
        
        # 历史数据缓冲区
        self.ohlc_buffer: Deque[pd.Series] = deque(
            maxlen=max(self.sequence_length, self.ts2vec_window_length) + 100
        )
        self.feature_buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        
        # 统计信息
        self.process_count = 0
        self.total_time = 0.0
        
        logger.info("推理数据管道初始化完成")
        logger.info(f"序列长度: {self.sequence_length}")
        logger.info(f"TS2Vec窗口长度: {self.ts2vec_window_length}")
    
    def process_new_bar(self, new_bar: pd.Series) -> Optional[torch.Tensor]:
        """
        处理新的K线数据
        
        Args:
            new_bar: 新的K线数据（包含Open, High, Low, Close, Volume等）
            
        Returns:
            模型输入序列 [1, seq_len, feature_dim]，如果缓冲区未满则返回None
        """
        import time
        start_time = time.time()
        
        # 1. 添加到缓冲区
        self.ohlc_buffer.append(new_bar)
        
        # 2. 检查是否有足够的历史数据
        if len(self.ohlc_buffer) < self.sequence_length:
            logger.debug(
                f"缓冲区数据不足: {len(self.ohlc_buffer)}/{self.sequence_length}"
            )
            return None
        
        # 3. 计算特征
        features = self._calculate_features()
        if features is None:
            return None
        
        # 4. 归一化
        normalized_features = self._normalize_features(features)
        
        # 5. 生成embedding（如果需要）
        if self.embedding_generator is not None:
            embedding = self._generate_embedding()
            if embedding is None:
                return None
            
            # 6. 特征融合
            fused_features = self._fuse_features(embedding, normalized_features)
        else:
            fused_features = torch.FloatTensor(normalized_features)
        
        # 7. 添加到特征缓冲区
        self.feature_buffer.append(fused_features.numpy())
        
        # 8. 构建序列
        if len(self.feature_buffer) < self.sequence_length:
            logger.debug(
                f"特征缓冲区不足: {len(self.feature_buffer)}/{self.sequence_length}"
            )
            return None
        
        sequence = self._build_sequence()
        
        # 统计
        elapsed = time.time() - start_time
        self.process_count += 1
        self.total_time += elapsed
        
        if self.process_count % 100 == 0:
            avg_time = self.total_time / self.process_count * 1000
            logger.info(f"处理{self.process_count}条数据，平均延迟: {avg_time:.2f}ms")
        
        return sequence
    
    def process_batch(self, bars: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        批量处理K线数据
        
        Args:
            bars: K线数据DataFrame
            
        Returns:
            模型输入序列 [batch, seq_len, feature_dim]
        """
        sequences = []
        
        for idx, bar in bars.iterrows():
            sequence = self.process_new_bar(bar)
            if sequence is not None:
                sequences.append(sequence)
        
        if not sequences:
            return None
        
        return torch.cat(sequences, dim=0)
    
    def _calculate_features(self) -> Optional[np.ndarray]:
        """计算当前时刻的特征"""
        try:
            # 转换为DataFrame
            df = pd.DataFrame(list(self.ohlc_buffer))
            
            # 计算特征
            features_df = self.feature_calculator.calculate_all_features(df)
            
            # 获取最后一行（当前时刻）
            if len(features_df) == 0:
                return None
            
            current_features = features_df.iloc[-1].values
            
            # 检查NaN
            if np.any(np.isnan(current_features)):
                logger.warning("特征包含NaN值")
                return None
            
            return current_features
        
        except Exception as e:
            logger.error(f"特征计算失败: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征（使用训练时的scaler）"""
        # 转换为DataFrame
        features_df = pd.DataFrame([features])
        
        # 使用训练时的scaler转换
        normalized = self.feature_scaler.transform(features_df)
        
        return normalized.iloc[0].values
    
    def _generate_embedding(self) -> Optional[torch.Tensor]:
        """生成TS2Vec embedding"""
        try:
            # 检查缓冲区长度
            if len(self.ohlc_buffer) < self.ts2vec_window_length:
                return None
            
            # 获取最近的窗口数据
            window_data = list(self.ohlc_buffer)[-self.ts2vec_window_length:]
            window_df = pd.DataFrame(window_data)
            
            # 提取OHLC
            ohlc = window_df[['Open', 'High', 'Low', 'Close']].values
            ohlc_tensor = torch.FloatTensor(ohlc).unsqueeze(0)  # [1, window_len, 4]
            
            # 生成embedding
            with torch.no_grad():
                embedding = self.embedding_generator.generate(ohlc_tensor)
            
            # 平均池化到单个向量
            if len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)  # [1, embedding_dim]
            
            return embedding.squeeze(0)  # [embedding_dim]
        
        except Exception as e:
            logger.error(f"Embedding生成失败: {e}")
            return None
    
    def _fuse_features(self,
                      embedding: torch.Tensor,
                      features: np.ndarray) -> torch.Tensor:
        """融合embedding和手工特征"""
        features_tensor = torch.FloatTensor(features)
        fused = self.feature_fusion.fuse(
            embedding.unsqueeze(0),
            features_tensor.unsqueeze(0)
        )
        return fused.squeeze(0)
    
    def _build_sequence(self) -> torch.Tensor:
        """构建输入序列"""
        # 获取最近的sequence_length个特征
        sequence = list(self.feature_buffer)[-self.sequence_length:]
        sequence = np.stack(sequence)
        
        # 转换为tensor并添加batch维度
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        return sequence_tensor
    
    def reset(self):
        """重置缓冲区"""
        self.ohlc_buffer.clear()
        self.feature_buffer.clear()
        self.process_count = 0
        self.total_time = 0.0
        logger.info("推理管道已重置")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.process_count == 0:
            return {
                'process_count': 0,
                'avg_latency_ms': 0.0,
                'buffer_size': len(self.ohlc_buffer)
            }
        
        return {
            'process_count': self.process_count,
            'avg_latency_ms': self.total_time / self.process_count * 1000,
            'buffer_size': len(self.ohlc_buffer),
            'feature_buffer_size': len(self.feature_buffer)
        }
    
    def warmup(self, historical_data: pd.DataFrame):
        """
        使用历史数据预热缓冲区
        
        Args:
            historical_data: 历史K线数据
        """
        logger.info(f"使用{len(historical_data)}条历史数据预热缓冲区")
        
        for idx, bar in historical_data.iterrows():
            self.process_new_bar(bar)
        
        logger.info(f"预热完成，缓冲区大小: {len(self.ohlc_buffer)}")