"""
任务7.1.1: 训练数据管道

实现端到端的训练数据管道，整合所有数据处理步骤
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import logging
from pathlib import Path
import joblib

from ..features.data_cleaner import DataCleaner
from ..features.feature_calculator import FeatureCalculator
from ..features.feature_scaler import FeatureScaler
from ..models.transformer.feature_fusion import (
    TS2VecEmbeddingGenerator,
    FeatureFusion,
    SequenceBuilder
)

logger = logging.getLogger(__name__)


class TrainingDataPipeline:
    """
    训练数据管道
    
    完整流程：
    1. 加载原始数据
    2. 数据清洗
    3. 特征计算
    4. 划分数据集
    5. 生成TS2Vec embedding
    6. 特征融合
    7. 创建序列
    
    注意：特征归一化应在特征计算阶段完成（见training/process_mes_features.py），
    本管道不再进行归一化，避免重复归一化。
    """
    
    def __init__(self,
                 ts2vec_model=None,
                 config: Optional[Dict] = None,
                 device: str = 'cpu'):
        """
        初始化训练数据管道
        
        Args:
            ts2vec_model: 预训练的TS2Vec模型
            config: 配置字典
            device: 计算设备
        """
        self.config = config or {}
        self.device = device
        
        # 初始化各个组件
        self.data_cleaner = DataCleaner()
        self.feature_calculator = FeatureCalculator()
        self.feature_scaler = FeatureScaler()
        
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
        
        # 序列构建器
        sequence_length = self.config.get('sequence_length', 64)
        self.sequence_builder = SequenceBuilder(sequence_length)
        
        # 保存路径
        self.scaler_save_path = None
        
        logger.info("训练数据管道初始化完成")
    
    def process(self,
                raw_data: pd.DataFrame,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                test_ratio: float = 0.15,
                target_column: Optional[str] = None,
                features_already_normalized: bool = False) -> Dict[str, Tuple]:
        """
        完整的数据处理流程
        
        Args:
            raw_data: 原始OHLC数据或已归一化的特征数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            target_column: 目标列名（用于生成标签）
            features_already_normalized: 特征是否已归一化（默认False）
            
        Returns:
            包含train/val/test数据集的字典
            
        注意：
            如果features_already_normalized=True，则跳过数据清洗和特征计算步骤，
            直接使用输入数据（假设已经过process_mes_features.py处理）
        """
        logger.info("=" * 50)
        logger.info("开始训练数据管道处理")
        logger.info(f"原始数据形状: {raw_data.shape}")
        
        if features_already_normalized:
            logger.info("特征已归一化，跳过清洗和特征计算步骤")
            features_df = raw_data
            cleaned_data = raw_data  # 用于生成标签
        else:
            # 1. 数据清洗
            logger.info("\n步骤1: 数据清洗")
            cleaned_data = self._clean_data(raw_data)
            logger.info(f"清洗后数据形状: {cleaned_data.shape}")
            
            # 2. 特征计算
            logger.info("\n步骤2: 特征计算")
            features_df = self._calculate_features(cleaned_data)
            logger.info(f"特征数据形状: {features_df.shape}")
        
        # 3. 生成标签（如果需要）
        labels = None
        if target_column:
            logger.info("\n步骤3: 生成标签")
            labels = self._generate_labels(cleaned_data, target_column)
            logger.info(f"标签形状: {labels.shape}")
        
        # 4. 划分数据集
        logger.info("\n步骤4: 划分数据集")
        splits = self._split_data(
            cleaned_data, features_df, labels,
            train_ratio, val_ratio, test_ratio
        )
        
        # 5. 生成TS2Vec embeddings
        if self.embedding_generator is not None:
            logger.info("\n步骤5: 生成TS2Vec embeddings")
            embedded_splits = self._generate_embeddings(splits)
        else:
            logger.info("\n步骤5: 跳过TS2Vec embeddings（未提供模型）")
            embedded_splits = splits
        
        # 6. 特征融合
        logger.info("\n步骤6: 特征融合")
        fused_splits = self._fuse_features(embedded_splits)
        
        # 7. 构建序列
        logger.info("\n步骤7: 构建序列")
        final_splits = self._build_sequences(fused_splits)
        
        logger.info("\n" + "=" * 50)
        logger.info("训练数据管道处理完成")
        self._print_summary(final_splits)
        
        return final_splits
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 处理缺失值
        cleaned = self.data_cleaner.handle_missing_values(data.copy())
        
        # 处理异常值
        cleaned = self.data_cleaner.detect_and_handle_outliers(cleaned)
        
        # 时间对齐
        cleaned = self.data_cleaner.align_time(cleaned)
        
        # 验证数据质量
        report = self.data_cleaner.validate_data_quality(cleaned)
        if not report['passed']:
            logger.warning(f"数据质量验证未通过: {report['issues']}")
        
        return cleaned
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        features = self.feature_calculator.calculate_all_features(data)
        
        # 删除包含NaN的行
        features = features.dropna()
        
        return features
    
    def _generate_labels(self, data: pd.DataFrame, target_column: str) -> pd.Series:
        """生成标签"""
        if target_column == 'future_return':
            # 计算未来收益率
            labels = data['Close'].pct_change(periods=1).shift(-1)
        elif target_column == 'future_direction':
            # 计算未来方向（涨/平/跌）
            future_return = data['Close'].pct_change(periods=1).shift(-1)
            labels = pd.cut(
                future_return,
                bins=[-np.inf, -0.001, 0.001, np.inf],
                labels=[0, 1, 2]  # 跌/平/涨
            ).astype(float)
        else:
            labels = data[target_column]
        
        return labels
    
    def _split_data(self,
                    ohlc_data: pd.DataFrame,
                    features: pd.DataFrame,
                    labels: Optional[pd.Series],
                    train_ratio: float,
                    val_ratio: float,
                    test_ratio: float) -> Dict:
        """划分数据集"""
        # 确保比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # 对齐数据
        min_len = min(len(ohlc_data), len(features))
        if labels is not None:
            min_len = min(min_len, len(labels))
        
        ohlc_data = ohlc_data.iloc[:min_len]
        features = features.iloc[:min_len]
        if labels is not None:
            labels = labels.iloc[:min_len]
        
        # 计算分割点
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': {
                'ohlc': ohlc_data.iloc[:train_end],
                'features': features.iloc[:train_end],
                'labels': labels.iloc[:train_end] if labels is not None else None
            },
            'val': {
                'ohlc': ohlc_data.iloc[train_end:val_end],
                'features': features.iloc[train_end:val_end],
                'labels': labels.iloc[train_end:val_end] if labels is not None else None
            },
            'test': {
                'ohlc': ohlc_data.iloc[val_end:],
                'features': features.iloc[val_end:],
                'labels': labels.iloc[val_end:] if labels is not None else None
            }
        }
        
        logger.info(f"训练集: {len(splits['train']['features'])} 样本")
        logger.info(f"验证集: {len(splits['val']['features'])} 样本")
        logger.info(f"测试集: {len(splits['test']['features'])} 样本")
        
        return splits
    
    
    def _generate_embeddings(self, splits: Dict) -> Dict:
        """生成TS2Vec embeddings"""
        embedded_splits = {}
        
        for split_name, split_data in splits.items():
            # 生成embeddings
            ohlc_tensor = torch.FloatTensor(
                split_data['ohlc'][['Open', 'High', 'Low', 'Close']].values
            )
            
            # 创建滑动窗口
            window_length = self.config.get('ts2vec_window_length', 256)
            if len(ohlc_tensor) < window_length:
                logger.warning(
                    f"{split_name}集数据长度{len(ohlc_tensor)}"
                    f"小于窗口长度{window_length}，跳过embedding生成"
                )
                embedded_splits[split_name] = {
                    'embeddings': None,
                    'features': split_data['features'],
                    'labels': split_data['labels']
                }
                continue
            
            # 生成embeddings
            embeddings = self.embedding_generator.generate_from_dataframe(
                split_data['ohlc'],
                window_length=window_length,
                columns=['Open', 'High', 'Low', 'Close']
            )
            
            embedded_splits[split_name] = {
                'embeddings': embeddings,
                'features': split_data['features'],
                'labels': split_data['labels']
            }
            
            logger.info(f"{split_name}集embeddings形状: {embeddings.shape}")
        
        return embedded_splits
    
    def _fuse_features(self, splits: Dict) -> Dict:
        """特征融合"""
        fused_splits = {}
        
        for split_name, split_data in splits.items():
            if 'embeddings' not in split_data or split_data['embeddings'] is None:
                # 没有embeddings，直接使用手工特征
                features = split_data['features']
                if isinstance(features, pd.DataFrame):
                    fused_features = torch.FloatTensor(features.values)
                else:
                    fused_features = torch.FloatTensor(features)
                labels = split_data.get('labels')
            else:
                # 对齐长度
                embeddings = split_data['embeddings']
                features = split_data['features']
                labels = split_data.get('labels')
                
                # embeddings可能是3D的，需要平均池化
                if len(embeddings.shape) == 3:
                    embeddings = embeddings.mean(dim=1)
                
                # 对齐长度
                min_len = min(len(embeddings), len(features))
                embeddings = embeddings[:min_len]
                
                if isinstance(features, pd.DataFrame):
                    features_tensor = torch.FloatTensor(features.iloc[:min_len].values)
                else:
                    features_tensor = torch.FloatTensor(features[:min_len])
                
                # 融合
                fused_features = self.feature_fusion.fuse(embeddings, features_tensor)
                
                # 对齐标签
                if labels is not None:
                    if isinstance(labels, pd.Series):
                        labels = labels.iloc[:min_len]
                    else:
                        labels = labels[:min_len]
            
            fused_splits[split_name] = {
                'features': fused_features,
                'labels': torch.FloatTensor(labels.values) if isinstance(labels, pd.Series) else
                         (torch.FloatTensor(labels) if labels is not None else None)
            }
            
            logger.info(f"{split_name}集融合特征形状: {fused_features.shape}")
        
        return fused_splits
    
    def _build_sequences(self, splits: Dict) -> Dict:
        """构建序列"""
        final_splits = {}
        
        for split_name, split_data in splits.items():
            features = split_data['features']
            labels = split_data['labels']
            
            # 构建序列
            try:
                sequences, sequence_labels = self.sequence_builder.build_sequences(
                    features, labels
                )
                
                final_splits[split_name] = {
                    'sequences': sequences,
                    'labels': sequence_labels
                }
                
                logger.info(f"{split_name}集序列形状: {sequences.shape}")
                if sequence_labels is not None:
                    logger.info(f"{split_name}集标签形状: {sequence_labels.shape}")
            
            except ValueError as e:
                logger.warning(f"{split_name}集序列构建失败: {e}")
                final_splits[split_name] = {
                    'sequences': None,
                    'labels': None
                }
        
        return final_splits
    
    def save_scaler(self, save_path: str):
        """保存scaler"""
        self.feature_scaler.save(save_path)
        self.scaler_save_path = save_path
        logger.info(f"Scaler已保存到: {save_path}")
    
    def _print_summary(self, splits: Dict):
        """打印处理摘要"""
        logger.info("\n处理摘要:")
        for split_name, split_data in splits.items():
            if split_data['sequences'] is not None:
                logger.info(f"{split_name}集:")
                logger.info(f"  - 序列数量: {len(split_data['sequences'])}")
                logger.info(f"  - 序列形状: {split_data['sequences'].shape}")
                if split_data['labels'] is not None:
                    logger.info(f"  - 标签形状: {split_data['labels'].shape}")