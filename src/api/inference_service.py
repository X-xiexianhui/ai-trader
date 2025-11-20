"""
推理服务模块

实现端到端的模型推理服务
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """推理配置"""
    ts2vec_model_path: str
    transformer_model_path: str
    ppo_model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    use_amp: bool = True
    cache_size: int = 100


class InferenceService:
    """推理服务"""
    
    def __init__(self, config: InferenceConfig):
        """
        初始化推理服务
        
        Args:
            config: 推理配置
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 加载模型
        self.ts2vec_model = None
        self.transformer_model = None
        self.ppo_model = None
        
        self._load_models()
        
        # 性能统计
        self.inference_count = 0
        self.total_latency = 0.0
        self.latencies = []
        
        logger.info(f"Inference service initialized on {self.device}")
    
    def _load_models(self):
        """加载所有模型"""
        try:
            # 加载TS2Vec模型
            if Path(self.config.ts2vec_model_path).exists():
                self.ts2vec_model = torch.load(
                    self.config.ts2vec_model_path,
                    map_location=self.device
                )
                self.ts2vec_model.eval()
                logger.info("TS2Vec model loaded")
            
            # 加载Transformer模型
            if Path(self.config.transformer_model_path).exists():
                self.transformer_model = torch.load(
                    self.config.transformer_model_path,
                    map_location=self.device
                )
                self.transformer_model.eval()
                logger.info("Transformer model loaded")
            
            # 加载PPO模型
            if Path(self.config.ppo_model_path).exists():
                self.ppo_model = torch.load(
                    self.config.ppo_model_path,
                    map_location=self.device
                )
                self.ppo_model.eval()
                logger.info("PPO model loaded")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def predict(
        self,
        market_data: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        端到端推理
        
        Args:
            market_data: 市场数据 (seq_len, features)
            features: 手工特征 (27,)
            
        Returns:
            prediction: 包含交易信号的字典
        """
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # 1. TS2Vec提取embedding
                if self.ts2vec_model is not None:
                    market_tensor = torch.FloatTensor(market_data).unsqueeze(0).to(self.device)
                    ts2vec_embedding = self.ts2vec_model(market_tensor)
                else:
                    # 如果没有TS2Vec模型，使用零向量
                    ts2vec_embedding = torch.zeros(1, 128).to(self.device)
                
                # 2. Transformer生成状态向量
                if self.transformer_model is not None:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    # 融合TS2Vec embedding和手工特征
                    combined_features = torch.cat([ts2vec_embedding, features_tensor], dim=-1)
                    state_vector = self.transformer_model(combined_features)
                else:
                    # 如果没有Transformer模型，直接使用特征
                    state_vector = torch.cat([
                        ts2vec_embedding,
                        torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    ], dim=-1)
                
                # 3. PPO生成交易决策
                if self.ppo_model is not None:
                    action_probs, action_values = self.ppo_model(state_vector)
                    
                    # 解析动作
                    # 假设动作空间: [direction, position_size, stop_loss, take_profit]
                    direction = torch.argmax(action_probs[0, :3]).item()  # 0:空仓, 1:多, 2:空
                    position_size = action_values[0, 0].item()  # 仓位大小 [0, 1]
                    stop_loss = action_values[0, 1].item()  # 止损百分比
                    take_profit = action_values[0, 2].item()  # 止盈百分比
                    
                    # 构建交易信号
                    signal = {
                        "direction": ["neutral", "long", "short"][direction],
                        "position_size": float(position_size),
                        "stop_loss": float(stop_loss),
                        "take_profit": float(take_profit),
                        "confidence": float(torch.max(action_probs).item()),
                        "state_vector": state_vector.cpu().numpy().tolist(),
                        "timestamp": time.time()
                    }
                else:
                    signal = {
                        "direction": "neutral",
                        "position_size": 0.0,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "confidence": 0.0,
                        "state_vector": state_vector.cpu().numpy().tolist(),
                        "timestamp": time.time()
                    }
            
            # 记录延迟
            latency = time.time() - start_time
            self.inference_count += 1
            self.total_latency += latency
            self.latencies.append(latency)
            
            # 保持最近1000次延迟记录
            if len(self.latencies) > 1000:
                self.latencies.pop(0)
            
            signal["latency_ms"] = latency * 1000
            
            return signal
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
    
    def predict_batch(
        self,
        market_data_batch: List[np.ndarray],
        features_batch: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            market_data_batch: 市场数据列表
            features_batch: 特征列表
            
        Returns:
            predictions: 预测结果列表
        """
        predictions = []
        
        for market_data, features in zip(market_data_batch, features_batch):
            prediction = self.predict(market_data, features)
            predictions.append(prediction)
        
        return predictions
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        if self.inference_count == 0:
            return {
                "inference_count": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0
            }
        
        latencies_ms = [l * 1000 for l in self.latencies]
        
        return {
            "inference_count": self.inference_count,
            "avg_latency_ms": self.total_latency / self.inference_count * 1000,
            "p50_latency_ms": np.percentile(latencies_ms, 50),
            "p95_latency_ms": np.percentile(latencies_ms, 95),
            "p99_latency_ms": np.percentile(latencies_ms, 99),
            "min_latency_ms": min(latencies_ms),
            "max_latency_ms": max(latencies_ms)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "device": str(self.device),
            "models_loaded": {
                "ts2vec": self.ts2vec_model is not None,
                "transformer": self.transformer_model is not None,
                "ppo": self.ppo_model is not None
            },
            "inference_count": self.inference_count,
            "avg_latency_ms": self.total_latency / max(self.inference_count, 1) * 1000
        }


class LocalInferenceService:
    """本地推理服务（简化版）"""
    
    def __init__(
        self,
        model_dir: str = "models",
        device: str = "auto"
    ):
        """
        初始化本地推理服务
        
        Args:
            model_dir: 模型目录
            device: 设备 (auto/cuda/cpu)
        """
        self.model_dir = Path(model_dir)
        
        # 自动选择设备
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        
        # 查找最新模型
        config = InferenceConfig(
            ts2vec_model_path=str(self.model_dir / "ts2vec" / "best_model.pt"),
            transformer_model_path=str(self.model_dir / "transformer" / "best_model.pt"),
            ppo_model_path=str(self.model_dir / "ppo" / "best_model.pt"),
            device=device
        )
        
        self.service = InferenceService(config)
        
        logger.info(f"Local inference service initialized on {device}")
    
    def predict(
        self,
        market_data: np.ndarray,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            market_data: 市场数据
            features: 手工特征
            
        Returns:
            signal: 交易信号
        """
        return self.service.predict(market_data, features)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.service.get_statistics()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return self.service.health_check()


def example_usage():
    """使用示例"""
    # 初始化服务
    service = LocalInferenceService(
        model_dir="models",
        device="auto"
    )
    
    # 准备测试数据
    market_data = np.random.randn(256, 4)  # 256个时间步，4个特征
    features = np.random.randn(27)  # 27维手工特征
    
    # 执行推理
    signal = service.predict(market_data, features)
    
    print("Trading Signal:")
    print(f"  Direction: {signal['direction']}")
    print(f"  Position Size: {signal['position_size']:.2%}")
    print(f"  Stop Loss: {signal['stop_loss']:.2%}")
    print(f"  Take Profit: {signal['take_profit']:.2%}")
    print(f"  Confidence: {signal['confidence']:.2%}")
    print(f"  Latency: {signal['latency_ms']:.2f}ms")
    
    # 获取统计信息
    stats = service.get_statistics()
    print(f"\nInference Statistics:")
    print(f"  Total Inferences: {stats['inference_count']}")
    print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P99 Latency: {stats['p99_latency_ms']:.2f}ms")
    
    # 健康检查
    health = service.health_check()
    print(f"\nHealth Check:")
    print(f"  Status: {health['status']}")
    print(f"  Device: {health['device']}")
    print(f"  Models Loaded: {health['models_loaded']}")


if __name__ == "__main__":
    example_usage()