"""
交易信号生成器

支持多种信号生成策略和AI模型集成
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import logging

from ..utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalStrength(Enum):
    """信号强度"""
    STRONG = 2
    MODERATE = 1
    WEAK = 0.5


class TradingSignal:
    """交易信号"""
    
    def __init__(self, 
                 signal_type: SignalType,
                 strength: float = 1.0,
                 confidence: float = 1.0,
                 timestamp: Optional[pd.Timestamp] = None,
                 price: Optional[float] = None,
                 metadata: Optional[Dict] = None):
        """
        初始化交易信号
        
        Args:
            signal_type: 信号类型
            strength: 信号强度 (0-2)
            confidence: 信号置信度 (0-1)
            timestamp: 时间戳
            price: 价格
            metadata: 元数据
        """
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence
        self.timestamp = timestamp
        self.price = price
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return (f"TradingSignal(type={self.signal_type.name}, "
                f"strength={self.strength:.2f}, "
                f"confidence={self.confidence:.2f})")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'signal_type': self.signal_type.value,
            'signal_name': self.signal_type.name,
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'price': self.price,
            'metadata': self.metadata
        }


class SignalGenerator:
    """信号生成器基类"""
    
    def __init__(self, name: str = "BaseSignalGenerator"):
        """
        初始化信号生成器
        
        Args:
            name: 生成器名称
        """
        self.name = name
        self.signals_history = []
    
    def generate(self, data: pd.DataFrame, **kwargs) -> TradingSignal:
        """
        生成交易信号
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            TradingSignal: 交易信号
        """
        raise NotImplementedError("子类必须实现generate方法")
    
    def batch_generate(self, data: pd.DataFrame, **kwargs) -> List[TradingSignal]:
        """
        批量生成信号
        
        Args:
            data: 市场数据
            **kwargs: 其他参数
            
        Returns:
            List[TradingSignal]: 信号列表
        """
        signals = []
        for i in range(len(data)):
            signal = self.generate(data.iloc[:i+1], **kwargs)
            signals.append(signal)
        return signals


class TechnicalSignalGenerator(SignalGenerator):
    """技术指标信号生成器"""
    
    def __init__(self, 
                 short_window: int = 20,
                 long_window: int = 50,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30):
        """
        初始化技术指标信号生成器
        
        Args:
            short_window: 短期均线窗口
            long_window: 长期均线窗口
            rsi_period: RSI周期
            rsi_overbought: RSI超买阈值
            rsi_oversold: RSI超卖阈值
        """
        super().__init__("TechnicalSignalGenerator")
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def generate(self, data: pd.DataFrame, **kwargs) -> TradingSignal:
        """生成技术指标信号"""
        if len(data) < self.long_window:
            return TradingSignal(SignalType.HOLD, strength=0, confidence=0)
        
        # 计算移动平均线
        short_ma = data['close'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = data['close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # 计算RSI
        rsi = self._calculate_rsi(data['close'], self.rsi_period)
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        # 生成信号
        signal_type = SignalType.HOLD
        strength = 0.0
        confidence = 0.5
        
        # 均线交叉信号
        if short_ma > long_ma:
            ma_signal = 1
        elif short_ma < long_ma:
            ma_signal = -1
        else:
            ma_signal = 0
        
        # RSI信号
        if rsi > self.rsi_overbought:
            rsi_signal = -1  # 超买，卖出
        elif rsi < self.rsi_oversold:
            rsi_signal = 1   # 超卖，买入
        else:
            rsi_signal = 0
        
        # 综合信号
        combined_signal = ma_signal + rsi_signal
        
        if combined_signal >= 2:
            signal_type = SignalType.BUY
            strength = 2.0
            confidence = 0.9
        elif combined_signal == 1:
            signal_type = SignalType.BUY
            strength = 1.0
            confidence = 0.7
        elif combined_signal <= -2:
            signal_type = SignalType.SELL
            strength = 2.0
            confidence = 0.9
        elif combined_signal == -1:
            signal_type = SignalType.SELL
            strength = 1.0
            confidence = 0.7
        
        metadata = {
            'short_ma': short_ma,
            'long_ma': long_ma,
            'rsi': rsi,
            'ma_signal': ma_signal,
            'rsi_signal': rsi_signal
        }
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=timestamp,
            price=current_price,
            metadata=metadata
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]


class AIModelSignalGenerator(SignalGenerator):
    """AI模型信号生成器"""
    
    def __init__(self, 
                 model: Optional[torch.nn.Module] = None,
                 device: Optional[str] = None,
                 threshold: float = 0.5):
        """
        初始化AI模型信号生成器
        
        Args:
            model: PyTorch模型
            device: 设备类型
            threshold: 信号阈值
        """
        super().__init__("AIModelSignalGenerator")
        self.model = model
        self.threshold = threshold
        
        # GPU管理
        self.gpu_manager = get_gpu_manager()
        if device is None or device == 'auto':
            self.device = self.gpu_manager.get_device()
        else:
            self.device = torch.device(device)
        
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
        
        logger.info(f"AI模型信号生成器初始化，使用设备: {self.device}")
    
    def set_model(self, model: torch.nn.Module):
        """设置模型"""
        self.model = model
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, data: pd.DataFrame, features: Optional[np.ndarray] = None, **kwargs) -> TradingSignal:
        """生成AI模型信号"""
        if self.model is None:
            logger.warning("模型未设置，返回HOLD信号")
            return TradingSignal(SignalType.HOLD, strength=0, confidence=0)
        
        # 准备特征
        if features is None:
            features = self._prepare_features(data)
        
        # 转换为张量
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(features_tensor)
            
            # 假设模型输出为 [batch, 3] (买入、持有、卖出的概率)
            if output.dim() == 2 and output.shape[1] == 3:
                probs = torch.softmax(output, dim=1)[0]
                buy_prob = probs[0].item()
                hold_prob = probs[1].item()
                sell_prob = probs[2].item()
                
                # 确定信号类型
                max_prob = max(buy_prob, hold_prob, sell_prob)
                
                if max_prob == buy_prob and buy_prob > self.threshold:
                    signal_type = SignalType.BUY
                    confidence = buy_prob
                elif max_prob == sell_prob and sell_prob > self.threshold:
                    signal_type = SignalType.SELL
                    confidence = sell_prob
                else:
                    signal_type = SignalType.HOLD
                    confidence = hold_prob
                
                # 计算强度
                strength = (max_prob - self.threshold) / (1 - self.threshold) * 2
                strength = max(0, min(2, strength))
                
            else:
                # 假设模型输出为单一值（回归）
                value = output.item()
                
                if value > self.threshold:
                    signal_type = SignalType.BUY
                    confidence = min(1.0, value)
                    strength = min(2.0, value * 2)
                elif value < -self.threshold:
                    signal_type = SignalType.SELL
                    confidence = min(1.0, abs(value))
                    strength = min(2.0, abs(value) * 2)
                else:
                    signal_type = SignalType.HOLD
                    confidence = 1 - abs(value)
                    strength = 0
        
        current_price = data['close'].iloc[-1]
        timestamp = data.index[-1]
        
        metadata = {
            'model_output': output.cpu().numpy().tolist(),
            'features_shape': features.shape
        }
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            timestamp=timestamp,
            price=current_price,
            metadata=metadata
        )
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征"""
        # 简单示例：使用最近的价格和成交量
        features = []
        
        if 'close' in data.columns:
            features.append(data['close'].iloc[-1])
        if 'volume' in data.columns:
            features.append(data['volume'].iloc[-1])
        if 'high' in data.columns:
            features.append(data['high'].iloc[-1])
        if 'low' in data.columns:
            features.append(data['low'].iloc[-1])
        
        return np.array(features)


class EnsembleSignalGenerator(SignalGenerator):
    """集成信号生成器"""
    
    def __init__(self, 
                 generators: List[SignalGenerator],
                 weights: Optional[List[float]] = None,
                 voting_method: str = 'weighted'):
        """
        初始化集成信号生成器
        
        Args:
            generators: 信号生成器列表
            weights: 权重列表
            voting_method: 投票方法 ('weighted', 'majority', 'unanimous')
        """
        super().__init__("EnsembleSignalGenerator")
        self.generators = generators
        
        if weights is None:
            self.weights = [1.0] * len(generators)
        else:
            self.weights = weights
        
        self.voting_method = voting_method
    
    def generate(self, data: pd.DataFrame, **kwargs) -> TradingSignal:
        """生成集成信号"""
        # 收集所有生成器的信号
        signals = []
        for generator in self.generators:
            signal = generator.generate(data, **kwargs)
            signals.append(signal)
        
        # 根据投票方法合并信号
        if self.voting_method == 'weighted':
            return self._weighted_voting(signals)
        elif self.voting_method == 'majority':
            return self._majority_voting(signals)
        elif self.voting_method == 'unanimous':
            return self._unanimous_voting(signals)
        else:
            raise ValueError(f"未知的投票方法: {self.voting_method}")
    
    def _weighted_voting(self, signals: List[TradingSignal]) -> TradingSignal:
        """加权投票"""
        total_weight = sum(self.weights)
        
        # 计算加权信号值
        weighted_signal = 0
        weighted_strength = 0
        weighted_confidence = 0
        
        for signal, weight in zip(signals, self.weights):
            weighted_signal += signal.signal_type.value * weight * signal.confidence
            weighted_strength += signal.strength * weight
            weighted_confidence += signal.confidence * weight
        
        weighted_signal /= total_weight
        weighted_strength /= total_weight
        weighted_confidence /= total_weight
        
        # 确定最终信号类型
        if weighted_signal > 0.3:
            signal_type = SignalType.BUY
        elif weighted_signal < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # 使用最新信号的时间戳和价格
        timestamp = signals[-1].timestamp
        price = signals[-1].price
        
        metadata = {
            'individual_signals': [s.to_dict() for s in signals],
            'weights': self.weights,
            'voting_method': self.voting_method
        }
        
        return TradingSignal(
            signal_type=signal_type,
            strength=weighted_strength,
            confidence=weighted_confidence,
            timestamp=timestamp,
            price=price,
            metadata=metadata
        )
    
    def _majority_voting(self, signals: List[TradingSignal]) -> TradingSignal:
        """多数投票"""
        # 统计各类型信号数量
        buy_count = sum(1 for s in signals if s.signal_type == SignalType.BUY)
        sell_count = sum(1 for s in signals if s.signal_type == SignalType.SELL)
        hold_count = sum(1 for s in signals if s.signal_type == SignalType.HOLD)
        
        # 确定多数信号
        max_count = max(buy_count, sell_count, hold_count)
        
        if max_count == buy_count:
            signal_type = SignalType.BUY
        elif max_count == sell_count:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # 计算平均强度和置信度
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        timestamp = signals[-1].timestamp
        price = signals[-1].price
        
        metadata = {
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'voting_method': self.voting_method
        }
        
        return TradingSignal(
            signal_type=signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            timestamp=timestamp,
            price=price,
            metadata=metadata
        )
    
    def _unanimous_voting(self, signals: List[TradingSignal]) -> TradingSignal:
        """一致投票（所有信号必须一致）"""
        # 检查是否所有信号一致
        signal_types = [s.signal_type for s in signals]
        
        if len(set(signal_types)) == 1:
            # 所有信号一致
            signal_type = signal_types[0]
            avg_strength = np.mean([s.strength for s in signals])
            avg_confidence = np.mean([s.confidence for s in signals])
        else:
            # 信号不一致，返回HOLD
            signal_type = SignalType.HOLD
            avg_strength = 0
            avg_confidence = 0.5
        
        timestamp = signals[-1].timestamp
        price = signals[-1].price
        
        metadata = {
            'unanimous': len(set(signal_types)) == 1,
            'voting_method': self.voting_method
        }
        
        return TradingSignal(
            signal_type=signal_type,
            strength=avg_strength,
            confidence=avg_confidence,
            timestamp=timestamp,
            price=price,
            metadata=metadata
        )


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
    }, index=dates)
    
    print("测试技术指标信号生成器...")
    tech_gen = TechnicalSignalGenerator()
    signal = tech_gen.generate(data)
    print(f"信号: {signal}")
    print(f"元数据: {signal.metadata}")
    
    print("\n测试AI模型信号生成器...")
    ai_gen = AIModelSignalGenerator()
    signal = ai_gen.generate(data)
    print(f"信号: {signal}")
    
    print("\n测试集成信号生成器...")
    ensemble_gen = EnsembleSignalGenerator(
        generators=[tech_gen, ai_gen],
        weights=[0.6, 0.4],
        voting_method='weighted'
    )
    signal = ensemble_gen.generate(data)
    print(f"集成信号: {signal}")
    print(f"置信度: {signal.confidence:.2f}")