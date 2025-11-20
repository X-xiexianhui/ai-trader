"""
GPU加速的特征计算

使用PyTorch实现GPU加速的技术指标计算
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
import logging

from ..utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class GPUFeatureCalculator:
    """GPU加速的特征计算器"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化GPU特征计算器
        
        Args:
            device: 设备类型 ('auto', 'cuda', 'rocm', 'cpu')
        """
        self.gpu_manager = get_gpu_manager()
        
        if device is None or device == 'auto':
            self.device = self.gpu_manager.get_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"GPU特征计算器初始化，使用设备: {self.device}")
    
    def to_tensor(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> torch.Tensor:
        """
        转换数据为GPU张量
        
        Args:
            data: 输入数据
            
        Returns:
            torch.Tensor: GPU张量
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif isinstance(data, pd.Series):
            data = data.values
        
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        转换GPU张量为numpy数组
        
        Args:
            tensor: GPU张量
            
        Returns:
            np.ndarray: numpy数组
        """
        return tensor.cpu().numpy()
    
    def sma(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """
        简单移动平均（GPU加速）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            torch.Tensor: SMA值
        """
        # 使用卷积实现移动平均
        kernel = torch.ones(period, device=self.device) / period
        
        # 填充以保持长度
        padded = torch.nn.functional.pad(data, (period - 1, 0), mode='replicate')
        
        # 1D卷积
        result = torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            stride=1
        )
        
        return result.squeeze()
    
    def ema(self, data: torch.Tensor, period: int) -> torch.Tensor:
        """
        指数移动平均（GPU加速）
        
        Args:
            data: 输入数据
            period: 周期
            
        Returns:
            torch.Tensor: EMA值
        """
        alpha = 2.0 / (period + 1)
        result = torch.zeros_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    def rsi(self, data: torch.Tensor, period: int = 14) -> torch.Tensor:
        """
        相对强弱指标（GPU加速）
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            torch.Tensor: RSI值
        """
        # 计算价格变化
        delta = data[1:] - data[:-1]
        
        # 分离涨跌
        gains = torch.where(delta > 0, delta, torch.zeros_like(delta))
        losses = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        # 计算平均涨跌
        avg_gains = self.sma(gains, period)
        avg_losses = self.sma(losses, period)
        
        # 计算RS和RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # 填充第一个值
        rsi = torch.cat([torch.tensor([50.0], device=self.device), rsi])
        
        return rsi
    
    def macd(self, data: torch.Tensor, 
             fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, torch.Tensor]:
        """
        MACD指标（GPU加速）
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            Dict: {'macd': MACD线, 'signal': 信号线, 'histogram': 柱状图}
        """
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, data: torch.Tensor, 
                       period: int = 20, std_dev: float = 2.0) -> Dict[str, torch.Tensor]:
        """
        布林带（GPU加速）
        
        Args:
            data: 价格数据
            period: 周期
            std_dev: 标准差倍数
            
        Returns:
            Dict: {'upper': 上轨, 'middle': 中轨, 'lower': 下轨}
        """
        middle = self.sma(data, period)
        
        # 计算标准差
        squared_diff = (data.unsqueeze(0) - middle.unsqueeze(0)) ** 2
        variance = self.sma(squared_diff.squeeze(), period)
        std = torch.sqrt(variance)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def atr(self, high: torch.Tensor, low: torch.Tensor, 
            close: torch.Tensor, period: int = 14) -> torch.Tensor:
        """
        平均真实波幅（GPU加速）
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            period: 周期
            
        Returns:
            torch.Tensor: ATR值
        """
        # 计算真实波幅
        tr1 = high - low
        tr2 = torch.abs(high - torch.cat([close[:1], close[:-1]]))
        tr3 = torch.abs(low - torch.cat([close[:1], close[:-1]]))
        
        tr = torch.maximum(tr1, torch.maximum(tr2, tr3))
        
        # 计算ATR
        atr = self.sma(tr, period)
        
        return atr
    
    def stochastic(self, high: torch.Tensor, low: torch.Tensor, 
                   close: torch.Tensor, k_period: int = 14, 
                   d_period: int = 3) -> Dict[str, torch.Tensor]:
        """
        随机指标（GPU加速）
        
        Args:
            high: 最高价
            low: 最低价
            close: 收盘价
            k_period: K线周期
            d_period: D线周期
            
        Returns:
            Dict: {'k': %K值, 'd': %D值}
        """
        # 计算最高价和最低价
        highest = torch.zeros_like(close)
        lowest = torch.zeros_like(close)
        
        for i in range(len(close)):
            start = max(0, i - k_period + 1)
            highest[i] = torch.max(high[start:i+1])
            lowest[i] = torch.min(low[start:i+1])
        
        # 计算%K
        k = 100 * (close - lowest) / (highest - lowest + 1e-10)
        
        # 计算%D
        d = self.sma(k, d_period)
        
        return {'k': k, 'd': d}
    
    def batch_normalize(self, data: torch.Tensor, 
                       method: str = 'standard') -> torch.Tensor:
        """
        批量归一化（GPU加速）
        
        Args:
            data: 输入数据
            method: 归一化方法 ('standard', 'minmax', 'robust')
            
        Returns:
            torch.Tensor: 归一化后的数据
        """
        if method == 'standard':
            mean = torch.mean(data)
            std = torch.std(data)
            return (data - mean) / (std + 1e-10)
        
        elif method == 'minmax':
            min_val = torch.min(data)
            max_val = torch.max(data)
            return (data - min_val) / (max_val - min_val + 1e-10)
        
        elif method == 'robust':
            median = torch.median(data)
            q75 = torch.quantile(data, 0.75)
            q25 = torch.quantile(data, 0.25)
            iqr = q75 - q25
            return (data - median) / (iqr + 1e-10)
        
        else:
            raise ValueError(f"未知的归一化方法: {method}")
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        批量计算所有特征（GPU加速）
        
        Args:
            data: OHLCV数据
            
        Returns:
            pd.DataFrame: 包含所有特征的数据
        """
        # 转换为GPU张量
        open_t = self.to_tensor(data['open'])
        high_t = self.to_tensor(data['high'])
        low_t = self.to_tensor(data['low'])
        close_t = self.to_tensor(data['close'])
        volume_t = self.to_tensor(data['volume'])
        
        features = {}
        
        # 价格特征
        features['sma_20'] = self.to_numpy(self.sma(close_t, 20))
        features['ema_20'] = self.to_numpy(self.ema(close_t, 20))
        
        # MACD
        macd_result = self.macd(close_t)
        features['macd'] = self.to_numpy(macd_result['macd'])
        features['macd_signal'] = self.to_numpy(macd_result['signal'])
        features['macd_hist'] = self.to_numpy(macd_result['histogram'])
        
        # 布林带
        bb_result = self.bollinger_bands(close_t)
        features['bb_upper'] = self.to_numpy(bb_result['upper'])
        features['bb_middle'] = self.to_numpy(bb_result['middle'])
        features['bb_lower'] = self.to_numpy(bb_result['lower'])
        
        # ATR
        features['atr_14'] = self.to_numpy(self.atr(high_t, low_t, close_t, 14))
        
        # RSI
        features['rsi_14'] = self.to_numpy(self.rsi(close_t, 14))
        
        # 随机指标
        stoch_result = self.stochastic(high_t, low_t, close_t)
        features['stoch_k'] = self.to_numpy(stoch_result['k'])
        features['stoch_d'] = self.to_numpy(stoch_result['d'])
        
        # 创建DataFrame
        result_df = pd.DataFrame(features, index=data.index)
        
        # 清理GPU缓存
        if self.gpu_manager.is_available():
            self.gpu_manager.clear_cache()
        
        return result_df


def calculate_features_gpu(data: pd.DataFrame, 
                          device: Optional[str] = None) -> pd.DataFrame:
    """
    使用GPU计算特征的便捷函数
    
    Args:
        data: OHLCV数据
        device: 设备类型
        
    Returns:
        pd.DataFrame: 特征数据
    """
    calculator = GPUFeatureCalculator(device)
    return calculator.calculate_features(data)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    n = 1000
    dates = pd.date_range('2023-01-01', periods=n, freq='5min')
    data = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 101,
        'low': np.random.randn(n).cumsum() + 99,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # 确保OHLC关系正确
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # 测试GPU特征计算
    print("测试GPU特征计算...")
    import time
    
    start = time.time()
    features = calculate_features_gpu(data)
    gpu_time = time.time() - start
    
    print(f"\nGPU计算时间: {gpu_time:.4f}秒")
    print(f"计算的特征数: {len(features.columns)}")
    print(f"\n特征列表:")
    for col in features.columns:
        print(f"  - {col}")
    
    print(f"\n前5行数据:")
    print(features.head())