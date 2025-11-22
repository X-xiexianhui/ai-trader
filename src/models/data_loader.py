"""
数据加载和预处理模块

功能：
1. 从processed目录加载训练/验证/测试数据
2. 读取OHLCV数据并归一化
3. 创建时序窗口数据集
4. 生成辅助任务的标签
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarketStateLabeler:
    """
    市场状态标签生成器
    
    根据未来5根K线的价格变化，将市场状态分为4类：
    1. 上涨 (Uptrend): 收益率 > threshold 且波动率较低
    2. 下跌 (Downtrend): 收益率 < -threshold 且波动率较低
    3. 震荡 (Sideways): |收益率| <= threshold
    4. 趋势反转 (Reversal): 波动率高且方向改变
    """
    
    def __init__(
        self,
        return_threshold: float = 0.005,  # 0.5%
        volatility_threshold: float = 0.01  # 1%
    ):
        """
        Args:
            return_threshold: 判断上涨/下跌的收益率阈值
            volatility_threshold: 判断震荡/反转的波动率阈值
        """
        self.return_threshold = return_threshold
        self.volatility_threshold = volatility_threshold
    
    def label_market_state(
        self,
        df: pd.DataFrame,
        future_periods: int = 5
    ) -> pd.Series:
        """
        生成市场状态标签
        
        Args:
            df: 包含OHLCV的DataFrame
            future_periods: 未来K线数量
            
        Returns:
            市场状态标签Series (0: 上涨, 1: 下跌, 2: 震荡, 3: 反转)
        """
        # 计算未来收益率
        future_return = df['Close'].pct_change(future_periods).shift(-future_periods)
        
        # 计算未来波动率（标准差）
        future_volatility = df['Close'].rolling(window=future_periods).std().shift(-future_periods)
        future_volatility = future_volatility / df['Close']  # 归一化
        
        # 计算当前趋势方向（过去5根K线）
        past_return = df['Close'].pct_change(future_periods)
        
        # 初始化标签
        labels = pd.Series(2, index=df.index)  # 默认为震荡
        
        # 1. 上涨趋势
        uptrend_mask = (
            (future_return > self.return_threshold) &
            (future_volatility < self.volatility_threshold)
        )
        labels[uptrend_mask] = 0
        
        # 2. 下跌趋势
        downtrend_mask = (
            (future_return < -self.return_threshold) &
            (future_volatility < self.volatility_threshold)
        )
        labels[downtrend_mask] = 1
        
        # 3. 趋势反转（高波动率且方向改变）
        reversal_mask = (
            (future_volatility >= self.volatility_threshold) &
            (np.sign(future_return) != np.sign(past_return))
        )
        labels[reversal_mask] = 3
        
        # 4. 震荡（默认值，已设置）
        
        return labels


class OHLCVDataset(Dataset):
    """
    OHLCV时序数据集
    
    功能：
    1. 创建滑动窗口序列
    2. 归一化OHLCV数据
    3. 生成辅助任务标签
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        future_periods: int = 5,
        scaler_type: str = 'standard',
        fit_scaler: bool = True,
        scaler: Optional[object] = None
    ):
        """
        Args:
            df: 包含OHLCV的DataFrame
            seq_len: 序列长度（输入窗口大小）
            future_periods: 预测未来K线数量
            scaler_type: 归一化方法 ('standard' 或 'minmax')
            fit_scaler: 是否拟合归一化器（训练集True，验证/测试集False）
            scaler: 预训练的归一化器（用于验证/测试集）
        """
        self.df = df.copy()
        self.seq_len = seq_len
        self.future_periods = future_periods
        
        # 确保列名正确
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.df.columns for col in required_cols):
            # 尝试小写列名
            self.df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
        
        # 提取OHLCV数据
        self.ohlcv_data = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # 归一化
        if fit_scaler:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"不支持的归一化方法: {scaler_type}")
            
            self.ohlcv_normalized = self.scaler.fit_transform(self.ohlcv_data)
        else:
            if scaler is None:
                raise ValueError("验证/测试集必须提供预训练的scaler")
            self.scaler = scaler
            self.ohlcv_normalized = self.scaler.transform(self.ohlcv_data)
        
        # 生成市场状态标签
        labeler = MarketStateLabeler()
        self.market_state_labels = labeler.label_market_state(
            self.df, future_periods=future_periods
        ).values
        
        # 计算未来波动率（辅助任务2）
        future_volatility = self.df['Close'].rolling(
            window=future_periods
        ).std().shift(-future_periods)
        self.volatility_targets = (future_volatility / self.df['Close']).values
        
        # 计算未来收益率（辅助任务3）
        future_return = self.df['Close'].pct_change(
            future_periods
        ).shift(-future_periods)
        self.return_targets = future_return.values
        
        # 计算有效样本数量（去除NaN）
        self.valid_indices = self._get_valid_indices()
        
        logger.info(f"数据集创建完成: {len(self.valid_indices)} 个有效样本")
    
    def _get_valid_indices(self) -> list:
        """获取有效样本索引（去除NaN）"""
        valid_indices = []
        
        for i in range(self.seq_len, len(self.df) - self.future_periods):
            # 检查输入序列是否有NaN
            if np.isnan(self.ohlcv_normalized[i-self.seq_len:i]).any():
                continue
            
            # 检查标签是否有NaN
            if (np.isnan(self.market_state_labels[i]) or
                np.isnan(self.volatility_targets[i]) or
                np.isnan(self.return_targets[i])):
                continue
            
            valid_indices.append(i)
        
        return valid_indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            字典包含：
            - sequence: OHLCV序列 [seq_len, 5]
            - market_state: 市场状态标签 (0-3)
            - volatility: 波动率目标
            - returns: 收益率目标
        """
        actual_idx = self.valid_indices[idx]
        
        # 获取输入序列
        sequence = self.ohlcv_normalized[actual_idx-self.seq_len:actual_idx]
        
        # 获取标签
        market_state = self.market_state_labels[actual_idx]
        volatility = self.volatility_targets[actual_idx]
        returns = self.return_targets[actual_idx]
        
        return {
            'sequence': torch.FloatTensor(sequence),
            'market_state': torch.LongTensor([int(market_state)])[0],
            'volatility': torch.FloatTensor([volatility]),
            'returns': torch.FloatTensor([returns])
        }
    
    def get_scaler(self):
        """获取归一化器（用于验证/测试集）"""
        return self.scaler


def load_data(
    data_path: str,
    data_type: str = 'train'
) -> pd.DataFrame:
    """
    从processed目录加载数据
    
    Args:
        data_path: 数据文件路径
        data_type: 数据类型 ('train', 'val', 'test')
        
    Returns:
        DataFrame
    """
    logger.info(f"加载{data_type}数据: {data_path}")
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    return df


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    seq_len: int = 60,
    future_periods: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    scaler_type: str = 'standard'
) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        test_path: 测试数据路径
        seq_len: 序列长度
        future_periods: 预测未来K线数量
        batch_size: 批次大小
        num_workers: 数据加载线程数
        scaler_type: 归一化方法
        
    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    # 加载数据
    train_df = load_data(train_path, 'train')
    val_df = load_data(val_path, 'val')
    test_df = load_data(test_path, 'test')
    
    # 创建训练集（拟合归一化器）
    train_dataset = OHLCVDataset(
        train_df,
        seq_len=seq_len,
        future_periods=future_periods,
        scaler_type=scaler_type,
        fit_scaler=True
    )
    
    # 获取训练集的归一化器
    scaler = train_dataset.get_scaler()
    
    # 创建验证集（使用训练集的归一化器）
    val_dataset = OHLCVDataset(
        val_df,
        seq_len=seq_len,
        future_periods=future_periods,
        fit_scaler=False,
        scaler=scaler
    )
    
    # 创建测试集（使用训练集的归一化器）
    test_dataset = OHLCVDataset(
        test_df,
        seq_len=seq_len,
        future_periods=future_periods,
        fit_scaler=False,
        scaler=scaler
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"数据加载器创建完成:")
    logger.info(f"  - 训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    logger.info(f"  - 验证集: {len(val_dataset)} 样本, {len(val_loader)} 批次")
    logger.info(f"  - 测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == '__main__':
    # 测试数据加载器
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.utils.logger import setup_logger
    
    # 设置日志
    logger = setup_logger('data_loader_test', log_level='INFO')
    
    print("测试数据加载器...")
    
    # 数据路径
    data_dir = project_root / 'data' / 'processed'
    train_path = data_dir / 'MES_train.csv'
    val_path = data_dir / 'MES_val.csv'
    test_path = data_dir / 'MES_test.csv'
    
    # 检查文件是否存在
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        print("错误: 数据文件不存在，请先运行 training/split_dataset.py")
        sys.exit(1)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        str(train_path),
        str(val_path),
        str(test_path),
        seq_len=60,
        future_periods=5,
        batch_size=32,
        num_workers=0,  # 测试时使用0
        scaler_type='standard'
    )
    
    # 测试一个批次
    print("\n测试一个批次...")
    batch = next(iter(train_loader))
    
    print(f"批次内容:")
    print(f"  - sequence: {batch['sequence'].shape}")
    print(f"  - market_state: {batch['market_state'].shape}")
    print(f"  - volatility: {batch['volatility'].shape}")
    print(f"  - returns: {batch['returns'].shape}")
    
    print(f"\n市场状态分布:")
    unique, counts = torch.unique(batch['market_state'], return_counts=True)
    for state, count in zip(unique, counts):
        state_names = ['上涨', '下跌', '震荡', '反转']
        print(f"  - {state_names[state]}: {count}")
    
    print("\n数据加载器测试完成！")