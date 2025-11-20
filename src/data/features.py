"""
特征计算模块 - 计算27维手工特征
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    特征计算器 - 计算27维手工特征
    
    特征组成：
    1. 价格与收益特征（5维）
    2. 波动率特征（5维）
    3. 技术指标特征（4维）
    4. 成交量特征（4维）
    5. K线形态特征（7维）
    6. 时间周期特征（2维）
    """
    
    def __init__(self):
        self.feature_names = []
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有27维特征
        
        Args:
            df: 清洗后的OHLC DataFrame
            
        Returns:
            带特征的DataFrame
        """
        logger.info("开始计算特征...")
        
        df = df.copy()
        
        # 1. 价格与收益特征（5维）
        df = self.calculate_price_return_features(df)
        
        # 2. 波动率特征（5维）
        df = self.calculate_volatility_features(df)
        
        # 3. 技术指标特征（4维）
        df = self.calculate_technical_indicators(df)
        
        # 4. 成交量特征（4维）
        df = self.calculate_volume_features(df)
        
        # 5. K线形态特征（7维）
        df = self.calculate_candlestick_features(df)
        
        # 6. 时间周期特征（2维）
        df = self.calculate_time_features(df)
        
        # 删除前期无法计算的行
        df = df.dropna()
        
        logger.info(f"特征计算完成: {len(self.feature_names)}个特征，{len(df)}行数据")
        
        return df
    
    def calculate_price_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.1: 计算价格与收益特征（5维）
        
        特征：
        1. ret_1: log(close[t] / close[t-1])
        2. ret_5: log(close[t] / close[t-5])
        3. ret_20: log(close[t] / close[t-20])
        4. price_slope_20: 20周期价格线性回归斜率
        5. C_div_MA20: close[t] / MA(close, 20)
        """
        df = df.copy()
        
        # 1. ret_1: 1周期对数收益率
        df['ret_1'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. ret_5: 5周期对数收益率
        df['ret_5'] = np.log(df['Close'] / df['Close'].shift(5))
        
        # 3. ret_20: 20周期对数收益率
        df['ret_20'] = np.log(df['Close'] / df['Close'].shift(20))
        
        # 4. price_slope_20: 20周期价格线性回归斜率
        df['price_slope_20'] = df['Close'].rolling(window=20).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else np.nan,
            raw=True
        )
        
        # 5. C_div_MA20: 收盘价除以20周期均线
        ma20 = df['Close'].rolling(window=20).mean()
        df['C_div_MA20'] = df['Close'] / ma20
        
        self.feature_names.extend(['ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20'])
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.2: 计算波动率特征（5维）
        
        特征：
        1. ATR14_norm: ATR(14) / close[t]
        2. vol_20: std(close[-20:])
        3. range_20_norm: (HH20 - LL20) / close[t]
        4. BB_width_norm: (BB_upper - BB_lower) / close[t]
        5. parkinson_vol: sqrt(1/(4*log(2)) * log(high/low)^2)
        """
        df = df.copy()
        
        # 1. ATR14_norm: 归一化ATR
        atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR14_norm'] = atr / df['Close']
        
        # 2. vol_20: 20周期收盘价标准差
        df['vol_20'] = df['Close'].rolling(window=20).std()
        
        # 3. range_20_norm: 归一化20周期价格范围
        hh20 = df['High'].rolling(window=20).max()
        ll20 = df['Low'].rolling(window=20).min()
        df['range_20_norm'] = (hh20 - ll20) / df['Close']
        
        # 4. BB_width_norm: 归一化布林带宽度
        bb = ta.bbands(df['Close'], length=20, std=2)
        if bb is not None and len(bb.columns) >= 3:
            bb_upper = bb.iloc[:, 0]  # BBU_20_2.0
            bb_lower = bb.iloc[:, 2]  # BBL_20_2.0
            df['BB_width_norm'] = (bb_upper - bb_lower) / df['Close']
        else:
            df['BB_width_norm'] = np.nan
        
        # 5. parkinson_vol: Parkinson波动率
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * np.log(df['High'] / df['Low']) ** 2
        )
        
        self.feature_names.extend([
            'ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol'
        ])
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.3: 计算技术指标特征（4维）
        
        特征：
        1. EMA20: EMA(close, 20)
        2. stoch: Stochastic(9, 3, 3) 的%K值
        3. MACD: MACD(12, 26, 9) 的MACD线
        4. VWAP: 成交量加权平均价
        """
        df = df.copy()
        
        # 1. EMA20: 20周期指数移动平均
        df['EMA20'] = ta.ema(df['Close'], length=20)
        
        # 2. stoch: 随机指标%K值
        stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=9, d=3, smooth_k=3)
        if stoch is not None and len(stoch.columns) >= 1:
            df['stoch'] = stoch.iloc[:, 0]  # STOCHk_9_3_3
        else:
            df['stoch'] = np.nan
        
        # 3. MACD: MACD线
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and len(macd.columns) >= 1:
            df['MACD'] = macd.iloc[:, 0]  # MACD_12_26_9
        else:
            df['MACD'] = np.nan
        
        # 4. VWAP: 成交量加权平均价（使用滚动窗口）
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / \
                     df['Volume'].rolling(window=20).sum()
        
        self.feature_names.extend(['EMA20', 'stoch', 'MACD', 'VWAP'])
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.4: 计算成交量特征（4维）
        
        特征：
        1. volume: 原始成交量
        2. volume_zscore: (volume - mean(volume[-20:])) / std(volume[-20:])
        3. volume_change_1: (volume[t] - volume[t-1]) / volume[t-1]
        4. OBV_slope_20: OBV的20周期线性回归斜率
        """
        df = df.copy()
        
        # 1. volume: 原始成交量（已存在）
        df['volume'] = df['Volume']
        
        # 2. volume_zscore: 成交量Z-score
        vol_mean = df['Volume'].rolling(window=20).mean()
        vol_std = df['Volume'].rolling(window=20).std()
        df['volume_zscore'] = (df['Volume'] - vol_mean) / vol_std
        
        # 3. volume_change_1: 成交量变化率
        df['volume_change_1'] = df['Volume'].pct_change()
        
        # 4. OBV_slope_20: OBV斜率
        obv = ta.obv(df['Close'], df['Volume'])
        if obv is not None:
            df['OBV_slope_20'] = obv.rolling(window=20).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == 20 else np.nan,
                raw=True
            )
        else:
            df['OBV_slope_20'] = np.nan
        
        self.feature_names.extend(['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'])
        
        return df
    
    def calculate_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.5: 计算K线形态特征（7维）
        
        特征：
        1. pos_in_range_20: (close - LL20) / (HH20 - LL20)
        2. dist_to_HH20_norm: (close - HH20) / close
        3. dist_to_LL20_norm: (close - LL20) / close
        4. body_ratio: |close - open| / (high - low)
        5. upper_shadow_ratio: (high - max(open, close)) / (high - low)
        6. lower_shadow_ratio: (min(open, close) - low) / (high - low)
        7. FVG: 公允价值缺口
        """
        df = df.copy()
        
        # 计算20周期高低点
        hh20 = df['High'].rolling(window=20).max()
        ll20 = df['Low'].rolling(window=20).min()
        
        # 1. pos_in_range_20: 在20周期范围内的相对位置
        range_20 = hh20 - ll20
        df['pos_in_range_20'] = np.where(
            range_20 > 0,
            (df['Close'] - ll20) / range_20,
            0.5  # 如果范围为0，返回中间位置
        )
        
        # 2. dist_to_HH20_norm: 到最高点的归一化距离
        df['dist_to_HH20_norm'] = (df['Close'] - hh20) / df['Close']
        
        # 3. dist_to_LL20_norm: 到最低点的归一化距离
        df['dist_to_LL20_norm'] = (df['Close'] - ll20) / df['Close']
        
        # 计算K线范围（避免除零）
        candle_range = df['High'] - df['Low']
        candle_range = candle_range.replace(0, np.nan)
        
        # 4. body_ratio: 实体比例
        df['body_ratio'] = np.abs(df['Close'] - df['Open']) / candle_range
        
        # 5. upper_shadow_ratio: 上影线比例
        max_oc = df[['Open', 'Close']].max(axis=1)
        df['upper_shadow_ratio'] = (df['High'] - max_oc) / candle_range
        
        # 6. lower_shadow_ratio: 下影线比例
        min_oc = df[['Open', 'Close']].min(axis=1)
        df['lower_shadow_ratio'] = (min_oc - df['Low']) / candle_range
        
        # 7. FVG: 公允价值缺口（任务1.2.7）
        df['FVG'] = self._calculate_fvg(df)
        
        self.feature_names.extend([
            'pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm',
            'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'
        ])
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.6: 计算时间周期特征（2维）
        
        特征：
        1. sin_tod: sin(2π * hour / 24)
        2. cos_tod: cos(2π * hour / 24)
        """
        df = df.copy()
        
        # 提取小时信息
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour + df.index.minute / 60.0
        else:
            hour = pd.to_datetime(df.index).hour + pd.to_datetime(df.index).minute / 60.0
        
        # 正弦余弦编码
        df['sin_tod'] = np.sin(2 * np.pi * hour / 24)
        df['cos_tod'] = np.cos(2 * np.pi * hour / 24)
        
        self.feature_names.extend(['sin_tod', 'cos_tod'])
        
        return df
    
    def _calculate_fvg(self, df: pd.DataFrame) -> pd.Series:
        """
        任务1.2.7: 计算FVG公允价值缺口
        
        FVG检测逻辑：
        - 多头FVG：第一根K线最高价 < 第三根K线最低价
        - 空头FVG：第一根K线最低价 > 第三根K线最高价
        
        Returns:
            FVG特征列（正值=多头FVG，负值=空头FVG，0=无FVG）
        """
        fvg = pd.Series(0.0, index=df.index)
        
        for i in range(2, len(df)):
            # 获取三根K线
            high_1 = df['High'].iloc[i-2]
            low_1 = df['Low'].iloc[i-2]
            high_3 = df['High'].iloc[i]
            low_3 = df['Low'].iloc[i]
            close_current = df['Close'].iloc[i]
            
            # 检测多头FVG
            if high_1 < low_3:
                gap_size = (low_3 - high_1) / close_current
                fvg.iloc[i] = gap_size
            
            # 检测空头FVG
            elif low_1 > high_3:
                gap_size = (low_1 - high_3) / close_current
                fvg.iloc[i] = -gap_size
        
        return fvg
    
    def get_feature_names(self) -> list:
        """获取所有特征名称"""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, list]:
        """获取特征分组"""
        return {
            'price_return': ['ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20'],
            'volatility': ['ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol'],
            'technical': ['EMA20', 'stoch', 'MACD', 'VWAP'],
            'volume': ['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'],
            'candlestick': ['pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm',
                          'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'],
            'time': ['sin_tod', 'cos_tod']
        }