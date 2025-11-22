"""
特征计算模块 - 计算20维手工特征

本模块实现了完整的20维手工特征计算，包括：
1. 价格与收益特征（3维）- 已删除price_slope_20和C_div_MA20
2. 波动率特征（3维）- 已删除BB_width和range_20以消除共线性
3. 技术指标特征（2维）- 已删除VWAP和MACD
4. 成交量特征（4维）
5. K线形态特征（6维）- 已删除body_ratio以消除完全共线性
6. 时间周期特征（2维）

所有特征计算严格遵循以下原则：
- 不引入未来信息（仅使用历史数据）
- 正确处理边界情况（前N根K线）
- 确保数值稳定性（处理inf/nan/除零）
- 提供详细的文档和注释
"""

import pandas as pd
import numpy as np
from ta import volatility, trend, momentum, volume as ta_volume
from typing import Dict, Optional, List
from scipy import stats
import logging
import warnings

# 配置日志
logger = logging.getLogger(__name__)

# 忽略pandas的警告
warnings.filterwarnings('ignore', category=FutureWarning)


class FeatureCalculator:
    """
    特征计算器 - 计算20维手工特征
    
    特征组成：
    1. 价格与收益特征（3维）- 已删除price_slope_20和C_div_MA20
    2. 波动率特征（3维）- 已删除BB_width和range_20
    3. 技术指标特征（2维）- 已删除VWAP和MACD
    4. 成交量特征（4维）
    5. K线形态特征（6维）- 已删除body_ratio
    6. 时间周期特征（2维）
    
    注意：
    - body_ratio已被删除，因为它与upper_shadow_ratio和lower_shadow_ratio
      存在完全共线性关系：body_ratio = 1 - upper_shadow_ratio - lower_shadow_ratio
    - BB_width已被删除，因为它与vol_20完全相关（ρ=1.0），VIF=inf
    - VWAP已被删除，因为它与EMA20几乎完全相关（ρ=0.998），VIF=774.92
    - range_20已被删除，因为它与vol_20高度相关（ρ=0.939），VIF=739.70
    - price_slope_20已被删除，因为计算复杂且信息量有限
    - C_div_MA20已被删除，因为与EMA20高度相关
    - MACD已被删除，因为与收益率特征高度相关
    """
    
    def __init__(self):
        self.feature_names = []
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有20维特征
        
        Args:
            df: 清洗后的OHLC DataFrame
            
        Returns:
            带特征的DataFrame
        """
        logger.info("开始计算特征...")
        
        df = df.copy()
        
        # 1. 价格与收益特征（3维）- 已删除price_slope_20和C_div_MA20
        df = self.calculate_price_return_features(df)
        
        # 2. 波动率特征（3维）- 已删除BB_width和range_20
        df = self.calculate_volatility_features(df)
        
        # 3. 技术指标特征（2维）- 已删除VWAP和MACD
        df = self.calculate_technical_indicators(df)
        
        # 4. 成交量特征（4维）
        df = self.calculate_volume_features(df)
        
        # 5. K线形态特征（6维）
        df = self.calculate_candlestick_features(df)
        
        # 6. 时间周期特征（2维）
        df = self.calculate_time_features(df)
        
        # 删除前期无法计算的行
        df = df.dropna()
        
        logger.info(f"特征计算完成: {len(self.feature_names)}个特征，{len(df)}行数据")
        
        return df
    
    def calculate_price_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.1: 计算价格与收益特征（3维）
        
        特征说明：
        1. ret_1: 1周期对数收益率 - log(close[t] / close[t-1])
        2. ret_5: 5周期对数收益率 - log(close[t] / close[t-5])
        3. ret_20: 20周期对数收益率 - log(close[t] / close[t-20])
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了价格与收益特征的DataFrame
            
        注意：
        - 所有计算仅使用历史数据，不引入未来信息
        - 前20根K线的某些特征会是NaN（正常现象）
        - 对数收益率处理了除零和负值情况
        """
        df = df.copy()
        
        # 1. ret_1: 1周期对数收益率
        # 处理可能的除零和负值
        close_shifted_1 = df['Close'].shift(1)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ret_1'] = np.log(df['Close'] / close_shifted_1)
        # 将inf和-inf替换为NaN
        df['ret_1'] = df['ret_1'].replace([np.inf, -np.inf], np.nan)
        
        # 2. ret_5: 5周期对数收益率
        close_shifted_5 = df['Close'].shift(5)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ret_5'] = np.log(df['Close'] / close_shifted_5)
        df['ret_5'] = df['ret_5'].replace([np.inf, -np.inf], np.nan)
        
        # 3. ret_20: 20周期对数收益率
        close_shifted_20 = df['Close'].shift(20)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['ret_20'] = np.log(df['Close'] / close_shifted_20)
        df['ret_20'] = df['ret_20'].replace([np.inf, -np.inf], np.nan)
        
        # 记录特征名称
        self.feature_names.extend(['ret_1', 'ret_5', 'ret_20'])
        
        logger.debug(f"价格与收益特征计算完成: {len(df)}行数据")
        
        return df
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.2: 计算波动率特征（3维）
        
        特征说明：
        1. ATR14: 14周期平均真实波动幅度
        2. vol_20: 20周期收盘价标准差
        3. parkinson_vol: Parkinson波动率估计（对数变换后）
        
        注意：
        - BB_width已被删除，因为它与vol_20完全相关（ρ=1.0），VIF=inf
        - range_20已被删除，因为它与vol_20高度相关（ρ=0.939），VIF=739.70
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了波动率特征的DataFrame
            
        注意：
        - 使用pandas-ta库计算ATR
        - 移除了除以Close的"伪归一化"，真正的归一化由FeatureScaler完成
        - Parkinson波动率对high=low的情况做了特殊处理
        - parkinson_vol已进行对数变换 log(parkinson_vol + ε)，以改善右偏分布
        """
        df = df.copy()
        
        # 1. ATR14: 14周期平均真实波动幅度
        # 使用ta库计算ATR
        atr_indicator = volatility.AverageTrueRange(
            high=df['High'], low=df['Low'], close=df['Close'], window=14
        )
        df['ATR14'] = atr_indicator.average_true_range()
        
        # 2. vol_20: 20周期收盘价标准差
        df['vol_20'] = df['Close'].rolling(window=20, min_periods=20).std()
        
        # 3. parkinson_vol: Parkinson波动率（对数变换）
        # 公式: sqrt(1/(4*log(2)) * log(high/low)^2)
        # 处理high=low的情况（避免log(1)=0导致的问题）
        with np.errstate(divide='ignore', invalid='ignore'):
            high_low_ratio = df['High'] / df['Low']
            log_ratio = np.log(high_low_ratio)
            parkinson_vol_raw = np.sqrt(1 / (4 * np.log(2)) * log_ratio ** 2)
        
        # 对数变换: log(parkinson_vol + ε)
        # 根据分析结果，原始parkinson_vol具有右偏、长尾和极端值聚集特征
        # 对数变换可以改善分布，使其更接近正态分布
        epsilon = 1e-10  # 避免log(0)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['parkinson_vol'] = np.log(parkinson_vol_raw + epsilon)
        df['parkinson_vol'] = df['parkinson_vol'].replace([np.inf, -np.inf], np.nan)
        
        # 记录特征名称（已删除BB_width和range_20）
        self.feature_names.extend([
            'ATR14', 'vol_20', 'parkinson_vol'
        ])
        
        logger.debug(f"波动率特征计算完成: {len(df)}行数据")
        
        return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.3: 计算技术指标特征（2维）
        
        特征说明：
        1. EMA20: 20周期指数移动平均
        2. stoch: 随机指标的%K值 - Stochastic(9, 3, 3)
        
        注意：
        - VWAP已被删除，因为它与EMA20几乎完全相关（ρ=0.998），VIF=774.92
        - MACD已被删除，因为与收益率特征高度相关
        
        Args:
            df: 包含OHLC和Volume数据的DataFrame
            
        Returns:
            添加了技术指标特征的DataFrame
            
        注意：
        - 使用pandas-ta库计算所有技术指标
        """
        df = df.copy()
        
        # 1. EMA20: 20周期指数移动平均
        ema_indicator = trend.EMAIndicator(close=df['Close'], window=20)
        df['EMA20'] = ema_indicator.ema_indicator()
        
        # 2. stoch: 随机指标%K值
        # ta库的StochasticOscillator
        stoch_indicator = momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'],
            window=9, smooth_window=3
        )
        df['stoch'] = stoch_indicator.stoch()
        
        # 记录特征名称（已删除VWAP和MACD）
        self.feature_names.extend(['EMA20', 'stoch'])
        
        logger.debug(f"技术指标特征计算完成: {len(df)}行数据")
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.4: 计算成交量特征（4维）
        
        特征说明：
        1. volume: 原始成交量
        2. volume_zscore: 成交量Z-score - (volume - mean(volume[-20:])) / std(volume[-20:])
        3. volume_change_1: 成交量变化率 - (volume[t] - volume[t-1]) / volume[t-1]
        4. OBV_slope_20: OBV的20周期线性回归斜率
        
        Args:
            df: 包含Volume数据的DataFrame
            
        Returns:
            添加了成交量特征的DataFrame
            
        注意：
        - 处理成交量为0的情况
        - OBV使用pandas-ta库计算
        - 斜率计算使用scipy.stats.linregress
        """
        df = df.copy()
        
        if 'Volume' not in df.columns:
            logger.warning("缺少Volume列，成交量特征使用NaN填充")
            df['volume'] = np.nan
            df['volume_zscore'] = np.nan
            df['volume_change_1'] = np.nan
            df['OBV_slope_20'] = np.nan
            self.feature_names.extend(['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'])
            return df
        
        # 1. volume: 原始成交量
        df['volume'] = df['Volume'].copy()
        
        # 2. volume_zscore: 成交量Z-score
        vol_mean = df['Volume'].rolling(window=20, min_periods=20).mean()
        vol_std = df['Volume'].rolling(window=20, min_periods=20).std()
        with np.errstate(divide='ignore', invalid='ignore'):
            df['volume_zscore'] = (df['Volume'] - vol_mean) / vol_std
        df['volume_zscore'] = df['volume_zscore'].replace([np.inf, -np.inf], np.nan)
        
        # 3. volume_change_1: 成交量变化率
        # 使用pct_change，自动处理除零情况
        df['volume_change_1'] = df['Volume'].pct_change()
        df['volume_change_1'] = df['volume_change_1'].replace([np.inf, -np.inf], np.nan)
        
        # 4. OBV_slope_20: OBV的20周期线性回归斜率
        # 使用ta库计算OBV
        obv_indicator = ta_volume.OnBalanceVolumeIndicator(
            close=df['Close'], volume=df['Volume']
        )
        obv_series = obv_indicator.on_balance_volume()
        
        def calculate_obv_slope(x):
            """计算OBV的线性回归斜率"""
            if len(x) < 20 or np.isnan(x).any():
                return np.nan
            try:
                slope, _, _, _, _ = stats.linregress(range(len(x)), x)
                return slope
            except:
                return np.nan
        
        df['OBV_slope_20'] = obv_series.rolling(window=20, min_periods=20).apply(
            calculate_obv_slope, raw=True
        )
        
        # 记录特征名称
        self.feature_names.extend(['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'])
        
        logger.debug(f"成交量特征计算完成: {len(df)}行数据")
        
        return df
    
    def calculate_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.5: 计算K线形态特征（6维）
        
        特征说明：
        1. pos_in_range_20: 在20周期范围内的相对位置 - (close - LL20) / (HH20 - LL20)
        2. dist_to_HH20: 到最高点的距离 - (close - HH20)
        3. dist_to_LL20: 到最低点的距离 - (close - LL20)
        4. upper_shadow_ratio: 上影线比例 - (high - max(open, close)) / (high - low)
        5. lower_shadow_ratio: 下影线比例 - (min(open, close) - low) / (high - low)
        6. FVG: 公允价值缺口
        
        注意：
        - body_ratio已删除（与shadow_ratio完全共线）
        - pos_in_range_20保留除法（这是真正的相对位置，范围[0,1]）
        - dist_to_HH20和dist_to_LL20移除了除以Close的操作
        - shadow_ratio保留除法（这是真正的比例特征，范围[0,1]）
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            添加了K线形态特征的DataFrame
            
        注意：
        - 比例特征在[0,1]范围内
        - 处理high=low的情况（避免除零）
        - FVG特征见_calculate_fvg方法
        """
        df = df.copy()
        
        # 计算20周期高低点
        hh20 = df['High'].rolling(window=20, min_periods=20).max()
        ll20 = df['Low'].rolling(window=20, min_periods=20).min()
        
        # 1. pos_in_range_20: 在20周期范围内的相对位置（保留，这是真正的相对位置）
        range_20 = hh20 - ll20
        with np.errstate(divide='ignore', invalid='ignore'):
            df['pos_in_range_20'] = np.where(
                range_20 > 0,
                (df['Close'] - ll20) / range_20,
                0.5  # 如果范围为0，返回中间位置
            )
        df['pos_in_range_20'] = df['pos_in_range_20'].replace([np.inf, -np.inf], np.nan)
        
        # 2. dist_to_HH20: 到最高点的距离（移除除以Close）
        df['dist_to_HH20'] = df['Close'] - hh20
        
        # 3. dist_to_LL20: 到最低点的距离（移除除以Close）
        df['dist_to_LL20'] = df['Close'] - ll20
        
        # 计算K线范围（避免除零）
        candle_range = df['High'] - df['Low']
        # 将0替换为一个很小的数，避免除零
        candle_range = candle_range.replace(0, 1e-10)
        
        # 4. upper_shadow_ratio: 上影线比例
        max_oc = df[['Open', 'Close']].max(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['upper_shadow_ratio'] = (df['High'] - max_oc) / candle_range
        df['upper_shadow_ratio'] = df['upper_shadow_ratio'].replace([np.inf, -np.inf], np.nan)
        df['upper_shadow_ratio'] = df['upper_shadow_ratio'].clip(0, 1)
        
        # 5. lower_shadow_ratio: 下影线比例
        min_oc = df[['Open', 'Close']].min(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            df['lower_shadow_ratio'] = (min_oc - df['Low']) / candle_range
        df['lower_shadow_ratio'] = df['lower_shadow_ratio'].replace([np.inf, -np.inf], np.nan)
        df['lower_shadow_ratio'] = df['lower_shadow_ratio'].clip(0, 1)
        
        # 6. FVG: 公允价值缺口（任务1.2.7）
        df['FVG'] = self._calculate_fvg(df)
        
        # 记录特征名称
        self.feature_names.extend([
            'pos_in_range_20', 'dist_to_HH20', 'dist_to_LL20',
            'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'
        ])
        
        logger.debug(f"K线形态特征计算完成: {len(df)}行数据")
        
        return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        任务1.2.6: 计算时间周期特征（2维）
        
        特征说明：
        1. sin_tod: 时间的正弦编码 - sin(2π * hour / 24)
        2. cos_tod: 时间的余弦编码 - cos(2π * hour / 24)
        
        Args:
            df: 带时间索引的DataFrame
            
        Returns:
            添加了时间周期特征的DataFrame
            
        注意：
        - 使用正弦余弦编码保持时间的周期性
        - 特征值在[-1, 1]范围内
        - 能正确处理跨天情况
        """
        df = df.copy()
        
        # 提取小时信息（包括分钟的小数部分）
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                hour = df.index.hour + df.index.minute / 60.0
            else:
                # 尝试转换为DatetimeIndex
                dt_index = pd.to_datetime(df.index)
                hour = dt_index.hour + dt_index.minute / 60.0
        except Exception as e:
            logger.warning(f"无法提取时间信息: {e}，使用0填充")
            hour = pd.Series(0.0, index=df.index)
        
        # 正弦余弦编码
        df['sin_tod'] = np.sin(2 * np.pi * hour / 24)
        df['cos_tod'] = np.cos(2 * np.pi * hour / 24)
        
        # 记录特征名称
        self.feature_names.extend(['sin_tod', 'cos_tod'])
        
        logger.debug(f"时间周期特征计算完成: {len(df)}行数据")
        
        return df
    
    def _calculate_fvg(self, df: pd.DataFrame) -> pd.Series:
        """
        任务1.2.7: 计算FVG公允价值缺口
        
        FVG (Fair Value Gap) 是一种短期价格失衡，以三根K线的结构呈现：
        
        多头FVG检测：
        - 第一根K线的最高价 < 第三根K线的最低价
        - 中间第二根K线强势上涨，形成可见缺口
        
        空头FVG检测：
        - 第一根K线的最低价 > 第三根K线的最高价
        - 中间第二根K线强势下跌，切穿价格区间
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            FVG特征Series:
            - 正值：多头FVG强度（缺口大小/当前价格）
            - 负值：空头FVG强度
            - 0：无FVG
            
        注意：
        - 前两根K线返回0（无法计算）
        - 缺口大小归一化到当前收盘价
        - 处理了除零情况
        """
        fvg = pd.Series(0.0, index=df.index)
        
        # 从第3根K线开始计算（索引2）
        for i in range(2, len(df)):
            try:
                # 获取三根K线的价格
                high_1 = df['High'].iloc[i-2]  # 第一根K线最高价
                low_1 = df['Low'].iloc[i-2]    # 第一根K线最低价
                high_3 = df['High'].iloc[i]    # 第三根K线最高价
                low_3 = df['Low'].iloc[i]      # 第三根K线最低价
                close_current = df['Close'].iloc[i]  # 当前收盘价
                
                # 检查数据有效性
                if np.isnan([high_1, low_1, high_3, low_3, close_current]).any():
                    continue
                if close_current <= 0:
                    continue
                
                # 检测多头FVG（Bullish FVG）
                if high_1 < low_3:
                    # 存在向上缺口
                    gap_size = low_3 - high_1
                    fvg_strength = gap_size / close_current  # 归一化
                    fvg.iloc[i] = fvg_strength
                
                # 检测空头FVG（Bearish FVG）
                elif low_1 > high_3:
                    # 存在向下缺口
                    gap_size = low_1 - high_3
                    fvg_strength = -gap_size / close_current  # 归一化，负值表示空头
                    fvg.iloc[i] = fvg_strength
                
                # 否则无FVG，保持为0
                
            except Exception as e:
                logger.debug(f"FVG计算出错 at index {i}: {e}")
                continue
        
        logger.debug(f"FVG特征计算完成: 检测到{(fvg != 0).sum()}个FVG")
        
        return fvg
    
    def get_feature_names(self) -> list:
        """获取所有特征名称"""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, list]:
        """获取特征分组"""
        return {
            'price_return': ['ret_1', 'ret_5', 'ret_20'],  # 已删除price_slope_20和C_div_MA20
            'volatility': ['ATR14', 'vol_20', 'parkinson_vol'],  # 已删除BB_width和range_20
            'technical': ['EMA20', 'stoch'],  # 已删除VWAP和MACD
            'volume': ['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'],
            'candlestick': ['pos_in_range_20', 'dist_to_HH20', 'dist_to_LL20',
                          'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'],
            'time': ['sin_tod', 'cos_tod']
        }