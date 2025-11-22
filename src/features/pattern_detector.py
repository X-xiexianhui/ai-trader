"""
技术形态检测器
识别K线图中的经典技术分析形态
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """形态类型枚举"""
    # 单K线形态
    PIN_BAR = "pin_bar"  # Pin Bar (长影线)
    DOJI = "doji"  # 十字星
    HAMMER = "hammer"  # 锤子线
    SHOOTING_STAR = "shooting_star"  # 流星线
    ENGULFING_BULLISH = "engulfing_bullish"  # 看涨吞没
    ENGULFING_BEARISH = "engulfing_bearish"  # 看跌吞没
    
    # 趋势形态
    UPTREND = "uptrend"  # 上涨趋势
    DOWNTREND = "downtrend"  # 下跌趋势
    SIDEWAYS = "sideways"  # 横盘震荡
    
    # 反转形态
    HEAD_SHOULDERS = "head_shoulders"  # 头肩顶
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"  # 头肩底
    DOUBLE_TOP = "double_top"  # 双顶
    DOUBLE_BOTTOM = "double_bottom"  # 双底
    TRIPLE_TOP = "triple_top"  # 三重顶
    TRIPLE_BOTTOM = "triple_bottom"  # 三重底
    
    # 持续形态
    WEDGE_RISING = "wedge_rising"  # 上升楔形
    WEDGE_FALLING = "wedge_falling"  # 下降楔形
    FLAG_BULLISH = "flag_bullish"  # 看涨旗形
    FLAG_BEARISH = "flag_bearish"  # 看跌旗形
    TRIANGLE_ASCENDING = "triangle_ascending"  # 上升三角形
    TRIANGLE_DESCENDING = "triangle_descending"  # 下降三角形
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"  # 对称三角形
    
    # 其他
    NONE = "none"  # 无明显形态


@dataclass
class PatternResult:
    """形态检测结果"""
    pattern_type: PatternType
    confidence: float  # 0-1之间的置信度
    start_idx: int  # 形态起始位置
    end_idx: int  # 形态结束位置
    key_points: Dict[str, int]  # 关键点位置
    description: str  # 形态描述


class PatternDetector:
    """技术形态检测器"""
    
    def __init__(self, min_pattern_length: int = 10, max_pattern_length: int = 100):
        """
        初始化形态检测器
        
        Args:
            min_pattern_length: 最小形态长度
            max_pattern_length: 最大形态长度
        """
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
    
    def detect_patterns(self, df: pd.DataFrame, window_size: int = 50) -> List[PatternResult]:
        """
        检测所有形态
        
        Args:
            df: OHLC数据，需包含 open, high, low, close 列
            window_size: 滑动窗口大小
            
        Returns:
            检测到的形态列表
        """
        patterns = []
        
        # 单K线形态检测
        for i in range(len(df)):
            pattern = self._detect_single_candle_pattern(df.iloc[i])
            if pattern:
                patterns.append(pattern)
        
        # 多K线形态检测（使用滑动窗口）
        for i in range(window_size, len(df)):
            window_df = df.iloc[i-window_size:i]
            
            # 趋势形态
            trend_pattern = self._detect_trend(window_df, i-window_size, i)
            if trend_pattern:
                patterns.append(trend_pattern)
            
            # 反转形态
            reversal_patterns = self._detect_reversal_patterns(window_df, i-window_size, i)
            patterns.extend(reversal_patterns)
            
            # 持续形态
            continuation_patterns = self._detect_continuation_patterns(window_df, i-window_size, i)
            patterns.extend(continuation_patterns)
        
        return patterns
    
    def _detect_single_candle_pattern(self, candle: pd.Series) -> Optional[PatternResult]:
        """检测单K线形态"""
        open_price = candle['open']
        high = candle['high']
        low = candle['low']
        close = candle['close']
        
        body = abs(close - open_price)
        upper_shadow = high - max(open_price, close)
        lower_shadow = min(open_price, close) - low
        total_range = high - low
        
        if total_range == 0:
            return None
        
        # Pin Bar检测
        if self._is_pin_bar(body, upper_shadow, lower_shadow, total_range):
            return PatternResult(
                pattern_type=PatternType.PIN_BAR,
                confidence=0.8,
                start_idx=candle.name,
                end_idx=candle.name,
                key_points={'pin': candle.name},
                description="Pin Bar - 长影线反转信号"
            )
        
        # Doji检测
        if body / total_range < 0.1:
            return PatternResult(
                pattern_type=PatternType.DOJI,
                confidence=0.7,
                start_idx=candle.name,
                end_idx=candle.name,
                key_points={'doji': candle.name},
                description="Doji - 十字星，市场犹豫信号"
            )
        
        # Hammer检测
        if self._is_hammer(body, upper_shadow, lower_shadow, total_range, close > open_price):
            return PatternResult(
                pattern_type=PatternType.HAMMER,
                confidence=0.75,
                start_idx=candle.name,
                end_idx=candle.name,
                key_points={'hammer': candle.name},
                description="Hammer - 锤子线，底部反转信号"
            )
        
        # Shooting Star检测
        if self._is_shooting_star(body, upper_shadow, lower_shadow, total_range, close < open_price):
            return PatternResult(
                pattern_type=PatternType.SHOOTING_STAR,
                confidence=0.75,
                start_idx=candle.name,
                end_idx=candle.name,
                key_points={'star': candle.name},
                description="Shooting Star - 流星线，顶部反转信号"
            )
        
        return None
    
    def _is_pin_bar(self, body: float, upper_shadow: float, lower_shadow: float, total_range: float) -> bool:
        """判断是否为Pin Bar"""
        # Pin Bar特征：一端影线很长（至少是实体的2倍），另一端影线很短
        long_shadow = max(upper_shadow, lower_shadow)
        short_shadow = min(upper_shadow, lower_shadow)
        
        return (long_shadow > body * 2 and 
                short_shadow < body * 0.5 and 
                body / total_range < 0.3)
    
    def _is_hammer(self, body: float, upper_shadow: float, lower_shadow: float, 
                   total_range: float, is_bullish: bool) -> bool:
        """判断是否为锤子线"""
        return (lower_shadow > body * 2 and 
                upper_shadow < body * 0.3 and 
                body / total_range > 0.2 and
                is_bullish)
    
    def _is_shooting_star(self, body: float, upper_shadow: float, lower_shadow: float,
                         total_range: float, is_bearish: bool) -> bool:
        """判断是否为流星线"""
        return (upper_shadow > body * 2 and 
                lower_shadow < body * 0.3 and 
                body / total_range > 0.2 and
                is_bearish)
    
    def _detect_trend(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测趋势"""
        closes = df['close'].values
        
        # 使用线性回归检测趋势
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        
        # 计算R²来衡量趋势强度
        y_pred = slope * x + np.mean(closes)
        ss_res = np.sum((closes - y_pred) ** 2)
        ss_tot = np.sum((closes - np.mean(closes)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # 趋势判断阈值
        slope_threshold = closes[0] * 0.001  # 0.1%的斜率
        r_squared_threshold = 0.5
        
        if r_squared > r_squared_threshold:
            if slope > slope_threshold:
                return PatternResult(
                    pattern_type=PatternType.UPTREND,
                    confidence=r_squared,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points={'start': start_idx, 'end': end_idx},
                    description=f"上涨趋势 (斜率: {slope:.2f}, R²: {r_squared:.2f})"
                )
            elif slope < -slope_threshold:
                return PatternResult(
                    pattern_type=PatternType.DOWNTREND,
                    confidence=r_squared,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points={'start': start_idx, 'end': end_idx},
                    description=f"下跌趋势 (斜率: {slope:.2f}, R²: {r_squared:.2f})"
                )
            else:
                return PatternResult(
                    pattern_type=PatternType.SIDEWAYS,
                    confidence=r_squared,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points={'start': start_idx, 'end': end_idx},
                    description=f"横盘震荡 (R²: {r_squared:.2f})"
                )
        
        return None
    
    def _detect_reversal_patterns(self, df: pd.DataFrame, start_idx: int, 
                                  end_idx: int) -> List[PatternResult]:
        """检测反转形态"""
        patterns = []
        highs = df['high'].values
        lows = df['low'].values
        
        # 头肩顶/底检测
        hs_pattern = self._detect_head_shoulders(highs, lows, start_idx, end_idx)
        if hs_pattern:
            patterns.append(hs_pattern)
        
        # 双顶/底检测
        double_pattern = self._detect_double_top_bottom(highs, lows, start_idx, end_idx)
        if double_pattern:
            patterns.append(double_pattern)
        
        return patterns
    
    def _detect_head_shoulders(self, highs: np.ndarray, lows: np.ndarray,
                               start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测头肩顶/底"""
        # 寻找局部极值点
        peaks = self._find_peaks(highs)
        troughs = self._find_peaks(-lows)
        
        # 头肩顶需要3个峰值
        if len(peaks) >= 3:
            # 检查是否符合头肩顶模式：左肩 < 头部 > 右肩，且左右肩高度相近
            for i in range(len(peaks) - 2):
                left_shoulder = highs[peaks[i]]
                head = highs[peaks[i+1]]
                right_shoulder = highs[peaks[i+2]]
                
                if (head > left_shoulder and head > right_shoulder and
                    abs(left_shoulder - right_shoulder) / head < 0.05):
                    return PatternResult(
                        pattern_type=PatternType.HEAD_SHOULDERS,
                        confidence=0.7,
                        start_idx=start_idx + peaks[i],
                        end_idx=start_idx + peaks[i+2],
                        key_points={
                            'left_shoulder': start_idx + peaks[i],
                            'head': start_idx + peaks[i+1],
                            'right_shoulder': start_idx + peaks[i+2]
                        },
                        description="头肩顶 - 顶部反转形态"
                    )
        
        # 头肩底需要3个谷值
        if len(troughs) >= 3:
            for i in range(len(troughs) - 2):
                left_shoulder = lows[troughs[i]]
                head = lows[troughs[i+1]]
                right_shoulder = lows[troughs[i+2]]
                
                if (head < left_shoulder and head < right_shoulder and
                    abs(left_shoulder - right_shoulder) / head < 0.05):
                    return PatternResult(
                        pattern_type=PatternType.INVERSE_HEAD_SHOULDERS,
                        confidence=0.7,
                        start_idx=start_idx + troughs[i],
                        end_idx=start_idx + troughs[i+2],
                        key_points={
                            'left_shoulder': start_idx + troughs[i],
                            'head': start_idx + troughs[i+1],
                            'right_shoulder': start_idx + troughs[i+2]
                        },
                        description="头肩底 - 底部反转形态"
                    )
        
        return None
    
    def _detect_double_top_bottom(self, highs: np.ndarray, lows: np.ndarray,
                                  start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测双顶/底"""
        peaks = self._find_peaks(highs)
        troughs = self._find_peaks(-lows)
        
        # 双顶检测
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = highs[peaks[i]]
                peak2 = highs[peaks[i+1]]
                
                # 两个峰值高度相近（误差<3%）
                if abs(peak1 - peak2) / max(peak1, peak2) < 0.03:
                    return PatternResult(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.65,
                        start_idx=start_idx + peaks[i],
                        end_idx=start_idx + peaks[i+1],
                        key_points={
                            'peak1': start_idx + peaks[i],
                            'peak2': start_idx + peaks[i+1]
                        },
                        description="双顶 - 顶部反转形态"
                    )
        
        # 双底检测
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1 = lows[troughs[i]]
                trough2 = lows[troughs[i+1]]
                
                if abs(trough1 - trough2) / max(trough1, trough2) < 0.03:
                    return PatternResult(
                        pattern_type=PatternType.DOUBLE_BOTTOM,
                        confidence=0.65,
                        start_idx=start_idx + troughs[i],
                        end_idx=start_idx + troughs[i+1],
                        key_points={
                            'trough1': start_idx + troughs[i],
                            'trough2': start_idx + troughs[i+1]
                        },
                        description="双底 - 底部反转形态"
                    )
        
        return None
    
    def _detect_continuation_patterns(self, df: pd.DataFrame, start_idx: int,
                                     end_idx: int) -> List[PatternResult]:
        """检测持续形态"""
        patterns = []
        highs = df['high'].values
        lows = df['low'].values
        
        # 楔形检测
        wedge_pattern = self._detect_wedge(highs, lows, start_idx, end_idx)
        if wedge_pattern:
            patterns.append(wedge_pattern)
        
        # 旗形检测
        flag_pattern = self._detect_flag(df, start_idx, end_idx)
        if flag_pattern:
            patterns.append(flag_pattern)
        
        # 三角形检测
        triangle_pattern = self._detect_triangle(highs, lows, start_idx, end_idx)
        if triangle_pattern:
            patterns.append(triangle_pattern)
        
        return patterns
    
    def _detect_wedge(self, highs: np.ndarray, lows: np.ndarray,
                     start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测楔形"""
        # 拟合上下轨趋势线
        x = np.arange(len(highs))
        upper_slope, _ = np.polyfit(x, highs, 1)
        lower_slope, _ = np.polyfit(x, lows, 1)
        
        # 楔形特征：上下轨都有明显斜率且方向相同，但收敛
        if abs(upper_slope) > 0.001 and abs(lower_slope) > 0.001:
            # 上升楔形：两条线都上升，但上轨斜率小于下轨
            if upper_slope > 0 and lower_slope > 0 and upper_slope < lower_slope:
                return PatternResult(
                    pattern_type=PatternType.WEDGE_RISING,
                    confidence=0.6,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points={'start': start_idx, 'end': end_idx},
                    description="上升楔形 - 看跌持续形态"
                )
            
            # 下降楔形：两条线都下降，但下轨斜率大于上轨（绝对值）
            if upper_slope < 0 and lower_slope < 0 and abs(lower_slope) < abs(upper_slope):
                return PatternResult(
                    pattern_type=PatternType.WEDGE_FALLING,
                    confidence=0.6,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    key_points={'start': start_idx, 'end': end_idx},
                    description="下降楔形 - 看涨持续形态"
                )
        
        return None
    
    def _detect_flag(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测旗形"""
        closes = df['close'].values
        
        # 旗形特征：前期有强势趋势，然后小幅回调整理
        # 分为前半段和后半段
        mid_point = len(closes) // 2
        first_half = closes[:mid_point]
        second_half = closes[mid_point:]
        
        # 计算前半段趋势
        x1 = np.arange(len(first_half))
        slope1, _ = np.polyfit(x1, first_half, 1)
        
        # 计算后半段趋势
        x2 = np.arange(len(second_half))
        slope2, _ = np.polyfit(x2, second_half, 1)
        
        # 看涨旗形：前期强势上涨，后期小幅下跌整理
        if slope1 > closes[0] * 0.002 and -closes[0] * 0.001 < slope2 < 0:
            return PatternResult(
                pattern_type=PatternType.FLAG_BULLISH,
                confidence=0.6,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points={'pole_start': start_idx, 'flag_start': start_idx + mid_point, 'end': end_idx},
                description="看涨旗形 - 上涨持续形态"
            )
        
        # 看跌旗形：前期强势下跌，后期小幅上涨整理
        if slope1 < -closes[0] * 0.002 and 0 < slope2 < closes[0] * 0.001:
            return PatternResult(
                pattern_type=PatternType.FLAG_BEARISH,
                confidence=0.6,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points={'pole_start': start_idx, 'flag_start': start_idx + mid_point, 'end': end_idx},
                description="看跌旗形 - 下跌持续形态"
            )
        
        return None
    
    def _detect_triangle(self, highs: np.ndarray, lows: np.ndarray,
                        start_idx: int, end_idx: int) -> Optional[PatternResult]:
        """检测三角形"""
        x = np.arange(len(highs))
        upper_slope, _ = np.polyfit(x, highs, 1)
        lower_slope, _ = np.polyfit(x, lows, 1)
        
        # 上升三角形：上轨水平，下轨上升
        if abs(upper_slope) < 0.0005 and lower_slope > 0.001:
            return PatternResult(
                pattern_type=PatternType.TRIANGLE_ASCENDING,
                confidence=0.65,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points={'start': start_idx, 'end': end_idx},
                description="上升三角形 - 看涨持续形态"
            )
        
        # 下降三角形：下轨水平，上轨下降
        if abs(lower_slope) < 0.0005 and upper_slope < -0.001:
            return PatternResult(
                pattern_type=PatternType.TRIANGLE_DESCENDING,
                confidence=0.65,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points={'start': start_idx, 'end': end_idx},
                description="下降三角形 - 看跌持续形态"
            )
        
        # 对称三角形：上轨下降，下轨上升，收敛
        if upper_slope < -0.001 and lower_slope > 0.001:
            return PatternResult(
                pattern_type=PatternType.TRIANGLE_SYMMETRICAL,
                confidence=0.6,
                start_idx=start_idx,
                end_idx=end_idx,
                key_points={'start': start_idx, 'end': end_idx},
                description="对称三角形 - 中性持续形态"
            )
        
        return None
    
    def _find_peaks(self, data: np.ndarray, prominence: float = 0.02) -> List[int]:
        """
        寻找局部极值点
        
        Args:
            data: 数据序列
            prominence: 显著性阈值（相对于数据范围）
            
        Returns:
            极值点索引列表
        """
        peaks = []
        data_range = np.max(data) - np.min(data)
        min_prominence = data_range * prominence
        
        for i in range(1, len(data) - 1):
            # 检查是否为局部最大值
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # 检查显著性
                left_min = np.min(data[max(0, i-10):i])
                right_min = np.min(data[i+1:min(len(data), i+11)])
                prominence_val = data[i] - max(left_min, right_min)
                
                if prominence_val >= min_prominence:
                    peaks.append(i)
        
        return peaks
    
    def create_pattern_labels(self, df: pd.DataFrame, window_size: int = 50) -> pd.DataFrame:
        """
        为数据集创建形态标签
        
        Args:
            df: OHLC数据
            window_size: 检测窗口大小
            
        Returns:
            包含形态标签的DataFrame
        """
        # 检测所有形态
        patterns = self.detect_patterns(df, window_size)
        
        # 创建标签列
        df = df.copy()
        df['pattern'] = PatternType.NONE.value
        df['pattern_confidence'] = 0.0
        
        # 为每个时间点分配最高置信度的形态
        for pattern in patterns:
            for idx in range(pattern.start_idx, pattern.end_idx + 1):
                if idx < len(df):
                    if pattern.confidence > df.loc[idx, 'pattern_confidence']:
                        df.loc[idx, 'pattern'] = pattern.pattern_type.value
                        df.loc[idx, 'pattern_confidence'] = pattern.confidence
        
        return df


def main():
    """测试形态检测器"""
    # 加载数据
    df = pd.read_csv('data/processed/MES_cleaned_5m.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 创建检测器
    detector = PatternDetector()
    
    # 检测形态
    print("检测技术形态...")
    patterns = detector.detect_patterns(df.head(1000), window_size=50)
    
    # 统计形态分布
    pattern_counts = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.value
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    print("\n形态分布:")
    for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern_type}: {count}")
    
    # 创建标签数据集
    print("\n创建标签数据集...")
    labeled_df = detector.create_pattern_labels(df.head(1000), window_size=50)
    
    # 保存结果
    output_path = 'data/processed/MES_with_patterns.csv'
    labeled_df.to_csv(output_path)
    print(f"\n已保存到: {output_path}")
    
    # 显示样例
    print("\n样例数据:")
    print(labeled_df[labeled_df['pattern'] != 'none'].head(10)[['open', 'high', 'low', 'close', 'pattern', 'pattern_confidence']])


if __name__ == '__main__':
    main()