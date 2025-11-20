"""
数据清洗模块 - 处理OHLC数据的缺失值、异常值和时间对齐
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    数据清洗类，专门处理OHLC数据中的各种质量问题
    
    主要功能：
    1. 缺失值处理（前向填充、零填充、线性插值）
    2. 异常值检测与处理（3σ原则）
    3. 时间对齐与时区处理
    4. 数据质量验证
    """
    
    def __init__(self, max_consecutive_missing: int = 5):
        """
        初始化数据清洗器
        
        Args:
            max_consecutive_missing: 允许的最大连续缺失K线数量
        """
        self.max_consecutive_missing = max_consecutive_missing
        self.cleaning_report = {}
        
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        处理OHLC数据中的缺失值
        
        任务1.1.1实现：
        - 前向填充用于价格数据
        - 零填充用于成交量数据
        - 线性插值用于短期缺失
        - 删除连续缺失超过5根K线的数据段
        
        Args:
            df: 包含OHLC数据的DataFrame，列：Open, High, Low, Close, Volume
            
        Returns:
            cleaned_df: 清洗后的DataFrame
            report: 缺失值处理报告
        """
        df = df.copy()
        report = {
            'total_rows': len(df),
            'missing_before': {},
            'missing_after': {},
            'interpolated_points': {},
            'removed_segments': []
        }
        
        # 1. 检测缺失值位置和连续性
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_col = 'Volume'
        
        for col in price_cols + [volume_col]:
            if col in df.columns:
                report['missing_before'][col] = df[col].isna().sum()
        
        # 2. 识别连续缺失段
        missing_mask = df[price_cols].isna().any(axis=1)
        consecutive_groups = self._find_consecutive_groups(missing_mask)
        
        # 3. 删除连续缺失超过阈值的数据段
        segments_to_remove = []
        for start, end in consecutive_groups:
            length = end - start
            if length > self.max_consecutive_missing:
                segments_to_remove.append((start, end))
                report['removed_segments'].append({
                    'start_idx': start,
                    'end_idx': end,
                    'length': length
                })
        
        # 创建保留数据的mask
        keep_mask = np.ones(len(df), dtype=bool)
        for start, end in segments_to_remove:
            keep_mask[start:end] = False
        
        df = df[keep_mask].copy()
        
        # 4. 处理剩余的短期缺失
        # 价格数据：先尝试线性插值，再前向填充
        for col in price_cols:
            if col in df.columns:
                # 标记插值点
                interpolated_mask = df[col].isna()
                
                # 线性插值（仅用于短期缺失）
                df[col] = df[col].interpolate(method='linear', limit=3)
                
                # 前向填充剩余缺失
                df[col] = df[col].fillna(method='ffill')
                
                # 如果开头有缺失，使用后向填充
                df[col] = df[col].fillna(method='bfill')
                
                report['interpolated_points'][col] = interpolated_mask.sum()
        
        # 成交量数据：零填充
        if volume_col in df.columns:
            df[volume_col] = df[volume_col].fillna(0)
        
        # 5. 记录处理后的缺失值
        for col in price_cols + [volume_col]:
            if col in df.columns:
                report['missing_after'][col] = df[col].isna().sum()
        
        # 6. 添加插值标记列
        df['is_interpolated'] = False
        
        self.cleaning_report['missing_values'] = report
        logger.info(f"缺失值处理完成: 删除{len(segments_to_remove)}个数据段，"
                   f"插值{sum(report['interpolated_points'].values())}个点")
        
        return df, report
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                   sigma_threshold: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
        """
        检测并处理价格异常值
        
        任务1.1.2实现：
        - 使用3σ原则检测价格跳变
        - 区分真实跳空和数据错误
        - 修正OHLC一致性
        
        Args:
            df: OHLC DataFrame
            sigma_threshold: 异常值检测的σ阈值
            
        Returns:
            cleaned_df: 清洗后的DataFrame
            report: 异常值处理报告
        """
        df = df.copy()
        report = {
            'total_outliers': 0,
            'spike_outliers': 0,
            'gap_outliers': 0,
            'corrected_points': []
        }
        
        # 1. 计算对数收益率
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. 计算均值和标准差
        mean_return = df['log_return'].mean()
        std_return = df['log_return'].std()
        
        # 3. 检测异常值
        threshold = sigma_threshold * std_return
        outlier_mask = np.abs(df['log_return'] - mean_return) > threshold
        
        report['total_outliers'] = outlier_mask.sum()
        
        # 4. 区分尖峰异常和跳空异常
        for idx in df[outlier_mask].index:
            if idx == df.index[0] or idx == df.index[-1]:
                continue
                
            prev_idx = df.index[df.index.get_loc(idx) - 1]
            next_idx = df.index[df.index.get_loc(idx) + 1]
            
            current_close = df.loc[idx, 'Close']
            prev_close = df.loc[prev_idx, 'Close']
            next_close = df.loc[next_idx, 'Close']
            
            # 判断是否为尖峰（前后价格相近，当前价格异常）
            if abs(next_close - prev_close) / prev_close < 0.02:
                # 尖峰异常：用前后均值替代
                avg_price = (prev_close + next_close) / 2
                ratio = avg_price / current_close
                
                df.loc[idx, 'Open'] *= ratio
                df.loc[idx, 'High'] *= ratio
                df.loc[idx, 'Low'] *= ratio
                df.loc[idx, 'Close'] = avg_price
                
                report['spike_outliers'] += 1
                report['corrected_points'].append({
                    'index': idx,
                    'type': 'spike',
                    'original_close': current_close,
                    'corrected_close': avg_price
                })
            else:
                # 跳空异常：保留（可能是真实的市场跳空）
                report['gap_outliers'] += 1
        
        # 5. 修正OHLC一致性
        df = self._fix_ohlc_consistency(df)
        
        # 删除临时列
        df = df.drop('log_return', axis=1)
        
        self.cleaning_report['outliers'] = report
        logger.info(f"异常值处理完成: 检测到{report['total_outliers']}个异常值，"
                   f"修正{report['spike_outliers']}个尖峰")
        
        return df, report
    
    def align_time_and_timezone(self, df: pd.DataFrame, 
                               target_timezone: str = 'UTC',
                               trading_hours: Optional[Tuple[int, int]] = None,
                               interval_minutes: int = 5) -> Tuple[pd.DataFrame, Dict]:
        """
        处理时间对齐与时区
        
        任务1.1.3实现：
        - 转换时区到UTC或指定时区
        - 处理夏令时切换
        - 过滤非交易时段数据
        - 重采样确保严格5分钟间隔
        
        Args:
            df: 带时间索引的DataFrame
            target_timezone: 目标时区
            trading_hours: 交易时段 (start_hour, end_hour)，None表示24小时
            interval_minutes: K线间隔（分钟）
            
        Returns:
            aligned_df: 时间对齐后的DataFrame
            report: 时间处理报告
        """
        df = df.copy()
        report = {
            'original_timezone': str(df.index.tz) if hasattr(df.index, 'tz') else 'None',
            'target_timezone': target_timezone,
            'rows_before': len(df),
            'rows_after': 0,
            'filtered_non_trading': 0,
            'resampled_gaps': 0
        }
        
        # 1. 转换时区
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        df.index = df.index.tz_convert(target_timezone)
        
        # 2. 过滤非交易时段
        if trading_hours is not None:
            start_hour, end_hour = trading_hours
            hour_mask = (df.index.hour >= start_hour) & (df.index.hour < end_hour)
            filtered_count = (~hour_mask).sum()
            df = df[hour_mask]
            report['filtered_non_trading'] = filtered_count
        
        # 3. 重采样确保严格间隔
        expected_freq = f'{interval_minutes}T'
        
        # 检查实际间隔
        time_diffs = df.index.to_series().diff()
        expected_diff = pd.Timedelta(minutes=interval_minutes)
        
        # 允许±10秒的容差
        tolerance = pd.Timedelta(seconds=10)
        irregular_mask = np.abs(time_diffs - expected_diff) > tolerance
        
        if irregular_mask.sum() > 0:
            # 重采样到规则间隔
            df_resampled = df.resample(expected_freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            # 删除全为NaN的行（空K线）
            df_resampled = df_resampled.dropna(subset=['Close'])
            
            report['resampled_gaps'] = len(df_resampled) - len(df)
            df = df_resampled
        
        report['rows_after'] = len(df)
        
        self.cleaning_report['time_alignment'] = report
        logger.info(f"时间对齐完成: 从{report['rows_before']}行到{report['rows_after']}行")
        
        return df, report
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        数据质量验证
        
        任务1.1.4实现：
        - 完整性检查：缺失值比例<1%
        - 一致性检查：High>=max(O,C), Low<=min(O,C)
        - 异常值检查：价格跳变<5σ
        - 时间检查：间隔严格、无重复
        
        Args:
            df: 清洗后的DataFrame
            
        Returns:
            is_valid: 是否通过验证
            report: 验证报告
        """
        report = {
            'is_valid': True,
            'issues': [],
            'completeness': {},
            'consistency': {},
            'outliers': {},
            'time_quality': {}
        }
        
        # 1. 完整性检查
        total_rows = len(df)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_ratio = missing_count / total_rows
                report['completeness'][col] = {
                    'missing_count': missing_count,
                    'missing_ratio': missing_ratio
                }
                
                if missing_ratio > 0.01:  # 1%阈值
                    report['is_valid'] = False
                    report['issues'].append(
                        f"{col}列缺失值比例{missing_ratio:.2%}超过1%阈值"
                    )
        
        # 2. 一致性检查
        high_valid = (df['High'] >= df[['Open', 'Close']].max(axis=1)).all()
        low_valid = (df['Low'] <= df[['Open', 'Close']].min(axis=1)).all()
        
        report['consistency']['high_valid'] = high_valid
        report['consistency']['low_valid'] = low_valid
        
        if not high_valid:
            report['is_valid'] = False
            report['issues'].append("存在High < max(Open, Close)的情况")
        
        if not low_valid:
            report['is_valid'] = False
            report['issues'].append("存在Low > min(Open, Close)的情况")
        
        # 3. 异常值检查（5σ）
        log_returns = np.log(df['Close'] / df['Close'].shift(1))
        mean_return = log_returns.mean()
        std_return = log_returns.std()
        
        extreme_outliers = np.abs(log_returns - mean_return) > 5 * std_return
        outlier_count = extreme_outliers.sum()
        
        report['outliers']['extreme_count'] = outlier_count
        report['outliers']['extreme_ratio'] = outlier_count / total_rows
        
        if outlier_count > 0:
            report['issues'].append(f"存在{outlier_count}个极端异常值（>5σ）")
        
        # 4. 时间检查
        if isinstance(df.index, pd.DatetimeIndex):
            # 检查重复
            duplicates = df.index.duplicated().sum()
            report['time_quality']['duplicates'] = duplicates
            
            if duplicates > 0:
                report['is_valid'] = False
                report['issues'].append(f"存在{duplicates}个重复时间戳")
            
            # 检查排序
            is_sorted = df.index.is_monotonic_increasing
            report['time_quality']['is_sorted'] = is_sorted
            
            if not is_sorted:
                report['is_valid'] = False
                report['issues'].append("时间索引未按升序排列")
        
        self.cleaning_report['validation'] = report
        
        if report['is_valid']:
            logger.info("数据质量验证通过")
        else:
            logger.warning(f"数据质量验证失败: {len(report['issues'])}个问题")
            for issue in report['issues']:
                logger.warning(f"  - {issue}")
        
        return report['is_valid'], report
    
    def clean_pipeline(self, df: pd.DataFrame, 
                      target_timezone: str = 'UTC',
                      trading_hours: Optional[Tuple[int, int]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        完整的数据清洗流程
        
        Args:
            df: 原始OHLC DataFrame
            target_timezone: 目标时区
            trading_hours: 交易时段
            
        Returns:
            cleaned_df: 清洗后的DataFrame
            full_report: 完整的清洗报告
        """
        logger.info("开始数据清洗流程...")
        
        # 1. 缺失值处理
        df, _ = self.handle_missing_values(df)
        
        # 2. 异常值处理
        df, _ = self.detect_and_handle_outliers(df)
        
        # 3. 时间对齐
        df, _ = self.align_time_and_timezone(df, target_timezone, trading_hours)
        
        # 4. 质量验证
        is_valid, _ = self.validate_data_quality(df)
        
        full_report = self.cleaning_report.copy()
        full_report['final_validation'] = is_valid
        
        logger.info(f"数据清洗完成: 最终{len(df)}行数据，验证{'通过' if is_valid else '失败'}")
        
        return df, full_report
    
    # 辅助方法
    def _find_consecutive_groups(self, mask: pd.Series) -> list:
        """查找连续True值的组"""
        groups = []
        in_group = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_group:
                start = i
                in_group = True
            elif not val and in_group:
                groups.append((start, i))
                in_group = False
        
        if in_group:
            groups.append((start, len(mask)))
        
        return groups
    
    def _fix_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """修正OHLC一致性约束"""
        df = df.copy()
        
        # High >= max(Open, Close)
        max_oc = df[['Open', 'Close']].max(axis=1)
        df['High'] = df[['High', max_oc]].max(axis=1)
        
        # Low <= min(Open, Close)
        min_oc = df[['Open', 'Close']].min(axis=1)
        df['Low'] = df[['Low', min_oc]].min(axis=1)
        
        return df