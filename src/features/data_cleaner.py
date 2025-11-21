"""
数据清洗模块 - 处理OHLC数据的缺失值和异常值

本模块提供高质量的数据清洗功能，确保数据满足后续模型训练的要求。

主要功能：
1. 缺失值处理：前向填充、零填充、线性插值
2. 异常值检测：3σ原则，区分尖峰和跳空
3. 质量验证：完整性、一致性、异常值、时间检查

设计理念：
- 极简原则：只处理数据质量问题，不做数据转换
- 职责分离：
  * IB Gateway负责：交易时段过滤（use_rth）、数据间隔保证（bar_size）、时区处理
  * DataCleaner负责：缺失值修复、异常值检测、质量验证
- 专注核心：让每个模块做好自己的事

性能要求：处理10万条数据<1秒
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging
import warnings
import time

# 配置日志
logger = logging.getLogger(__name__)

# 忽略pandas的FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)


class DataCleaner:
    """
    数据清洗类，专门处理OHLC数据中的质量问题
    
    核心功能：
    1. 缺失值处理（前向填充、零填充、线性插值）
    2. 异常值检测与处理（3σ原则，区分尖峰和跳空）
    3. 数据质量验证（完整性、一致性、异常值、时间检查）
    
    职责边界：
    - 本模块负责：数据质量修复和验证
    - IB Gateway负责：数据过滤、间隔保证、时区处理
    
    使用示例：
        >>> cleaner = DataCleaner(max_consecutive_missing=5)
        >>> cleaned_df, report = cleaner.clean_pipeline(df)
        >>> # 查看清洗报告
        >>> cleaner.print_report_summary()
    """
    
    def __init__(self, max_consecutive_missing: int = 5, 
                 interpolation_limit: int = 3,
                 sigma_threshold: float = 3.0):
        """
        初始化数据清洗器
        
        Args:
            max_consecutive_missing: 允许的最大连续缺失K线数量，默认5
            interpolation_limit: 线性插值的最大连续点数，默认3
            sigma_threshold: 异常值检测的σ阈值，默认3.0
            
        Raises:
            ValueError: 如果参数不合法
        """
        if max_consecutive_missing < 1:
            raise ValueError("max_consecutive_missing必须>=1")
        if interpolation_limit < 1:
            raise ValueError("interpolation_limit必须>=1")
        if sigma_threshold <= 0:
            raise ValueError("sigma_threshold必须>0")
            
        self.max_consecutive_missing = max_consecutive_missing
        self.interpolation_limit = interpolation_limit
        self.sigma_threshold = sigma_threshold
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
            
        Raises:
            ValueError: 如果输入数据不合法
        """
        start_time = time.time()
        
        # 参数验证
        if df is None or df.empty:
            raise ValueError("输入DataFrame不能为空")
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        df = df.copy()
        report = {
            'total_rows': len(df),
            'missing_before': {},
            'missing_after': {},
            'interpolated_points': {},
            'removed_segments': [],
            'removed_rows': 0,
            'processing_time': 0
        }
        
        # 1. 检测缺失值位置和连续性
        price_cols = ['Open', 'High', 'Low', 'Close']
        volume_col = 'Volume' if 'Volume' in df.columns else None
        
        # 记录处理前的缺失值
        for col in price_cols:
            report['missing_before'][col] = int(df[col].isna().sum())
        if volume_col:
            report['missing_before'][volume_col] = int(df[volume_col].isna().sum())
        
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
        
        # 创建保留数据的mask并删除长缺失段
        if segments_to_remove:
            keep_mask = np.ones(len(df), dtype=bool)
            for start, end in segments_to_remove:
                keep_mask[start:end] = False
            
            rows_before = len(df)
            df = df[keep_mask].copy()
            report['removed_rows'] = rows_before - len(df)
            
            logger.info(f"删除了{len(segments_to_remove)}个长缺失段，共{report['removed_rows']}行")
        
        # 4. 处理剩余的短期缺失
        # 价格数据：先尝试线性插值，再前向填充
        for col in price_cols:
            # 标记插值点（在插值前）
            interpolated_mask = df[col].isna().copy()
            
            # 线性插值（仅用于短期缺失，限制连续插值点数）
            df[col] = df[col].interpolate(
                method='linear', 
                limit=self.interpolation_limit,
                limit_direction='both'
            )
            
            # 前向填充剩余缺失（使用新API）
            df[col] = df[col].ffill()
            
            # 如果开头有缺失，使用后向填充
            df[col] = df[col].bfill()
            
            report['interpolated_points'][col] = int(interpolated_mask.sum())
        
        # 成交量数据：零填充
        if volume_col:
            df[volume_col] = df[volume_col].fillna(0)
            report['interpolated_points'][volume_col] = 0
        
        # 5. 记录处理后的缺失值
        for col in price_cols:
            report['missing_after'][col] = int(df[col].isna().sum())
        if volume_col:
            report['missing_after'][volume_col] = int(df[volume_col].isna().sum())
        
        # 6. 验证处理结果
        total_missing_after = sum(report['missing_after'].values())
        if total_missing_after > 0:
            logger.warning(f"警告: 处理后仍有{total_missing_after}个缺失值")
        
        # 7. 重置索引（如果删除了行）
        if report['removed_rows'] > 0:
            df = df.reset_index(drop=False)
            if 'index' in df.columns and isinstance(df['index'].iloc[0], (pd.Timestamp, datetime)):
                df = df.set_index('index')
        
        report['processing_time'] = time.time() - start_time
        self.cleaning_report['missing_values'] = report
        
        total_interpolated = sum(report['interpolated_points'].values())
        logger.info(
            f"缺失值处理完成: "
            f"删除{len(segments_to_remove)}个数据段({report['removed_rows']}行), "
            f"插值{total_interpolated}个点, "
            f"剩余缺失{total_missing_after}个, "
            f"耗时{report['processing_time']:.3f}秒"
        )
        
        return df, report
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, 
                                   sigma_threshold: float = None) -> Tuple[pd.DataFrame, Dict]:
        """
        检测并处理价格异常值
        
        任务1.1.2实现：
        - 使用3σ原则检测价格跳变
        - 区分真实跳空和数据错误
        - 修正OHLC一致性
        
        Args:
            df: OHLC DataFrame
            sigma_threshold: 异常值检测的σ阈值，None则使用初始化时的值
            
        Returns:
            cleaned_df: 清洗后的DataFrame
            report: 异常值处理报告
            
        Raises:
            ValueError: 如果输入数据不合法
        """
        start_time = time.time()
        
        # 参数验证
        if df is None or df.empty:
            raise ValueError("输入DataFrame不能为空")
        
        if sigma_threshold is None:
            sigma_threshold = self.sigma_threshold
        elif sigma_threshold <= 0:
            raise ValueError("sigma_threshold必须>0")
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        df = df.copy()
        report = {
            'total_outliers': 0,
            'spike_outliers': 0,
            'gap_outliers': 0,
            'corrected_points': [],
            'sigma_threshold': sigma_threshold,
            'processing_time': 0
        }
        
        # 1. 计算对数收益率（处理可能的除零和负值）
        close_prices = df['Close'].replace(0, np.nan)
        df['log_return'] = np.log(close_prices / close_prices.shift(1))
        
        # 2. 计算均值和标准差（排除NaN）
        valid_returns = df['log_return'].dropna()
        if len(valid_returns) < 10:
            logger.warning("有效收益率数据点太少，跳过异常值检测")
            df = df.drop('log_return', axis=1)
            report['processing_time'] = time.time() - start_time
            return df, report
        
        mean_return = valid_returns.mean()
        std_return = valid_returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            logger.warning("收益率标准差为0或NaN，跳过异常值检测")
            df = df.drop('log_return', axis=1)
            report['processing_time'] = time.time() - start_time
            return df, report
        
        # 3. 检测异常值
        threshold = sigma_threshold * std_return
        outlier_mask = np.abs(df['log_return'] - mean_return) > threshold
        outlier_mask = outlier_mask.fillna(False)
        
        report['total_outliers'] = int(outlier_mask.sum())
        report['mean_return'] = float(mean_return)
        report['std_return'] = float(std_return)
        report['threshold'] = float(threshold)
        
        # 4. 区分尖峰异常和跳空异常
        if report['total_outliers'] > 0:
            outlier_indices = df[outlier_mask].index.tolist()
            
            for idx in outlier_indices:
                # 跳过首尾数据点
                idx_loc = df.index.get_loc(idx)
                if idx_loc == 0 or idx_loc == len(df) - 1:
                    report['gap_outliers'] += 1
                    continue
                
                try:
                    prev_idx = df.index[idx_loc - 1]
                    next_idx = df.index[idx_loc + 1]
                    
                    current_close = df.loc[idx, 'Close']
                    prev_close = df.loc[prev_idx, 'Close']
                    next_close = df.loc[next_idx, 'Close']
                    
                    # 检查数据有效性
                    if any(np.isnan([current_close, prev_close, next_close])) or \
                       any(x <= 0 for x in [current_close, prev_close, next_close]):
                        continue
                    
                    # 判断是否为尖峰（前后价格相近，当前价格异常）
                    # 阈值2%：如果前后价格变化<2%，认为是尖峰
                    price_change_ratio = abs(next_close - prev_close) / prev_close
                    
                    if price_change_ratio < 0.02:
                        # 尖峰异常：用前后均值替代
                        avg_price = (prev_close + next_close) / 2
                        ratio = avg_price / current_close
                        
                        # 按比例调整所有价格
                        df.loc[idx, 'Open'] *= ratio
                        df.loc[idx, 'High'] *= ratio
                        df.loc[idx, 'Low'] *= ratio
                        df.loc[idx, 'Close'] = avg_price
                        
                        report['spike_outliers'] += 1
                        report['corrected_points'].append({
                            'index': str(idx),
                            'type': 'spike',
                            'original_close': float(current_close),
                            'corrected_close': float(avg_price),
                            'ratio': float(ratio)
                        })
                    else:
                        # 跳空异常：保留（可能是真实的市场跳空）
                        report['gap_outliers'] += 1
                        
                except Exception as e:
                    logger.warning(f"处理异常值时出错 at index {idx}: {e}")
                    continue
        
        # 5. 修正OHLC一致性
        df = self._fix_ohlc_consistency(df)
        
        # 删除临时列
        df = df.drop('log_return', axis=1)
        
        report['processing_time'] = time.time() - start_time
        self.cleaning_report['outliers'] = report
        
        logger.info(
            f"异常值处理完成: "
            f"检测到{report['total_outliers']}个异常值(>{sigma_threshold}σ), "
            f"修正{report['spike_outliers']}个尖峰, "
            f"保留{report['gap_outliers']}个跳空, "
            f"耗时{report['processing_time']:.3f}秒"
        )
        
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
            
        Raises:
            ValueError: 如果输入数据不合法
        """
        start_time = time.time()
        
        # 参数验证
        if df is None or df.empty:
            raise ValueError("输入DataFrame不能为空")
        
        report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'completeness': {},
            'consistency': {},
            'outliers': {},
            'time_quality': {},
            'summary': {},
            'processing_time': 0
        }
        
        # 1. 完整性检查
        total_rows = len(df)
        report['summary']['total_rows'] = total_rows
        
        required_cols = ['Open', 'High', 'Low', 'Close']
        optional_cols = ['Volume']
        
        for col in required_cols + optional_cols:
            if col in df.columns:
                missing_count = int(df[col].isna().sum())
                missing_ratio = missing_count / total_rows if total_rows > 0 else 0
                
                report['completeness'][col] = {
                    'missing_count': missing_count,
                    'missing_ratio': float(missing_ratio)
                }
                
                # 必需列的缺失值检查
                if col in required_cols and missing_ratio > 0.01:  # 1%阈值
                    report['is_valid'] = False
                    report['issues'].append(
                        f"{col}列缺失值比例{missing_ratio:.2%}超过1%阈值"
                    )
                elif missing_ratio > 0:
                    report['warnings'].append(
                        f"{col}列存在{missing_count}个缺失值({missing_ratio:.2%})"
                    )
        
        # 2. 一致性检查
        try:
            # High >= max(Open, Close)
            max_oc = df[['Open', 'Close']].max(axis=1)
            high_violations = (df['High'] < max_oc).sum()
            high_valid = high_violations == 0
            
            # Low <= min(Open, Close)
            min_oc = df[['Open', 'Close']].min(axis=1)
            low_violations = (df['Low'] > min_oc).sum()
            low_valid = low_violations == 0
            
            # High >= Low
            hl_violations = (df['High'] < df['Low']).sum()
            hl_valid = hl_violations == 0
            
            report['consistency'] = {
                'high_valid': bool(high_valid),
                'low_valid': bool(low_valid),
                'high_low_valid': bool(hl_valid),
                'high_violations': int(high_violations),
                'low_violations': int(low_violations),
                'hl_violations': int(hl_violations)
            }
            
            if not high_valid:
                report['is_valid'] = False
                report['issues'].append(
                    f"存在{high_violations}处High < max(Open, Close)的情况"
                )
            
            if not low_valid:
                report['is_valid'] = False
                report['issues'].append(
                    f"存在{low_violations}处Low > min(Open, Close)的情况"
                )
            
            if not hl_valid:
                report['is_valid'] = False
                report['issues'].append(
                    f"存在{hl_violations}处High < Low的情况"
                )
                
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            report['warnings'].append(f"一致性检查失败: {e}")
        
        # 3. 异常值检查（5σ）
        try:
            close_prices = df['Close'].replace(0, np.nan)
            log_returns = np.log(close_prices / close_prices.shift(1))
            valid_returns = log_returns.dropna()
            
            if len(valid_returns) >= 10:
                mean_return = valid_returns.mean()
                std_return = valid_returns.std()
                
                if std_return > 0 and not np.isnan(std_return):
                    extreme_outliers = np.abs(log_returns - mean_return) > 5 * std_return
                    outlier_count = int(extreme_outliers.sum())
                    outlier_ratio = outlier_count / total_rows if total_rows > 0 else 0
                    
                    report['outliers'] = {
                        'extreme_count': outlier_count,
                        'extreme_ratio': float(outlier_ratio),
                        'mean_return': float(mean_return),
                        'std_return': float(std_return),
                        'threshold_5sigma': float(5 * std_return)
                    }
                    
                    if outlier_count > 0:
                        report['warnings'].append(
                            f"存在{outlier_count}个极端异常值（>5σ，占比{outlier_ratio:.2%}）"
                        )
                else:
                    report['outliers'] = {'note': '标准差为0或无效，跳过异常值检查'}
            else:
                report['outliers'] = {'note': '数据点太少，跳过异常值检查'}
                
        except Exception as e:
            logger.error(f"异常值检查失败: {e}")
            report['warnings'].append(f"异常值检查失败: {e}")
        
        # 4. 时间检查
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                # 检查重复
                duplicates = int(df.index.duplicated().sum())
                
                # 检查排序
                is_sorted = bool(df.index.is_monotonic_increasing)
                
                # 检查时间间隔（如果有足够数据）
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    if len(time_diffs) > 0:
                        min_interval = time_diffs.min()
                        max_interval = time_diffs.max()
                        median_interval = time_diffs.median()
                    else:
                        min_interval = max_interval = median_interval = pd.Timedelta(0)
                else:
                    min_interval = max_interval = median_interval = pd.Timedelta(0)
                
                report['time_quality'] = {
                    'duplicates': duplicates,
                    'is_sorted': is_sorted,
                    'min_interval': str(min_interval),
                    'max_interval': str(max_interval),
                    'median_interval': str(median_interval)
                }
                
                if duplicates > 0:
                    report['is_valid'] = False
                    report['issues'].append(f"存在{duplicates}个重复时间戳")
                
                if not is_sorted:
                    report['is_valid'] = False
                    report['issues'].append("时间索引未按升序排列")
            else:
                report['time_quality'] = {'note': '索引不是DatetimeIndex'}
                report['warnings'].append("索引不是DatetimeIndex，跳过时间检查")
                
        except Exception as e:
            logger.error(f"时间检查失败: {e}")
            report['warnings'].append(f"时间检查失败: {e}")
        
        # 5. 生成摘要
        report['processing_time'] = time.time() - start_time
        report['summary'].update({
            'total_issues': len(report['issues']),
            'total_warnings': len(report['warnings']),
            'validation_passed': report['is_valid']
        })
        
        self.cleaning_report['validation'] = report
        
        # 输出日志
        if report['is_valid']:
            logger.info("✓ 数据质量验证通过")
            if report['warnings']:
                logger.info(f"  {len(report['warnings'])}个警告:")
                for warning in report['warnings']:
                    logger.info(f"    - {warning}")
        else:
            logger.warning(f"✗ 数据质量验证失败: {len(report['issues'])}个问题")
            for issue in report['issues']:
                logger.warning(f"  - {issue}")
            if report['warnings']:
                logger.info(f"  {len(report['warnings'])}个警告:")
                for warning in report['warnings']:
                    logger.info(f"    - {warning}")
        
        return report['is_valid'], report
    
    def clean_pipeline(self, df: pd.DataFrame,
                      sigma_threshold: float = None) -> Tuple[pd.DataFrame, Dict]:
        """
        完整的数据清洗流程
        
        按顺序执行：缺失值处理 → 异常值处理 → 质量验证
        
        注意：
        - 时区处理已移除（在数据加载时处理，见ib_historical_data.py）
        - 交易时段筛选已移除（使用IB的use_rth参数）
        - 重采样功能已移除（IB保证数据间隔严格）
        
        Args:
            df: 原始OHLC DataFrame
            sigma_threshold: 异常值检测阈值，None则使用初始化时的值
            
        Returns:
            cleaned_df: 清洗后的DataFrame
            full_report: 完整的清洗报告
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("开始数据清洗流程...")
        logger.info(f"输入数据: {len(df)}行 × {len(df.columns)}列")
        
        try:
            # 1. 缺失值处理
            logger.info("-" * 60)
            logger.info("步骤1/3: 缺失值处理")
            df, _ = self.handle_missing_values(df)
            
            # 2. 异常值处理
            logger.info("-" * 60)
            logger.info("步骤2/3: 异常值检测与处理")
            df, _ = self.detect_and_handle_outliers(df, sigma_threshold)
            
            # 3. 质量验证
            logger.info("-" * 60)
            logger.info("步骤3/3: 数据质量验证")
            is_valid, _ = self.validate_data_quality(df)
            
            # 生成完整报告
            full_report = self.cleaning_report.copy()
            full_report['final_validation'] = is_valid
            full_report['total_processing_time'] = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info(
                f"数据清洗完成: "
                f"最终{len(df)}行数据, "
                f"验证{'通过✓' if is_valid else '失败✗'}, "
                f"总耗时{full_report['total_processing_time']:.3f}秒"
            )
            logger.info("=" * 60)
            
            return df, full_report
            
        except Exception as e:
            logger.error(f"数据清洗流程失败: {e}")
            raise
    
    # 辅助方法
    def _find_consecutive_groups(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """
        查找连续True值的组
        
        Args:
            mask: 布尔Series
            
        Returns:
            连续组的列表，每个元素为(start_idx, end_idx)
        """
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
        """
        修正OHLC一致性约束
        
        确保：
        - High >= max(Open, Close)
        - Low <= min(Open, Close)
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            修正后的DataFrame
        """
        df = df.copy()
        
        # High >= max(Open, Close)
        max_oc = df[['Open', 'Close']].max(axis=1)
        df['High'] = pd.concat([df['High'], max_oc], axis=1).max(axis=1)
        
        # Low <= min(Open, Close)
        min_oc = df[['Open', 'Close']].min(axis=1)
        df['Low'] = pd.concat([df['Low'], min_oc], axis=1).min(axis=1)
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """
        获取完整的清洗报告
        
        Returns:
            包含所有清洗步骤详细信息的字典
        """
        return self.cleaning_report.copy()
    
    def print_report_summary(self):
        """打印清洗报告摘要"""
        if not self.cleaning_report:
            print("尚未执行任何清洗操作")
            return
        
        print("\n" + "=" * 60)
        print("数据清洗报告摘要")
        print("=" * 60)
        
        if 'missing_values' in self.cleaning_report:
            mv = self.cleaning_report['missing_values']
            print(f"\n【缺失值处理】")
            print(f"  删除数据段: {len(mv['removed_segments'])}个 ({mv['removed_rows']}行)")
            print(f"  插值点数: {sum(mv['interpolated_points'].values())}个")
            print(f"  剩余缺失: {sum(mv['missing_after'].values())}个")
        
        if 'outliers' in self.cleaning_report:
            ol = self.cleaning_report['outliers']
            print(f"\n【异常值处理】")
            print(f"  检测异常: {ol['total_outliers']}个")
            print(f"  修正尖峰: {ol['spike_outliers']}个")
            print(f"  保留跳空: {ol['gap_outliers']}个")
        
        
        if 'validation' in self.cleaning_report:
            vd = self.cleaning_report['validation']
            print(f"\n【质量验证】")
            print(f"  验证结果: {'通过✓' if vd['is_valid'] else '失败✗'}")
            print(f"  问题数: {vd['summary']['total_issues']}个")
            print(f"  警告数: {vd['summary']['total_warnings']}个")
        
        if 'total_processing_time' in self.cleaning_report:
            print(f"\n总耗时: {self.cleaning_report['total_processing_time']:.3f}秒")
        
        print("=" * 60 + "\n")