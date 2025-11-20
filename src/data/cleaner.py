"""
数据清洗模块
Data Cleaner Module

实现数据清洗功能，包括:
- TASK-021: 缺失值处理
- TASK-022: 价格异常值处理
- TASK-023: 成交量异常处理
- TASK-024: OHLC一致性修正
- TASK-025: 时间对齐功能
- TASK-026: 数据标准化
- TASK-027: 数据清洗管道
- TASK-028: 清洗前后对比
- TASK-029: 数据质量评分
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import pytz
import sys
import pickle
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.data.validator import DataValidator

logger = get_logger(__name__)


class MissingValueHandler:
    """
    TASK-021: 缺失值处理器
    
    支持多种缺失值处理策略:
    - 前向填充(ffill)
    - 线性插值
    - 删除缺失段(连续>5根K线)
    """
    
    def __init__(self, max_consecutive_missing: int = 5):
        """
        初始化缺失值处理器
        
        Args:
            max_consecutive_missing: 最大连续缺失数，超过则删除
        """
        self.max_consecutive_missing = max_consecutive_missing
        logger.info(f"缺失值处理器初始化: max_consecutive_missing={max_consecutive_missing}")
    
    def handle_missing(
        self,
        data: pd.DataFrame,
        method: str = 'ffill',
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理缺失值
        
        Args:
            data: 数据
            method: 处理方法 ('ffill', 'interpolate', 'drop')
            columns: 要处理的列，None表示所有数值列
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (处理后的数据, 处理报告)
        """
        df = data.copy()
        report = {
            'method': method,
            'original_missing': {},
            'final_missing': {},
            'rows_dropped': 0
        }
        
        # 确定要处理的列
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 记录原始缺失情况
        for col in columns:
            if col in df.columns:
                report['original_missing'][col] = df[col].isna().sum()
        
        if method == 'ffill':
            # 前向填充
            df[columns] = df[columns].fillna(method='ffill')
            # 如果开头有缺失，用后向填充
            df[columns] = df[columns].fillna(method='bfill')
            
        elif method == 'interpolate':
            # 线性插值
            for col in columns:
                if col in df.columns:
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        elif method == 'drop':
            # 删除连续缺失超过阈值的段
            df, dropped = self._drop_long_missing_segments(df, columns)
            report['rows_dropped'] = dropped
        
        # 记录最终缺失情况
        for col in columns:
            if col in df.columns:
                report['final_missing'][col] = df[col].isna().sum()
        
        total_original = sum(report['original_missing'].values())
        total_final = sum(report['final_missing'].values())
        report['missing_handled'] = total_original - total_final
        
        logger.info(
            f"缺失值处理完成: {method}, "
            f"处理 {report['missing_handled']} 个缺失值, "
            f"删除 {report['rows_dropped']} 行"
        )
        
        return df, report
    
    def _drop_long_missing_segments(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """删除连续缺失超过阈值的段"""
        df = data.copy()
        original_len = len(df)
        
        # 对每列检查连续缺失
        for col in columns:
            if col not in df.columns:
                continue
            
            # 标记缺失值
            is_missing = df[col].isna()
            
            # 找到连续缺失段
            missing_groups = (is_missing != is_missing.shift()).cumsum()
            
            # 计算每段的长度
            segment_lengths = is_missing.groupby(missing_groups).transform('sum')
            
            # 标记要删除的行（连续缺失超过阈值）
            to_drop = (is_missing) & (segment_lengths > self.max_consecutive_missing)
            
            # 删除这些行
            df = df[~to_drop]
        
        dropped = original_len - len(df)
        return df, dropped


class PriceAnomalyHandler:
    """
    TASK-022: 价格异常值处理器
    
    检测和修正价格尖峰(spike)，保留真实跳空(gap)
    """
    
    def __init__(self, spike_threshold: float = 5.0):
        """
        初始化价格异常处理器
        
        Args:
            spike_threshold: 尖峰检测阈值(标准差倍数)
        """
        self.spike_threshold = spike_threshold
        logger.info(f"价格异常处理器初始化: spike_threshold={spike_threshold}σ")
    
    def handle_price_anomalies(
        self,
        data: pd.DataFrame,
        price_col: str = 'close'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理价格异常值
        
        Args:
            data: 数据
            price_col: 价格列名
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (处理后的数据, 处理报告)
        """
        df = data.copy()
        report = {
            'spikes_detected': 0,
            'spikes_fixed': 0,
            'spike_indices': []
        }
        
        if price_col not in df.columns:
            logger.warning(f"列 {price_col} 不存在")
            return df, report
        
        # 计算收益率
        returns = df[price_col].pct_change()
        
        # 计算收益率的均值和标准差
        mean_return = returns.mean()
        std_return = returns.std()
        
        # 检测异常值（|ret - mean| > threshold * std）
        threshold = self.spike_threshold * std_return
        is_spike = np.abs(returns - mean_return) > threshold
        
        spike_indices = df.index[is_spike].tolist()
        report['spikes_detected'] = len(spike_indices)
        report['spike_indices'] = spike_indices
        
        # 修正尖峰：使用前后均值
        for idx in spike_indices:
            if idx > 0 and idx < len(df) - 1:
                # 使用前后值的均值
                prev_val = df.loc[idx - 1, price_col]
                next_val = df.loc[idx + 1, price_col]
                df.loc[idx, price_col] = (prev_val + next_val) / 2
                report['spikes_fixed'] += 1
        
        logger.info(
            f"价格异常处理完成: 检测到 {report['spikes_detected']} 个尖峰, "
            f"修正 {report['spikes_fixed']} 个"
        )
        
        return df, report


class VolumeAnomalyHandler:
    """
    TASK-023: 成交量异常处理器
    
    处理异常大成交量和零成交量
    """
    
    def __init__(self, volume_threshold: float = 3.0):
        """
        初始化成交量异常处理器
        
        Args:
            volume_threshold: 异常成交量阈值(MA+N*std)
        """
        self.volume_threshold = volume_threshold
        logger.info(f"成交量异常处理器初始化: threshold=MA+{volume_threshold}σ")
    
    def handle_volume_anomalies(
        self,
        data: pd.DataFrame,
        volume_col: str = 'volume',
        window: int = 20
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理成交量异常值
        
        Args:
            data: 数据
            volume_col: 成交量列名
            window: 移动平均窗口
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (处理后的数据, 处理报告)
        """
        df = data.copy()
        report = {
            'high_volume_anomalies': 0,
            'zero_volume_count': 0,
            'anomalies_fixed': 0
        }
        
        if volume_col not in df.columns:
            logger.warning(f"列 {volume_col} 不存在")
            return df, report
        
        # 计算移动平均和标准差
        ma = df[volume_col].rolling(window=window, min_periods=1).mean()
        std = df[volume_col].rolling(window=window, min_periods=1).std()
        
        # 检测异常大成交量
        upper_bound = ma + self.volume_threshold * std
        is_high_anomaly = df[volume_col] > upper_bound
        report['high_volume_anomalies'] = is_high_anomaly.sum()
        
        # Cap到合理范围
        df.loc[is_high_anomaly, volume_col] = upper_bound[is_high_anomaly]
        report['anomalies_fixed'] = is_high_anomaly.sum()
        
        # 处理零成交量（用前一个非零值填充）
        zero_volume = df[volume_col] == 0
        report['zero_volume_count'] = zero_volume.sum()
        
        if zero_volume.any():
            df.loc[zero_volume, volume_col] = df[volume_col].replace(0, np.nan).fillna(method='ffill')
        
        logger.info(
            f"成交量异常处理完成: "
            f"异常大成交量 {report['high_volume_anomalies']} 个, "
            f"零成交量 {report['zero_volume_count']} 个"
        )
        
        return df, report


class OHLCConsistencyFixer:
    """
    TASK-024: OHLC一致性修正器
    
    确保OHLC数据的逻辑一致性
    """
    
    def __init__(self):
        """初始化OHLC一致性修正器"""
        logger.info("OHLC一致性修正器初始化")
    
    def fix_ohlc_consistency(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        修正OHLC一致性
        
        Args:
            data: 数据
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (修正后的数据, 修正报告)
        """
        df = data.copy()
        report = {
            'high_low_fixes': 0,
            'high_oc_fixes': 0,
            'low_oc_fixes': 0
        }
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("缺少必要的OHLC列")
            return df, report
        
        # 1. 确保 High >= Low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            # 交换high和low
            df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values
            report['high_low_fixes'] = invalid_hl.sum()
        
        # 2. 确保 High >= max(Open, Close)
        max_oc = df[['open', 'close']].max(axis=1)
        invalid_high = df['high'] < max_oc
        if invalid_high.any():
            df.loc[invalid_high, 'high'] = max_oc[invalid_high]
            report['high_oc_fixes'] = invalid_high.sum()
        
        # 3. 确保 Low <= min(Open, Close)
        min_oc = df[['open', 'close']].min(axis=1)
        invalid_low = df['low'] > min_oc
        if invalid_low.any():
            df.loc[invalid_low, 'low'] = min_oc[invalid_low]
            report['low_oc_fixes'] = invalid_low.sum()
        
        total_fixes = sum(report.values())
        logger.info(f"OHLC一致性修正完成: 共修正 {total_fixes} 处不一致")
        
        return df, report


class TimeAligner:
    """
    TASK-025: 时间对齐功能
    
    重采样到指定时间间隔，确保时间连续性
    """
    
    def __init__(self, target_freq: str = '5min'):
        """
        初始化时间对齐器
        
        Args:
            target_freq: 目标频率
        """
        self.target_freq = target_freq
        logger.info(f"时间对齐器初始化: target_freq={target_freq}")
    
    def align_time(
        self,
        data: pd.DataFrame,
        datetime_col: str = 'datetime'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        对齐时间
        
        Args:
            data: 数据
            datetime_col: 时间列名
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (对齐后的数据, 对齐报告)
        """
        df = data.copy()
        report = {
            'original_records': len(df),
            'aligned_records': 0,
            'empty_bars_removed': 0
        }
        
        if datetime_col not in df.columns:
            logger.warning(f"列 {datetime_col} 不存在")
            return df, report
        
        # 设置datetime为索引
        if df.index.name != datetime_col:
            df = df.set_index(datetime_col)
        
        # 重采样OHLCV数据
        resampled = pd.DataFrame()
        
        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(self.target_freq).first()
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(self.target_freq).max()
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(self.target_freq).min()
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(self.target_freq).last()
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(self.target_freq).sum()
        
        # 删除空K线（所有值都是NaN的行）
        before_drop = len(resampled)
        resampled = resampled.dropna(how='all')
        after_drop = len(resampled)
        
        report['aligned_records'] = after_drop
        report['empty_bars_removed'] = before_drop - after_drop
        
        # 重置索引
        resampled = resampled.reset_index()
        resampled = resampled.rename(columns={'index': datetime_col})
        
        logger.info(
            f"时间对齐完成: {report['original_records']} -> {report['aligned_records']} 条记录, "
            f"删除 {report['empty_bars_removed']} 个空K线"
        )
        
        return resampled, report


class DataNormalizer:
    """
    TASK-026: 数据标准化器
    
    支持保存和加载归一化参数
    """
    
    def __init__(self, scaler_dir: Optional[Path] = None):
        """
        初始化数据标准化器
        
        Args:
            scaler_dir: scaler保存目录
        """
        if scaler_dir is None:
            scaler_dir = project_root / "scalers"
        self.scaler_dir = Path(scaler_dir)
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        
        self.scalers = {}
        logger.info(f"数据标准化器初始化: scaler_dir={self.scaler_dir}")
    
    def normalize(
        self,
        data: pd.DataFrame,
        method: str = 'standard',
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        标准化数据
        
        Args:
            data: 数据
            method: 标准化方法 ('standard', 'minmax', 'robust')
            columns: 要标准化的列
            fit: 是否拟合scaler
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (标准化后的数据, 标准化参数)
        """
        df = data.copy()
        params = {'method': method, 'columns': {}}
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
            columns = [col for col in columns if col in df.columns]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit:
                # 拟合并转换
                if method == 'standard':
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = (df[col] - mean) / std if std > 0 else 0
                    params['columns'][col] = {'mean': float(mean), 'std': float(std)}
                    self.scalers[col] = {'method': method, 'mean': mean, 'std': std}
                
                elif method == 'minmax':
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val) if max_val > min_val else 0
                    params['columns'][col] = {'min': float(min_val), 'max': float(max_val)}
                    self.scalers[col] = {'method': method, 'min': min_val, 'max': max_val}
                
                elif method == 'robust':
                    median = df[col].median()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    df[col] = (df[col] - median) / iqr if iqr > 0 else 0
                    params['columns'][col] = {'median': float(median), 'iqr': float(iqr)}
                    self.scalers[col] = {'method': method, 'median': median, 'iqr': iqr}
            else:
                # 使用已有的scaler转换
                if col in self.scalers:
                    scaler = self.scalers[col]
                    if scaler['method'] == 'standard':
                        df[col] = (df[col] - scaler['mean']) / scaler['std']
                    elif scaler['method'] == 'minmax':
                        df[col] = (df[col] - scaler['min']) / (scaler['max'] - scaler['min'])
                    elif scaler['method'] == 'robust':
                        df[col] = (df[col] - scaler['median']) / scaler['iqr']
        
        logger.info(f"数据标准化完成: {method}, {len(columns)} 列")
        
        return df, params
    
    def save_scalers(self, filename: str = 'scalers.pkl'):
        """保存scaler参数"""
        filepath = self.scaler_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Scaler参数已保存: {filepath}")
    
    def load_scalers(self, filename: str = 'scalers.pkl'):
        """加载scaler参数"""
        filepath = self.scaler_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.scalers = pickle.load(f)
            logger.info(f"Scaler参数已加载: {filepath}")
        else:
            logger.warning(f"Scaler文件不存在: {filepath}")
    
    def inverse_transform(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """反标准化"""
        df = data.copy()
        
        if columns is None:
            columns = list(self.scalers.keys())
        
        for col in columns:
            if col not in df.columns or col not in self.scalers:
                continue
            
            scaler = self.scalers[col]
            if scaler['method'] == 'standard':
                df[col] = df[col] * scaler['std'] + scaler['mean']
            elif scaler['method'] == 'minmax':
                df[col] = df[col] * (scaler['max'] - scaler['min']) + scaler['min']
            elif scaler['method'] == 'robust':
                df[col] = df[col] * scaler['iqr'] + scaler['median']
        
        return df


class DataCleaningPipeline:
    """
    TASK-027: 数据清洗管道
    
    串联所有清洗步骤，提供统一的清洗接口
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化清洗管道
        
        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 初始化各个处理器
        self.missing_handler = MissingValueHandler()
        self.price_handler = PriceAnomalyHandler()
        self.volume_handler = VolumeAnomalyHandler()
        self.ohlc_fixer = OHLCConsistencyFixer()
        self.time_aligner = TimeAligner()
        self.normalizer = DataNormalizer()
        
        logger.info("数据清洗管道初始化完成")
    
    def clean(
        self,
        data: pd.DataFrame,
        symbol: str = "Unknown",
        handle_missing: bool = True,
        fix_price_anomalies: bool = True,
        fix_volume_anomalies: bool = True,
        fix_ohlc: bool = True,
        align_time: bool = True,
        normalize: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行完整的数据清洗流程
        
        Args:
            data: 原始数据
            symbol: 品种代码
            handle_missing: 是否处理缺失值
            fix_price_anomalies: 是否修正价格异常
            fix_volume_anomalies: 是否修正成交量异常
            fix_ohlc: 是否修正OHLC一致性
            align_time: 是否对齐时间
            normalize: 是否标准化
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (清洗后的数据, 清洗报告)
        """
        logger.info(f"开始清洗数据: {symbol}, {len(data)} 条记录")
        
        report = {
            'symbol': symbol,
            'original_records': len(data),
            'steps': [],
            'final_records': 0
        }
        
        cleaned_data = data.copy()
        
        # 1. 处理缺失值
        if handle_missing:
            cleaned_data, missing_report = self.missing_handler.handle_missing(
                cleaned_data, method='ffill'
            )
            report['steps'].append({'step': 'missing_values', 'report': missing_report})
        
        # 2. 修正OHLC一致性
        if fix_ohlc:
            cleaned_data, ohlc_report = self.ohlc_fixer.fix_ohlc_consistency(cleaned_data)
            report['steps'].append({'step': 'ohlc_consistency', 'report': ohlc_report})
        
        # 3. 修正价格异常
        if fix_price_anomalies:
            cleaned_data, price_report = self.price_handler.handle_price_anomalies(cleaned_data)
            report['steps'].append({'step': 'price_anomalies', 'report': price_report})
        
        # 4. 修正成交量异常
        if fix_volume_anomalies:
            cleaned_data, volume_report = self.volume_handler.handle_volume_anomalies(cleaned_data)
            report['steps'].append({'step': 'volume_anomalies', 'report': volume_report})
        
        # 5. 时间对齐
        if align_time and 'datetime' in cleaned_data.columns:
            cleaned_data, align_report = self.time_aligner.align_time(cleaned_data)
            report['steps'].append({'step': 'time_alignment', 'report': align_report})
        
        # 6. 数据标准化
        if normalize:
            cleaned_data, norm_params = self.normalizer.normalize(cleaned_data)
            report['steps'].append({'step': 'normalization', 'params': norm_params})
        
        report['final_records'] = len(cleaned_data)
        report['removed_records'] = report['original_records'] - report['final_records']
        
        logger.info(
            f"清洗完成: {symbol}, "
            f"{report['final_records']}/{report['original_records']} 条记录"
        )
        
        return cleaned_data, report


class DataQualityComparator:
    """
    TASK-028: 清洗前后对比器
    
    统计和可视化清洗前后的差异
    """
    
    def __init__(self):
        """初始化对比器"""
        logger.info("数据质量对比器初始化")
    
    def compare(
        self,
        before: pd.DataFrame,
        after: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        对比清洗前后的数据
        
        Args:
            before: 清洗前的数据
            after: 清洗后的数据
        
        Returns:
            Dict: 对比报告
        """
        report = {
            'record_count': {
                'before': len(before),
                'after': len(after),
                'change': len(after) - len(before)
            },
            'missing_values': {},
            'statistics': {}
        }
        
        # 对比缺失值
        numeric_cols = before.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in before.columns and col in after.columns:
                report['missing_values'][col] = {
                    'before': int(before[col].isna().sum()),
                    'after': int(after[col].isna().sum())
                }
        
        # 对比统计量
        for col in numeric_cols:
            if col in before.columns and col in after.columns:
                report['statistics'][col] = {
                    'before': {
                        'mean': float(before[col].mean()),
                        'std': float(before[col].std()),
                        'min': float(before[col].min()),
                        'max': float(before[col].max())
                    },
                    'after': {
                        'mean': float(after[col].mean()),
                        'std': float(after[col].std()),
                        'min': float(after[col].min()),
                        'max': float(after[col].max())
                    }
                }
        
        logger.info("数据对比完成")
        
        return report
    
    def generate_report(
        self,
        comparison: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> str:
        """
        生成对比报告
        
        Args:
            comparison: 对比结果
            output_path: 输出路径
        
        Returns:
            str: 报告内容
        """
        report_lines = [
            "=" * 60,
            "数据清洗前后对比报告",
            "=" * 60,
            "",
            "1. 记录数量变化:",
            f"   清洗前: {comparison['record_count']['before']} 条",
            f"   清洗后: {comparison['record_count']['after']} 条",
            f"   变化: {comparison['record_count']['change']:+d} 条",
            "",
            "2. 缺失值变化:",
        ]
        
        for col, values in comparison['missing_values'].items():
            report_lines.append(
                f"   {col}: {values['before']} -> {values['after']} "
                f"({values['after'] - values['before']:+d})"
            )
        
        report_lines.extend(["", "3. 统计量变化:"])
        for col, stats in comparison['statistics'].items():
            report_lines.extend([
                f"   {col}:",
                f"     均值: {stats['before']['mean']:.4f} -> {stats['after']['mean']:.4f}",
                f"     标准差: {stats['before']['std']:.4f} -> {stats['after']['std']:.4f}",
            ])
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"对比报告已保存: {output_path}")
        
        return report_text


class DataQualityScorer:
    """
    TASK-029: 数据质量评分器
    
    计算综合质量分数
    """
    
    def __init__(self):
        """初始化质量评分器"""
        self.weights = {
            'completeness': 0.3,  # 完整性
            'consistency': 0.3,   # 一致性
            'validity': 0.2,      # 有效性
            'timeliness': 0.2     # 时效性
        }
        logger.info("数据质量评分器初始化")
    
    def score(
        self,
        data: pd.DataFrame,
        datetime_col: str = 'datetime'
    ) -> Dict[str, Any]:
        """
        计算数据质量分数
        
        Args:
            data: 数据
            datetime_col: 时间列名
        
        Returns:
            Dict: 质量评分报告
        """
        scores = {}
        
        # 1. 完整性评分 (0-100)
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        scores['completeness'] = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # 2. 一致性评分 (0-100)
        consistency_score = 100
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # 检查OHLC一致性
            invalid_hl = (data['high'] < data['low']).sum()
            max_oc = data[['open', 'close']].max(axis=1)
            invalid_high = (data['high'] < max_oc).sum()
            min_oc = data[['open', 'close']].min(axis=1)
            invalid_low = (data['low'] > min_oc).sum()
            
            total_invalid = invalid_hl + invalid_high + invalid_low
            consistency_score = max(0, 100 - (total_invalid / len(data)) * 100)
        
        scores['consistency'] = consistency_score
        
        # 3. 有效性评分 (0-100)
        validity_score = 100
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # 检查异常值（超过3σ）
            if len(data[col].dropna()) > 0:
                mean = data[col].mean()
                std = data[col].std()
                outliers = np.abs(data[col] - mean) > 3 * std
                validity_score -= (outliers.sum() / len(data)) * 10
        
        scores['validity'] = max(0, validity_score)
        
        # 4. 时效性评分 (0-100)
        timeliness_score = 100
        if datetime_col in data.columns:
            # 检查时间间隔的一致性
            if len(data) > 1:
                time_diffs = data[datetime_col].diff().dropna()
                if len(time_diffs) > 0:
                    # 计算时间间隔的变异系数
                    mean_diff = time_diffs.mean()
                    std_diff = time_diffs.std()
                    cv = std_diff / mean_diff if mean_diff.total_seconds() > 0 else 0
                    timeliness_score = max(0, 100 - cv * 100)
        
        scores['timeliness'] = timeliness_score
        
        # 计算综合分数
        total_score = sum(
            scores[key] * self.weights[key]
            for key in self.weights.keys()
        )
        
        report = {
            'total_score': round(total_score, 2),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'weights': self.weights,
            'grade': self._get_grade(total_score)
        }
        
        logger.info(f"数据质量评分完成: {report['total_score']:.2f} ({report['grade']})")
        
        return report
    
    def _get_grade(self, score: float) -> str:
        """根据分数获取等级"""
        if score >= 90:
            return 'A (优秀)'
        elif score >= 80:
            return 'B (良好)'
        elif score >= 70:
            return 'C (中等)'
        elif score >= 60:
            return 'D (及格)'
        else:
            return 'F (不及格)'


def main():
    """测试函数"""
    print("\n" + "=" * 60)
    print("数据清洗模块测试")
    print("=" * 60)
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='5min')
    
    test_data = pd.DataFrame({
        'datetime': dates,
        'open': 100 + np.random.randn(n) * 2,
        'high': 102 + np.random.randn(n) * 2,
        'low': 98 + np.random.randn(n) * 2,
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # 人为添加一些问题
    test_data.loc[10:12, 'close'] = np.nan  # 缺失值
    test_data.loc[20, 'close'] = 200  # 价格尖峰
    test_data.loc[30, 'volume'] = 100000  # 成交量异常
    test_data.loc[40, 'high'] = test_data.loc[40, 'low'] - 1  # OHLC不一致
    
    print(f"测试数据: {len(test_data)} 条记录")
    
    # 测试清洗管道
    print("\n2. 测试数据清洗管道...")
    pipeline = DataCleaningPipeline()
    
    # 保存清洗前的数据
    before_data = test_data.copy()
    
    # 执行清洗
    cleaned_data, report = pipeline.clean(
        test_data,
        symbol="TEST",
        handle_missing=True,
        fix_price_anomalies=True,
        fix_volume_anomalies=True,
        fix_ohlc=True,
        align_time=False,
        normalize=False
    )
    
    print(f"\n清洗报告:")
    print(f"  原始记录: {report['original_records']}")
    print(f"  最终记录: {report['final_records']}")
    print(f"  处理步骤: {len(report['steps'])}")
    
    # 测试清洗前后对比
    print("\n3. 测试清洗前后对比...")
    comparator = DataQualityComparator()
    comparison = comparator.compare(before_data, cleaned_data)
    report_text = comparator.generate_report(comparison)
    print(report_text)
    
    # 测试数据质量评分
    print("\n4. 测试数据质量评分...")
    scorer = DataQualityScorer()
    
    print("\n清洗前质量评分:")
    before_score = scorer.score(before_data)
    print(f"  总分: {before_score['total_score']:.2f} {before_score['grade']}")
    for key, value in before_score['scores'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\n清洗后质量评分:")
    after_score = scorer.score(cleaned_data)
    print(f"  总分: {after_score['total_score']:.2f} {after_score['grade']}")
    for key, value in after_score['scores'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()