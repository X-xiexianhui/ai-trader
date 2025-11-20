"""
数据验证器模块
Data Validator Module

实现数据完整性检查和异常值检测功能
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataValidator:
    """
    数据验证器
    
    功能:
    - 检查缺失值比例
    - 检查时间间隔连续性
    - 检查OHLC数据一致性
    - 检测价格异常值
    - 检测成交量异常值
    - 生成数据质量报告
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化验证器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 数据质量配置
        self.max_missing_ratio = self.config.get('quality.missing_values.max_missing_ratio', 0.01)
        self.max_consecutive_missing = self.config.get('quality.missing_values.max_consecutive_missing', 5)
        
        # 异常值检测配置
        self.price_jump_threshold = self.config.get('quality.outliers.price_jump.threshold_sigma', 5)
        self.volume_spike_threshold = self.config.get('quality.outliers.volume_spike.threshold_multiplier', 10)
        self.volume_ma_window = self.config.get('quality.outliers.volume_spike.ma_window', 20)
        
        # OHLC一致性配置
        self.ohlc_check_enabled = self.config.get('quality.ohlc_consistency.enabled', True)
        self.ohlc_auto_fix = self.config.get('quality.ohlc_consistency.auto_fix', True)
        
        logger.info("数据验证器初始化完成")
    
    def validate(
        self,
        data: pd.DataFrame,
        symbol: str = "Unknown",
        fix_issues: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        执行完整的数据验证
        
        Args:
            data: 要验证的数据
            symbol: 品种代码（用于日志）
            fix_issues: 是否自动修复问题
        
        Returns:
            Tuple[pd.DataFrame, Dict]: (修复后的数据, 验证报告)
        """
        logger.info(f"开始验证数据: {symbol}, {len(data)} 条记录")
        
        report = {
            'symbol': symbol,
            'total_records': len(data),
            'checks': {},
            'issues_found': [],
            'issues_fixed': [],
            'quality_score': 100.0
        }
        
        # 复制数据以避免修改原始数据
        validated_data = data.copy()
        
        # 1. 检查缺失值
        missing_check = self.check_missing_values(validated_data)
        report['checks']['missing_values'] = missing_check
        if not missing_check['passed']:
            report['issues_found'].append('缺失值超标')
            if fix_issues:
                validated_data = self.fix_missing_values(validated_data)
                report['issues_fixed'].append('缺失值已修复')
        
        # 2. 检查时间连续性
        time_check = self.check_time_continuity(validated_data)
        report['checks']['time_continuity'] = time_check
        if not time_check['passed']:
            report['issues_found'].append('时间不连续')
        
        # 3. 检查OHLC一致性
        if self.ohlc_check_enabled:
            ohlc_check = self.check_ohlc_consistency(validated_data)
            report['checks']['ohlc_consistency'] = ohlc_check
            if not ohlc_check['passed']:
                report['issues_found'].append('OHLC数据不一致')
                if fix_issues and self.ohlc_auto_fix:
                    validated_data = self.fix_ohlc_consistency(validated_data)
                    report['issues_fixed'].append('OHLC数据已修复')
        
        # 4. 检测价格异常值
        price_outliers = self.detect_price_outliers(validated_data)
        report['checks']['price_outliers'] = price_outliers
        if price_outliers['outlier_count'] > 0:
            report['issues_found'].append(f'发现{price_outliers["outlier_count"]}个价格异常值')
            if fix_issues:
                validated_data = self.fix_price_outliers(validated_data, price_outliers['outlier_indices'])
                report['issues_fixed'].append('价格异常值已修复')
        
        # 5. 检测成交量异常值
        volume_outliers = self.detect_volume_outliers(validated_data)
        report['checks']['volume_outliers'] = volume_outliers
        if volume_outliers['outlier_count'] > 0:
            report['issues_found'].append(f'发现{volume_outliers["outlier_count"]}个成交量异常值')
            if fix_issues:
                validated_data = self.fix_volume_outliers(validated_data, volume_outliers['outlier_indices'])
                report['issues_fixed'].append('成交量异常值已修复')
        
        # 计算质量分数
        report['quality_score'] = self._calculate_quality_score(report)
        
        logger.info(f"验证完成: {symbol}, 质量分数: {report['quality_score']:.2f}")
        
        return validated_data, report
    
    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查缺失值
        
        Args:
            data: 数据
        
        Returns:
            Dict: 检查结果
        """
        result = {
            'passed': True,
            'total_missing': 0,
            'missing_ratio': 0.0,
            'columns_with_missing': {},
            'max_consecutive_missing': 0
        }
        
        if data.empty:
            return result
        
        # 检查每列的缺失值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    result['columns_with_missing'][col] = {
                        'count': int(missing_count),
                        'ratio': float(missing_count / len(data))
                    }
        
        # 计算总缺失值
        result['total_missing'] = sum(
            info['count'] for info in result['columns_with_missing'].values()
        )
        result['missing_ratio'] = result['total_missing'] / (len(data) * 5) if len(data) > 0 else 0
        
        # 检查连续缺失
        if 'close' in data.columns:
            is_missing = data['close'].isna()
            consecutive_missing = []
            count = 0
            for val in is_missing:
                if val:
                    count += 1
                else:
                    if count > 0:
                        consecutive_missing.append(count)
                    count = 0
            if count > 0:
                consecutive_missing.append(count)
            
            result['max_consecutive_missing'] = max(consecutive_missing) if consecutive_missing else 0
        
        # 判断是否通过
        result['passed'] = (
            result['missing_ratio'] <= self.max_missing_ratio and
            result['max_consecutive_missing'] <= self.max_consecutive_missing
        )
        
        return result
    
    def check_time_continuity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查时间连续性
        
        Args:
            data: 数据
        
        Returns:
            Dict: 检查结果
        """
        result = {
            'passed': True,
            'gaps_found': 0,
            'gap_details': []
        }
        
        if 'datetime' not in data.columns or len(data) < 2:
            return result
        
        # 计算时间间隔
        time_diffs = data['datetime'].diff()
        
        # 获取最常见的时间间隔（模式）
        mode_diff = time_diffs.mode()
        if len(mode_diff) == 0:
            return result
        
        expected_diff = mode_diff.iloc[0]
        
        # 查找异常间隔（超过预期的2倍）
        threshold = expected_diff * 2
        gaps = time_diffs[time_diffs > threshold]
        
        result['gaps_found'] = len(gaps)
        result['passed'] = result['gaps_found'] == 0
        
        # 记录间隔详情（最多10个）
        for idx in gaps.index[:10]:
            result['gap_details'].append({
                'index': int(idx),
                'datetime': str(data.loc[idx, 'datetime']),
                'gap_size': str(time_diffs.loc[idx])
            })
        
        return result
    
    def check_ohlc_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查OHLC数据一致性
        
        Args:
            data: 数据
        
        Returns:
            Dict: 检查结果
        """
        result = {
            'passed': True,
            'inconsistent_count': 0,
            'inconsistent_indices': []
        }
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return result
        
        # 检查 High >= max(Open, Close)
        high_check = data['high'] >= data[['open', 'close']].max(axis=1)
        
        # 检查 Low <= min(Open, Close)
        low_check = data['low'] <= data[['open', 'close']].min(axis=1)
        
        # 检查 High >= Low
        high_low_check = data['high'] >= data['low']
        
        # 找出不一致的行
        inconsistent = ~(high_check & low_check & high_low_check)
        inconsistent_indices = data[inconsistent].index.tolist()
        
        result['inconsistent_count'] = len(inconsistent_indices)
        result['inconsistent_indices'] = inconsistent_indices[:100]  # 最多记录100个
        result['passed'] = result['inconsistent_count'] == 0
        
        return result
    
    def detect_price_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测价格异常值
        
        Args:
            data: 数据
        
        Returns:
            Dict: 检测结果
        """
        result = {
            'outlier_count': 0,
            'outlier_indices': [],
            'outlier_details': []
        }
        
        if 'close' not in data.columns or len(data) < 2:
            return result
        
        # 计算收益率
        returns = data['close'].pct_change()
        
        # 计算均值和标准差
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0 or np.isnan(std_return):
            return result
        
        # 检测异常值（超过N倍标准差）
        threshold = self.price_jump_threshold * std_return
        outliers = np.abs(returns - mean_return) > threshold
        
        outlier_indices = data[outliers].index.tolist()
        
        result['outlier_count'] = len(outlier_indices)
        result['outlier_indices'] = outlier_indices
        
        # 记录异常值详情（最多10个）
        for idx in outlier_indices[:10]:
            result['outlier_details'].append({
                'index': int(idx),
                'datetime': str(data.loc[idx, 'datetime']) if 'datetime' in data.columns else None,
                'price': float(data.loc[idx, 'close']),
                'return': float(returns.loc[idx]) if idx in returns.index else None
            })
        
        return result
    
    def detect_volume_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测成交量异常值
        
        Args:
            data: 数据
        
        Returns:
            Dict: 检测结果
        """
        result = {
            'outlier_count': 0,
            'outlier_indices': [],
            'zero_volume_count': 0
        }
        
        if 'volume' not in data.columns:
            return result
        
        # 检测零成交量
        zero_volume = data['volume'] == 0
        result['zero_volume_count'] = int(zero_volume.sum())
        
        # 计算移动平均
        ma = data['volume'].rolling(window=self.volume_ma_window, min_periods=1).mean()
        
        # 检测异常大成交量
        threshold = ma * self.volume_spike_threshold
        outliers = data['volume'] > threshold
        
        outlier_indices = data[outliers].index.tolist()
        
        result['outlier_count'] = len(outlier_indices)
        result['outlier_indices'] = outlier_indices
        
        return result
    
    def fix_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        修复缺失值
        
        Args:
            data: 数据
        
        Returns:
            pd.DataFrame: 修复后的数据
        """
        df = data.copy()
        
        # 对价格数据使用前向填充
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                # 如果还有缺失值，使用后向填充
                df[col] = df[col].fillna(method='bfill')
        
        # 对成交量使用0填充
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # 删除仍然有缺失值的行（连续缺失超过阈值）
        if 'close' in df.columns:
            is_missing = df['close'].isna()
            consecutive_missing = []
            indices_to_drop = []
            count = 0
            start_idx = None
            
            for idx, val in enumerate(is_missing):
                if val:
                    if count == 0:
                        start_idx = idx
                    count += 1
                else:
                    if count > self.max_consecutive_missing:
                        indices_to_drop.extend(range(start_idx, idx))
                    count = 0
            
            if count > self.max_consecutive_missing:
                indices_to_drop.extend(range(start_idx, len(df)))
            
            if indices_to_drop:
                df = df.drop(df.index[indices_to_drop])
                logger.info(f"删除了{len(indices_to_drop)}行连续缺失数据")
        
        return df
    
    def fix_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        修复OHLC数据不一致
        
        Args:
            data: 数据
        
        Returns:
            pd.DataFrame: 修复后的数据
        """
        df = data.copy()
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return df
        
        # 确保 High >= max(Open, Close)
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        
        # 确保 Low <= min(Open, Close)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # 确保 High >= Low
        mask = df['high'] < df['low']
        if mask.any():
            # 交换high和low
            df.loc[mask, ['high', 'low']] = df.loc[mask, ['low', 'high']].values
        
        return df
    
    def fix_price_outliers(self, data: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        """
        修复价格异常值
        
        Args:
            data: 数据
            outlier_indices: 异常值索引列表
        
        Returns:
            pd.DataFrame: 修复后的数据
        """
        df = data.copy()
        
        if not outlier_indices or 'close' not in df.columns:
            return df
        
        # 使用前后均值替代异常值
        for idx in outlier_indices:
            if idx > 0 and idx < len(df) - 1:
                prev_val = df.loc[idx - 1, 'close']
                next_val = df.loc[idx + 1, 'close']
                df.loc[idx, 'close'] = (prev_val + next_val) / 2
                
                # 同时调整OHLC
                if all(col in df.columns for col in ['open', 'high', 'low']):
                    df.loc[idx, 'open'] = df.loc[idx, 'close']
                    df.loc[idx, 'high'] = df.loc[idx, 'close']
                    df.loc[idx, 'low'] = df.loc[idx, 'close']
        
        logger.info(f"修复了{len(outlier_indices)}个价格异常值")
        return df
    
    def fix_volume_outliers(self, data: pd.DataFrame, outlier_indices: List[int]) -> pd.DataFrame:
        """
        修复成交量异常值
        
        Args:
            data: 数据
            outlier_indices: 异常值索引列表
        
        Returns:
            pd.DataFrame: 修复后的数据
        """
        df = data.copy()
        
        if not outlier_indices or 'volume' not in df.columns:
            return df
        
        # 计算移动平均
        ma = df['volume'].rolling(window=self.volume_ma_window, min_periods=1).mean()
        
        # 将异常值替换为移动平均的3倍（保留一定的波动性）
        for idx in outlier_indices:
            df.loc[idx, 'volume'] = ma.loc[idx] * 3
        
        logger.info(f"修复了{len(outlier_indices)}个成交量异常值")
        return df
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """
        计算数据质量分数
        
        Args:
            report: 验证报告
        
        Returns:
            float: 质量分数 (0-100)
        """
        score = 100.0
        
        # 缺失值扣分
        if 'missing_values' in report['checks']:
            missing_ratio = report['checks']['missing_values']['missing_ratio']
            score -= missing_ratio * 1000  # 每1%缺失扣10分
        
        # 时间不连续扣分
        if 'time_continuity' in report['checks']:
            gaps = report['checks']['time_continuity']['gaps_found']
            score -= min(gaps * 2, 20)  # 每个间隔扣2分，最多扣20分
        
        # OHLC不一致扣分
        if 'ohlc_consistency' in report['checks']:
            inconsistent = report['checks']['ohlc_consistency']['inconsistent_count']
            total = report['total_records']
            if total > 0:
                score -= (inconsistent / total) * 100  # 按比例扣分
        
        # 价格异常值扣分
        if 'price_outliers' in report['checks']:
            outliers = report['checks']['price_outliers']['outlier_count']
            total = report['total_records']
            if total > 0:
                score -= (outliers / total) * 50  # 按比例扣分
        
        # 成交量异常值扣分
        if 'volume_outliers' in report['checks']:
            outliers = report['checks']['volume_outliers']['outlier_count']
            total = report['total_records']
            if total > 0:
                score -= (outliers / total) * 30  # 按比例扣分
        
        return max(0.0, min(100.0, score))


def main():
    """测试函数"""
    # 创建验证器
    validator = DataValidator()
    
    # 创建测试数据
    print("\n=== 创建测试数据 ===")
    np.random.seed(42)
    n = 100
    
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'open': 100 + np.random.randn(n) * 2,
        'high': 102 + np.random.randn(n) * 2,
        'low': 98 + np.random.randn(n) * 2,
        'close': 100 + np.random.randn(n) * 2,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # 添加一些问题数据
    test_data.loc[10, 'close'] = np.nan  # 缺失值
    test_data.loc[20, 'high'] = 90  # OHLC不一致
    test_data.loc[30, 'close'] = 150  # 价格异常
    test_data.loc[40, 'volume'] = 100000  # 成交量异常
    
    print(f"测试数据: {len(test_data)} 条记录")
    
    # 执行验证
    print("\n=== 执行验证 ===")
    validated_data, report = validator.validate(test_data, symbol="TEST", fix_issues=True)
    
    print(f"\n验证报告:")
    print(f"  总记录数: {report['total_records']}")
    print(f"  质量分数: {report['quality_score']:.2f}")
    print(f"  发现问题: {len(report['issues_found'])}")
    for issue in report['issues_found']:
        print(f"    - {issue}")
    print(f"  已修复: {len(report['issues_fixed'])}")
    for fix in report['issues_fixed']:
        print(f"    - {fix}")
    
    print(f"\n修复后数据: {len(validated_data)} 条记录")


if __name__ == "__main__":
    main()