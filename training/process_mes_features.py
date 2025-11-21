"""
MES数据特征处理完整流程

功能：
1. 读取本地mes_5m_data.csv文件
2. 数据清洗（缺失值、异常值、时间对齐）
3. 计算27个手工特征
4. 特征归一化
5. 特征验证并生成详细报告

使用方法:
    python training/process_mes_features.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.features.data_cleaner import DataCleaner
from src.features.feature_calculator import FeatureCalculator
from src.features.feature_scaler import FeatureScaler
from src.features.feature_validator import FeatureValidator
from src.utils.logger import setup_logger
import logging

# 设置日志
logger = setup_logger('process_features', log_dir='training/output', log_file='process_features.log')


def load_mes_data(filepath: str) -> pd.DataFrame:
    """
    加载MES数据
    
    Args:
        filepath: CSV文件路径
        
    Returns:
        DataFrame with datetime index
    """
    logger.info(f"加载数据: {filepath}")
    
    # 读取CSV
    df = pd.read_csv(filepath)
    
    # 转换datetime列为索引，并统一转换为UTC时区
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.set_index('datetime')
    
    # 标准化列名（转换为首字母大写）
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_mapping)
    
    # 只保留OHLCV列
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    
    logger.info(f"数据加载完成: {len(df)}行 × {len(df.columns)}列")
    logger.info(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    logger.info(f"时区: {df.index.tz}")
    
    return df


def clean_data(df: pd.DataFrame) -> tuple:
    """
    数据清洗
    
    Args:
        df: 原始DataFrame
        
    Returns:
        (cleaned_df, cleaning_report)
    """
    logger.info("=" * 80)
    logger.info("开始数据清洗...")
    logger.info("=" * 80)
    
    # 创建数据清洗器
    cleaner = DataCleaner(
        max_consecutive_missing=5,
        interpolation_limit=3,
        sigma_threshold=3.0
    )
    
    # 美国股市正常交易时间（美东时间 ET）：
    # 周一至周五 9:30 AM - 4:00 PM
    # 转换为UTC时间需要考虑夏令时：
    # - 夏令时（EDT，UTC-4）: 9:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC
    # - 标准时（EST，UTC-5）: 9:30 ET = 14:30 UTC, 16:00 ET = 21:00 UTC
    #
    # 为了简化，我们使用UTC时间 13:30-21:00（覆盖两种情况）
    
    # 执行完整清洗流程
    cleaned_df, report = cleaner.clean_pipeline(
        df,
        target_timezone='UTC',
        trading_hours=(13, 21),  # UTC 13:00-21:00，对应美东约9:00-17:00
        sigma_threshold=3.0
    )
    
    # 额外过滤：只保留工作日（周一至周五）
    logger.info("过滤周末数据...")
    original_len = len(cleaned_df)
    
    # 0=周一, 6=周日
    weekday = cleaned_df.index.dayofweek
    weekday_mask = weekday < 5  # 周一到周五
    cleaned_df = cleaned_df[weekday_mask]
    
    filtered_count = original_len - len(cleaned_df)
    logger.info(f"过滤了{filtered_count}行周末数据")
    logger.info(f"最终数据: {len(cleaned_df)}行")
    
    # 打印清洗报告摘要
    cleaner.print_report_summary()
    
    return cleaned_df, report


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算27个手工特征
    
    Args:
        df: 清洗后的DataFrame
        
    Returns:
        带特征的DataFrame
    """
    logger.info("=" * 80)
    logger.info("开始计算特征...")
    logger.info("=" * 80)
    
    # 创建特征计算器
    calculator = FeatureCalculator()
    
    # 计算所有特征
    df_with_features = calculator.calculate_all_features(df)
    
    # 获取特征分组
    feature_groups = calculator.get_feature_groups()
    
    logger.info(f"特征计算完成: {len(calculator.get_feature_names())}个特征")
    logger.info("\n特征分组:")
    for group_name, features in feature_groups.items():
        logger.info(f"  {group_name}: {len(features)}个特征")
        logger.info(f"    {', '.join(features)}")
    
    return df_with_features, feature_groups


def normalize_features(df: pd.DataFrame, feature_groups: dict) -> tuple:
    """
    特征归一化
    
    Args:
        df: 带特征的DataFrame
        feature_groups: 特征分组字典
        
    Returns:
        (normalized_df, scaler)
    """
    logger.info("=" * 80)
    logger.info("开始特征归一化...")
    logger.info("=" * 80)
    
    # 提取特征列
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    X = df[all_features].copy()
    
    # 创建特征归一化器
    scaler = FeatureScaler()
    
    # 拟合并转换
    X_normalized = scaler.fit_transform(X, feature_groups)
    
    # 保存scaler
    scaler_path = Path('training/output/scalers')
    scaler.save(scaler_path)
    logger.info(f"Scaler已保存到: {scaler_path}")
    
    # 创建归一化后的完整DataFrame
    df_normalized = df.copy()
    df_normalized[all_features] = X_normalized
    
    logger.info(f"特征归一化完成: {len(all_features)}个特征")
    
    # 显示归一化前后的统计信息
    logger.info("\n归一化前后对比（前5个特征）:")
    for feat in all_features[:5]:
        logger.info(f"  {feat}:")
        logger.info(f"    原始: mean={X[feat].mean():.4f}, std={X[feat].std():.4f}")
        logger.info(f"    归一化: mean={X_normalized[feat].mean():.4f}, std={X_normalized[feat].std():.4f}")
    
    return df_normalized, scaler


def validate_features(df: pd.DataFrame, feature_groups: dict) -> dict:
    """
    特征验证
    
    Args:
        df: 归一化后的DataFrame
        feature_groups: 特征分组字典
        
    Returns:
        验证结果字典
    """
    logger.info("=" * 80)
    logger.info("开始特征验证...")
    logger.info("=" * 80)
    
    # 提取特征
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    X = df[all_features].copy()
    
    # 创建目标变量（未来1周期收益率）
    y = df['Close'].pct_change(1).shift(-1)
    
    # 删除NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    logger.info(f"有效样本数: {len(X_valid)}")
    
    # 创建特征验证器
    validator = FeatureValidator()
    
    # 1. 单特征信息量测试
    logger.info("\n1. 单特征信息量测试...")
    info_results = validator.test_single_feature_information(X_valid, y_valid, top_n=10)
    
    # 2. 特征相关性检测
    logger.info("\n2. 特征相关性检测...")
    corr_matrix, high_corr_pairs = validator.test_feature_correlation(
        X_valid, 
        threshold=0.85,
        plot=True
    )
    
    # 3. VIF多重共线性检测
    logger.info("\n3. VIF多重共线性检测...")
    vif_results = validator.test_vif_multicollinearity(X_valid, threshold=10.0)
    
    # 生成完整验证报告
    logger.info("\n生成特征验证报告...")
    report_path = 'training/output/feature_validation_report.txt'
    validation_results = validator.generate_validation_report(report_path)
    
    logger.info(f"特征验证报告已保存: {report_path}")
    
    return validation_results


def generate_summary_report(
    df_original: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    df_features: pd.DataFrame,
    df_normalized: pd.DataFrame,
    cleaning_report: dict,
    feature_groups: dict,
    validation_results: dict
) -> None:
    """
    生成完整的处理摘要报告
    """
    logger.info("=" * 80)
    logger.info("生成处理摘要报告...")
    logger.info("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MES数据特征处理完整报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. 数据概览
    report_lines.append("1. 数据概览")
    report_lines.append("-" * 80)
    report_lines.append(f"原始数据: {len(df_original)}行")
    report_lines.append(f"清洗后数据: {len(df_cleaned)}行")
    report_lines.append(f"特征计算后: {len(df_features)}行")
    report_lines.append(f"最终数据: {len(df_normalized)}行")
    report_lines.append(f"时间范围: {df_normalized.index.min()} 至 {df_normalized.index.max()}")
    report_lines.append("")
    
    # 2. 数据清洗摘要
    report_lines.append("2. 数据清洗摘要")
    report_lines.append("-" * 80)
    if 'missing_values' in cleaning_report:
        mv = cleaning_report['missing_values']
        report_lines.append(f"缺失值处理:")
        report_lines.append(f"  - 删除数据段: {len(mv['removed_segments'])}个 ({mv['removed_rows']}行)")
        report_lines.append(f"  - 插值点数: {sum(mv['interpolated_points'].values())}个")
    
    if 'outliers' in cleaning_report:
        ol = cleaning_report['outliers']
        report_lines.append(f"异常值处理:")
        report_lines.append(f"  - 检测异常: {ol['total_outliers']}个")
        report_lines.append(f"  - 修正尖峰: {ol['spike_outliers']}个")
        report_lines.append(f"  - 保留跳空: {ol['gap_outliers']}个")
    
    if 'validation' in cleaning_report:
        vd = cleaning_report['validation']
        report_lines.append(f"质量验证: {'通过✓' if vd['is_valid'] else '失败✗'}")
    report_lines.append("")
    
    # 3. 特征计算摘要
    report_lines.append("3. 特征计算摘要")
    report_lines.append("-" * 80)
    total_features = sum(len(features) for features in feature_groups.values())
    report_lines.append(f"总特征数: {total_features}个")
    report_lines.append("")
    for group_name, features in feature_groups.items():
        report_lines.append(f"{group_name} ({len(features)}个):")
        for feat in features:
            report_lines.append(f"  - {feat}")
    report_lines.append("")
    
    # 4. 特征归一化摘要
    report_lines.append("4. 特征归一化摘要")
    report_lines.append("-" * 80)
    report_lines.append("归一化方法:")
    report_lines.append("  - price_return: StandardScaler (z-score)")
    report_lines.append("  - volatility: RobustScaler (median/IQR)")
    report_lines.append("  - technical: RobustScaler (median/IQR)")
    report_lines.append("  - volume: RobustScaler (median/IQR)")
    report_lines.append("  - candlestick: StandardScaler (z-score)")
    report_lines.append("  - time: 无需归一化（已在[-1,1]范围）")
    report_lines.append("")
    
    # 5. 特征验证摘要
    report_lines.append("5. 特征验证摘要")
    report_lines.append("-" * 80)
    
    if 'single_feature_info' in validation_results:
        df = validation_results['single_feature_info'].head(5)
        report_lines.append("Top 5 信息量最高的特征:")
        for idx, row in df.iterrows():
            report_lines.append(f"  {row['feature']}: R²={row['r2_score']:.4f}, MI={row['mutual_info']:.4f}")
    
    if 'high_corr_pairs' in validation_results:
        pairs = validation_results['high_corr_pairs'][:5]
        report_lines.append(f"\nTop 5 高度相关特征对:")
        for feat1, feat2, corr in pairs:
            report_lines.append(f"  {feat1} <-> {feat2}: {corr:.4f}")
    
    if 'vif' in validation_results:
        df = validation_results['vif']
        high_vif = df[df['has_multicollinearity']]
        report_lines.append(f"\n多重共线性特征: {len(high_vif)}个")
        for idx, row in high_vif.head(5).iterrows():
            report_lines.append(f"  {row['feature']}: VIF={row['VIF']:.2f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("处理完成！")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = 'training/output/processing_summary_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"\n处理摘要报告已保存: {report_path}")
    
    # 同时打印到控制台
    print("\n" + report_text)


def main():
    """主函数"""
    try:
        # 1. 加载数据
        data_path = 'training/output/mes_5m_data.csv'
        df_original = load_mes_data(data_path)
        
        # 2. 数据清洗
        df_cleaned, cleaning_report = clean_data(df_original)
        
        # 3. 计算特征
        df_features, feature_groups = calculate_features(df_cleaned)
        
        # 4. 特征归一化
        df_normalized, scaler = normalize_features(df_features, feature_groups)
        
        # 5. 特征验证
        validation_results = validate_features(df_normalized, feature_groups)
        
        # 6. 保存处理后的数据
        output_path = 'training/output/mes_features_normalized.csv'
        df_normalized.to_csv(output_path)
        logger.info(f"\n归一化后的数据已保存: {output_path}")
        
        # 同时保存为Parquet格式
        parquet_path = 'training/output/mes_features_normalized.parquet'
        df_normalized.to_parquet(parquet_path)
        logger.info(f"归一化后的数据已保存: {parquet_path}")
        
        # 7. 生成摘要报告
        generate_summary_report(
            df_original,
            df_cleaned,
            df_features,
            df_normalized,
            cleaning_report,
            feature_groups,
            validation_results
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ 所有处理步骤完成！")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ 处理失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()