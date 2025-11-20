"""
数据处理流程示例
演示如何使用数据清洗、特征计算和归一化模块
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data.cleaning import DataCleaner
from src.data.features import FeatureCalculator
from src.data.normalization import FeatureScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_days=100):
    """生成示例OHLC数据"""
    logger.info(f"生成{n_days}天的示例数据...")
    
    # 生成时间索引（5分钟K线）
    start_date = datetime(2024, 1, 1)
    periods = n_days * 24 * 12  # 每天288根5分钟K线
    date_range = pd.date_range(start=start_date, periods=periods, freq='5T')
    
    # 生成价格数据（随机游走）
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, periods)
    close = 4000 * np.exp(np.cumsum(returns))
    
    # 生成OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.005, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, periods)))
    open_price = close * (1 + np.random.normal(0, 0.003, periods))
    
    # 生成成交量
    volume = np.random.lognormal(10, 1, periods)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=date_range)
    
    # 添加一些缺失值和异常值来测试清洗功能
    df.loc[df.index[100:105], 'Close'] = np.nan  # 连续缺失
    df.loc[df.index[200], 'Close'] *= 1.5  # 异常值
    
    logger.info(f"生成完成: {len(df)}行数据")
    return df


def main():
    """主函数 - 演示完整的数据处理流程"""
    
    # 1. 生成示例数据
    logger.info("=" * 60)
    logger.info("步骤1: 生成示例数据")
    logger.info("=" * 60)
    df_raw = generate_sample_data(n_days=30)
    logger.info(f"原始数据形状: {df_raw.shape}")
    logger.info(f"原始数据缺失值:\n{df_raw.isna().sum()}")
    
    # 2. 数据清洗
    logger.info("\n" + "=" * 60)
    logger.info("步骤2: 数据清洗")
    logger.info("=" * 60)
    
    cleaner = DataCleaner(max_consecutive_missing=5)
    df_clean, clean_report = cleaner.clean_pipeline(
        df_raw,
        target_timezone='UTC',
        trading_hours=None  # 24小时交易
    )
    
    logger.info(f"清洗后数据形状: {df_clean.shape}")
    logger.info(f"清洗后缺失值:\n{df_clean.isna().sum()}")
    logger.info(f"数据质量验证: {'通过' if clean_report['final_validation'] else '失败'}")
    
    # 3. 特征计算
    logger.info("\n" + "=" * 60)
    logger.info("步骤3: 特征计算")
    logger.info("=" * 60)
    
    feature_calc = FeatureCalculator()
    df_features = feature_calc.calculate_all_features(df_clean)
    
    logger.info(f"特征计算后数据形状: {df_features.shape}")
    logger.info(f"计算的特征数量: {len(feature_calc.get_feature_names())}")
    logger.info(f"特征名称: {feature_calc.get_feature_names()}")
    
    # 显示特征分组
    feature_groups = feature_calc.get_feature_groups()
    logger.info("\n特征分组:")
    for group_name, features in feature_groups.items():
        logger.info(f"  {group_name}: {len(features)}个特征")
    
    # 4. 特征归一化
    logger.info("\n" + "=" * 60)
    logger.info("步骤4: 特征归一化")
    logger.info("=" * 60)
    
    # 划分训练集和测试集
    train_size = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:train_size]
    df_test = df_features.iloc[train_size:]
    
    logger.info(f"训练集大小: {len(df_train)}")
    logger.info(f"测试集大小: {len(df_test)}")
    
    # 提取特征列
    feature_cols = feature_calc.get_feature_names()
    X_train = df_train[feature_cols]
    X_test = df_test[feature_cols]
    
    # 创建并拟合FeatureScaler
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train, feature_groups)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"归一化后训练集形状: {X_train_scaled.shape}")
    logger.info(f"归一化后测试集形状: {X_test_scaled.shape}")
    
    # 显示归一化前后的统计信息
    logger.info("\n归一化前后对比（训练集）:")
    logger.info(f"原始数据均值范围: [{X_train.mean().min():.4f}, {X_train.mean().max():.4f}]")
    logger.info(f"原始数据标准差范围: [{X_train.std().min():.4f}, {X_train.std().max():.4f}]")
    logger.info(f"归一化后均值范围: [{X_train_scaled.mean().min():.4f}, {X_train_scaled.mean().max():.4f}]")
    logger.info(f"归一化后标准差范围: [{X_train_scaled.std().min():.4f}, {X_train_scaled.std().max():.4f}]")
    
    # 5. 保存和加载scaler
    logger.info("\n" + "=" * 60)
    logger.info("步骤5: 保存和加载Scaler")
    logger.info("=" * 60)
    
    scaler_path = "../models/scalers"
    scaler.save(scaler_path)
    logger.info(f"Scaler已保存到: {scaler_path}")
    
    # 加载scaler
    scaler_loaded = FeatureScaler.load(scaler_path)
    logger.info(f"Scaler已加载")
    
    # 验证加载的scaler
    X_test_scaled_loaded = scaler_loaded.transform(X_test)
    is_same = np.allclose(X_test_scaled.values, X_test_scaled_loaded.values)
    logger.info(f"加载的scaler验证: {'通过' if is_same else '失败'}")
    
    # 6. 总结
    logger.info("\n" + "=" * 60)
    logger.info("数据处理流程完成!")
    logger.info("=" * 60)
    logger.info(f"最终数据形状: {X_train_scaled.shape}")
    logger.info(f"特征数量: {len(feature_cols)}")
    logger.info(f"训练样本数: {len(X_train_scaled)}")
    logger.info(f"测试样本数: {len(X_test_scaled)}")
    
    # 显示部分数据
    logger.info("\n归一化后的特征样本（前5行，前5列）:")
    logger.info(X_train_scaled.iloc[:5, :5])
    
    return df_features, X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    df_features, X_train, X_test, scaler = main()
    print("\n✅ 数据处理流程演示完成！")