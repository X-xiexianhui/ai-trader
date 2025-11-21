"""
特征归一化脚本

功能：
1. 加载带特征的数据
2. 划分训练集和测试集
3. 对27维手工特征进行归一化
4. 保存归一化后的数据和scaler

使用方法：
    python training/02_feature_normalization.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.data.storage import DataStorage
from src.features.feature_calculator import FeatureCalculator
from src.features.feature_scaler import FeatureScaler, StandardScaler, RobustScaler
from src.utils.logger import setup_logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    """
    按时间顺序划分训练集和测试集
    
    Args:
        df: 数据DataFrame
        test_ratio: 测试集比例
        
    Returns:
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="feature_normalization",
        log_file="02_feature_normalization.log",
        log_level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("开始特征归一化流程")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("\n步骤1: 加载配置文件")
    config = load_config()
    
    data_config = config['data']
    symbols = data_config['symbols']
    
    logger.info(f"品种: {symbols}")
    
    # 2. 加载带特征的数据
    logger.info("\n步骤2: 加载带特征的数据")
    processed_storage = DataStorage(base_path='data/processed')
    
    feature_data_dict = {}
    for symbol in symbols:
        logger.info(f"\n加载 {symbol}...")
        
        df = processed_storage.load_parquet(f"{symbol}_with_features")
        
        if df is not None and not df.empty:
            feature_data_dict[symbol] = df
            logger.info(f"✓ {symbol} 加载成功: {len(df)} 条记录, {len(df.columns)} 列")
        else:
            logger.warning(f"✗ {symbol} 加载失败")
    
    if not feature_data_dict:
        logger.error("没有成功加载任何数据，退出")
        return
    
    # 3. 获取特征名称和分组
    logger.info("\n步骤3: 识别特征列")
    feature_calculator = FeatureCalculator()
    
    # 手工计算一次以获取特征名称（不实际使用结果）
    sample_df = list(feature_data_dict.values())[0]
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    sample_ohlcv = sample_df[ohlcv_cols].head(100)
    feature_calculator.calculate_all_features(sample_ohlcv)
    
    feature_names = feature_calculator.get_feature_names()
    feature_groups = feature_calculator.get_feature_groups()
    
    logger.info(f"手工特征数量: {len(feature_names)}")
    logger.info("特征分组:")
    for group_name, features in feature_groups.items():
        logger.info(f"  - {group_name}: {features}")
    
    # 4. 划分训练集和测试集
    logger.info("\n步骤4: 划分训练集和测试集")
    test_ratio = 0.2
    
    train_data_dict = {}
    test_data_dict = {}
    
    for symbol, df in feature_data_dict.items():
        train_df, test_df = split_train_test(df, test_ratio=test_ratio)
        train_data_dict[symbol] = train_df
        test_data_dict[symbol] = test_df
        
        logger.info(f"{symbol}:")
        logger.info(f"  训练集: {len(train_df)} 条 ({train_df.index[0]} 到 {train_df.index[-1]})")
        logger.info(f"  测试集: {len(test_df)} 条 ({test_df.index[0]} 到 {test_df.index[-1]})")
    
    # 5. 特征归一化
    logger.info("\n步骤5: 特征归一化")
    
    normalized_train_dict = {}
    normalized_test_dict = {}
    scaler_dict = {}
    
    for symbol in feature_data_dict.keys():
        logger.info(f"\n归一化 {symbol}...")
        
        train_df = train_data_dict[symbol]
        test_df = test_data_dict[symbol]
        
        # 提取特征列
        X_train = train_df[feature_names].copy()
        X_test = test_df[feature_names].copy()
        
        # 创建FeatureScaler并拟合训练集
        feature_scaler = FeatureScaler()
        feature_scaler.fit(X_train, feature_groups)
        
        # 转换训练集和测试集
        X_train_scaled = feature_scaler.transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)
        
        # 创建归一化后的完整DataFrame
        train_normalized = train_df.copy()
        train_normalized[feature_names] = X_train_scaled
        
        test_normalized = test_df.copy()
        test_normalized[feature_names] = X_test_scaled
        
        normalized_train_dict[symbol] = train_normalized
        normalized_test_dict[symbol] = test_normalized
        scaler_dict[symbol] = feature_scaler
        
        logger.info(f"✓ {symbol} 归一化完成")
        logger.info(f"  训练集特征范围: [{X_train_scaled.min().min():.4f}, {X_train_scaled.max().max():.4f}]")
        logger.info(f"  测试集特征范围: [{X_test_scaled.min().min():.4f}, {X_test_scaled.max().max():.4f}]")
    
    # 6. 保存归一化后的数据
    logger.info("\n步骤6: 保存归一化后的数据")
    
    for symbol in feature_data_dict.keys():
        # 保存训练集
        success_train = processed_storage.save_parquet(
            data=normalized_train_dict[symbol],
            symbol=f"{symbol}_train_normalized",
            compression='snappy'
        )
        
        # 保存测试集
        success_test = processed_storage.save_parquet(
            data=normalized_test_dict[symbol],
            symbol=f"{symbol}_test_normalized",
            compression='snappy'
        )
        
        if success_train and success_test:
            logger.info(f"✓ {symbol} 归一化数据已保存")
        else:
            logger.warning(f"✗ {symbol} 数据保存失败")
    
    # 7. 保存scaler
    logger.info("\n步骤7: 保存scaler")
    scaler_dir = Path("models/scalers")
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol, scaler in scaler_dict.items():
        scaler_path = scaler_dir / symbol
        scaler.save(scaler_path)
        logger.info(f"✓ {symbol} scaler已保存到: {scaler_path}")
    
    # 8. 生成归一化报告
    logger.info("\n步骤8: 生成归一化报告")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("特征归一化报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("特征归一化策略:")
    report_lines.append("-" * 80)
    for group_name, features in feature_groups.items():
        if group_name == 'time':
            report_lines.append(f"{group_name}: 无需归一化（已在[-1,1]范围）")
        elif group_name in ['price_return', 'candlestick']:
            report_lines.append(f"{group_name}: StandardScaler (z-score标准化)")
        else:
            report_lines.append(f"{group_name}: RobustScaler (中位数和IQR)")
    
    report_lines.append("")
    
    for symbol in feature_data_dict.keys():
        report_lines.append(f"\n品种: {symbol}")
        report_lines.append("-" * 80)
        
        train_df = normalized_train_dict[symbol]
        test_df = normalized_test_dict[symbol]
        
        report_lines.append(f"训练集: {len(train_df)} 条记录")
        report_lines.append(f"  时间范围: {train_df.index[0]} 到 {train_df.index[-1]}")
        
        report_lines.append(f"测试集: {len(test_df)} 条记录")
        report_lines.append(f"  时间范围: {test_df.index[0]} 到 {test_df.index[-1]}")
        
        # 归一化后的统计信息
        X_train_scaled = train_df[feature_names]
        X_test_scaled = test_df[feature_names]
        
        report_lines.append(f"\n归一化后统计:")
        report_lines.append(f"  训练集:")
        report_lines.append(f"    均值范围: [{X_train_scaled.mean().min():.4f}, {X_train_scaled.mean().max():.4f}]")
        report_lines.append(f"    标准差范围: [{X_train_scaled.std().min():.4f}, {X_train_scaled.std().max():.4f}]")
        report_lines.append(f"    最小值: {X_train_scaled.min().min():.4f}")
        report_lines.append(f"    最大值: {X_train_scaled.max().max():.4f}")
        
        report_lines.append(f"  测试集:")
        report_lines.append(f"    均值范围: [{X_test_scaled.mean().min():.4f}, {X_test_scaled.mean().max():.4f}]")
        report_lines.append(f"    标准差范围: [{X_test_scaled.std().min():.4f}, {X_test_scaled.std().max():.4f}]")
        report_lines.append(f"    最小值: {X_test_scaled.min().min():.4f}")
        report_lines.append(f"    最大值: {X_test_scaled.max().max():.4f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"总计: 成功归一化 {len(feature_data_dict)} 个品种")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = "training/output/02_normalization_report.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n报告已保存: {report_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("特征归一化流程完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()