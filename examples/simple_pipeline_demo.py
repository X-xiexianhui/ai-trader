"""
简化的数据管道示例
演示数据清洗、特征计算和归一化的基本流程
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features import DataCleaner, FeatureCalculator, FeatureScaler


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """生成模拟的OHLCV数据"""
    print("\n" + "="*60)
    print("步骤1: 生成模拟数据")
    print("="*60)
    
    # 生成时间序列
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # 生成价格数据
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.02
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLC数据
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_prices = close_prices * (1 + np.random.randn(n_samples) * 0.005)
    volumes = np.random.lognormal(10, 1, n_samples)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volumes
    }, index=pd.DatetimeIndex(timestamps))
    
    print(f"✓ 生成了 {len(df)} 条数据")
    print(f"  时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"  价格范围: {df['Close'].min():.2f} 到 {df['Close'].max():.2f}")
    
    return df


def main():
    """主函数"""
    print("\n" + "="*60)
    print("简化数据管道演示")
    print("="*60)
    print("\n本示例演示:")
    print("  1. 数据清洗")
    print("  2. 特征计算")
    print("  3. 特征归一化")
    
    # 步骤1: 生成数据
    raw_data = generate_sample_data(n_samples=1000)
    
    # 步骤2: 数据清洗
    print("\n" + "="*60)
    print("步骤2: 数据清洗")
    print("="*60)
    
    cleaner = DataCleaner()
    cleaned_data, _ = cleaner.clean_pipeline(raw_data)
    
    print(f"✓ 数据清洗完成")
    print(f"  原始数据: {len(raw_data)} 条")
    print(f"  清洗后: {len(cleaned_data)} 条")
    
    # 步骤3: 特征计算
    print("\n" + "="*60)
    print("步骤3: 特征计算")
    print("="*60)
    
    calculator = FeatureCalculator()
    features_df = calculator.calculate_all_features(cleaned_data)
    
    print(f"✓ 特征计算完成")
    print(f"  总特征数: {len(features_df.columns)} 维")
    print(f"  数据行数: {len(features_df)} 条")
    
    # 显示前几个特征
    feature_names = calculator.get_feature_names()
    print(f"\n  计算的特征 (共{len(feature_names)}个):")
    for i, name in enumerate(feature_names[:10], 1):
        print(f"    {i:2d}. {name}")
    if len(feature_names) > 10:
        print(f"    ... 还有 {len(feature_names)-10} 个特征")
    
    # 步骤4: 特征归一化
    print("\n" + "="*60)
    print("步骤4: 特征归一化")
    print("="*60)
    
    # 划分数据集
    n = len(features_df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)
    
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size:train_size+val_size]
    test_df = features_df.iloc[train_size+val_size:]
    
    print(f"✓ 数据集划分:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/n*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
    
    # 定义特征组
    feature_groups = calculator.get_feature_groups()
    
    # 创建并拟合归一化器
    scaler = FeatureScaler()
    train_normalized = scaler.fit_transform(train_df, feature_groups)
    val_normalized = scaler.transform(val_df)
    test_normalized = scaler.transform(test_df)
    
    print(f"\n✓ 特征归一化完成")
    print(f"  使用的归一化器:")
    for group, scaler_obj in scaler.scalers.items():
        scaler_type = type(scaler_obj).__name__
        print(f"    {group}: {scaler_type}")
    
    # 显示归一化前后的统计信息
    print(f"\n  归一化前后对比 (训练集, 前5个特征):")
    print(f"  {'特征':<20} {'原始均值':>12} {'原始标准差':>12} {'归一化均值':>12} {'归一化标准差':>12}")
    print(f"  {'-'*68}")
    
    for feat in feature_names[:5]:
        if feat in train_df.columns:
            orig_mean = train_df[feat].mean()
            orig_std = train_df[feat].std()
            norm_mean = train_normalized[feat].mean()
            norm_std = train_normalized[feat].std()
            print(f"  {feat:<20} {orig_mean:>12.4f} {orig_std:>12.4f} {norm_mean:>12.4f} {norm_std:>12.4f}")
    
    # 步骤5: 保存归一化器
    print("\n" + "="*60)
    print("步骤5: 保存归一化器")
    print("="*60)
    
    save_dir = Path("models/scalers")
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(str(save_dir))
    
    print(f"✓ 归一化器已保存到: {save_dir}")
    
    # 总结
    print("\n" + "="*60)
    print("数据管道演示完成")
    print("="*60)
    print("\n✓ 所有步骤执行成功!")
    print("\n数据管道总结:")
    print(f"  原始数据: {len(raw_data)} 条")
    print(f"  清洗后: {len(cleaned_data)} 条")
    print(f"  特征维度: {len(feature_names)} 维")
    print(f"  训练集: {len(train_normalized)} 条")
    print(f"  验证集: {len(val_normalized)} 条")
    print(f"  测试集: {len(test_normalized)} 条")
    
    print("\n下一步:")
    print("  - 使用真实市场数据替换模拟数据")
    print("  - 将归一化后的数据输入TS2Vec模型进行训练")
    print("  - 进行特征验证和重要性分析")


if __name__ == "__main__":
    main()