"""
完整数据管道示例
演示从原始数据到特征验证的完整流程

流程:
1. 数据清洗 (DataCleaner)
2. 特征计算 (FeatureCalculator)
3. 特征归一化 (FeatureScaler)
4. 特征验证 (FeatureValidator)
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

from src.data.cleaning import DataCleaner
from src.data.features import FeatureCalculator
from src.data.normalization import FeatureScaler
from src.data.validation import FeatureValidator


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    生成模拟的OHLCV数据
    
    Args:
        n_samples: 样本数量
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    print(f"\n{'='*60}")
    print("步骤1: 生成模拟数据")
    print(f"{'='*60}")
    
    # 生成时间序列
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # 生成价格数据(随机游走)
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.02
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLC数据
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_samples) * 0.01))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_samples) * 0.01))
    open_prices = close_prices * (1 + np.random.randn(n_samples) * 0.005)
    
    # 生成成交量数据
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


def step1_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤1: 数据清洗
    
    Args:
        df: 原始数据
        
    Returns:
        清洗后的数据
    """
    print(f"\n{'='*60}")
    print("步骤2: 数据清洗")
    print(f"{'='*60}")
    
    cleaner = DataCleaner()
    
    # 执行清洗
    cleaned_df, _ = cleaner.clean_pipeline(df)
    
    print(f"✓ 数据清洗完成")
    print(f"  原始数据: {len(df)} 条")
    print(f"  清洗后: {len(cleaned_df)} 条")
    print(f"  移除: {len(df) - len(cleaned_df)} 条")
    
    return cleaned_df


def step2_feature_calculation(df: pd.DataFrame) -> pd.DataFrame:
    """
    步骤2: 特征计算
    
    Args:
        df: 清洗后的数据
        
    Returns:
        包含特征的数据
    """
    print(f"\n{'='*60}")
    print("步骤3: 特征计算")
    print(f"{'='*60}")
    
    calculator = FeatureCalculator()
    
    # 计算所有特征
    features_df = calculator.calculate_all_features(df)
    
    print(f"✓ 特征计算完成")
    print(f"  总特征数: {len(features_df.columns)} 维")
    print(f"  特征列表:")
    for i, col in enumerate(features_df.columns, 1):
        print(f"    {i:2d}. {col}")
    
    # 检查缺失值
    missing = features_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  缺失值统计:")
        for col, count in missing[missing > 0].items():
            print(f"    {col}: {count} ({count/len(features_df)*100:.1f}%)")
    
    return features_df


def step3_feature_normalization(
    features_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> tuple:
    """
    步骤3: 特征归一化
    
    Args:
        features_df: 特征数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (归一化后的训练集, 验证集, 测试集, scaler)
    """
    print(f"\n{'='*60}")
    print("步骤4: 特征归一化")
    print(f"{'='*60}")
    
    # 移除缺失值
    features_df = features_df.dropna()
    
    # 划分数据集
    n = len(features_df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_df = features_df.iloc[:train_size]
    val_df = features_df.iloc[train_size:train_size+val_size]
    test_df = features_df.iloc[train_size+val_size:]
    
    print(f"✓ 数据集划分:")
    print(f"  训练集: {len(train_df)} 条 ({len(train_df)/n*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 条 ({len(val_df)/n*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 条 ({len(test_df)/n*100:.1f}%)")
    
    # 定义特征组
    feature_groups = {
        'price_return': ['ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20'],
        'volatility': ['ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol'],
        'technical': ['EMA20', 'stoch', 'MACD', 'VWAP'],
        'volume': ['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20'],
        'pattern': ['pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm', 
                   'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG'],
        'time': ['sin_tod', 'cos_tod']
    }
    
    # 创建归一化器
    scaler = FeatureScaler()
    
    # 拟合并转换
    train_normalized = scaler.fit_transform(train_df, feature_groups)
    val_normalized = scaler.transform(val_df)
    test_normalized = scaler.transform(test_df)
    
    print(f"\n✓ 特征归一化完成")
    print(f"  使用的归一化器:")
    for group, scaler_obj in scaler.scalers.items():
        scaler_type = type(scaler_obj).__name__
        print(f"    {group}: {scaler_type}")
    
    # 显示归一化前后的统计信息
    print(f"\n  归一化前后对比 (训练集):")
    print(f"  {'特征':<20} {'原始均值':>12} {'原始标准差':>12} {'归一化均值':>12} {'归一化标准差':>12}")
    print(f"  {'-'*68}")
    
    sample_features = ['ret_1', 'ATR14_norm', 'volume', 'pos_in_range_20']
    for feat in sample_features:
        if feat in train_df.columns:
            orig_mean = train_df[feat].mean()
            orig_std = train_df[feat].std()
            norm_mean = train_normalized[feat].mean()
            norm_std = train_normalized[feat].std()
            print(f"  {feat:<20} {orig_mean:>12.4f} {orig_std:>12.4f} {norm_mean:>12.4f} {norm_std:>12.4f}")
    
    return train_normalized, val_normalized, test_normalized, scaler


def step4_feature_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> dict:
    """
    步骤4: 特征验证
    
    Args:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        
    Returns:
        验证报告
    """
    print(f"\n{'='*60}")
    print("步骤5: 特征验证")
    print(f"{'='*60}")
    
    # 创建目标变量(未来1期收益率)
    def create_target(df):
        target = df['ret_1'].shift(-1)
        return df[:-1], target[:-1]
    
    X_train, y_train = create_target(train_df)
    X_val, y_val = create_target(val_df)
    X_test, y_test = create_target(test_df)
    
    print(f"✓ 创建目标变量 (未来1期收益率)")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  验证集: {len(X_val)} 条")
    print(f"  测试集: {len(X_test)} 条")
    
    # 训练简单模型用于验证
    print(f"\n✓ 训练RandomForest模型用于特征验证...")
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    
    print(f"  训练集 R²: {train_score:.4f}")
    print(f"  验证集 R²: {val_score:.4f}")
    print(f"  测试集 R²: {test_score:.4f}")
    
    # 创建验证器
    validator = FeatureValidator()
    
    # 1. 单特征信息测试
    print(f"\n{'='*60}")
    print("5.1 单特征信息测试")
    print(f"{'='*60}")
    
    info_results = validator.test_single_feature_information(X_train, y_train)
    
    print(f"\n  Top 10 特征 (按R²排序):")
    print(f"  {'排名':<6} {'特征':<20} {'R²':>10} {'互信息':>10}")
    print(f"  {'-'*46}")
    
    sorted_features = sorted(info_results.items(), key=lambda x: x[1]['r2'], reverse=True)
    for i, (feat, metrics) in enumerate(sorted_features[:10], 1):
        print(f"  {i:<6} {feat:<20} {metrics['r2']:>10.4f} {metrics['mi']:>10.4f}")
    
    # 2. 排列重要性测试
    print(f"\n{'='*60}")
    print("5.2 排列重要性测试")
    print(f"{'='*60}")
    
    perm_results = validator.test_permutation_importance(
        model, X_val, y_val, n_repeats=5
    )
    
    print(f"\n  Top 10 特征 (按排列重要性排序):")
    print(f"  {'排名':<6} {'特征':<20} {'重要性':>12} {'标准差':>12}")
    print(f"  {'-'*50}")
    
    sorted_perm = sorted(perm_results.items(), key=lambda x: x[1]['importance'], reverse=True)
    for i, (feat, metrics) in enumerate(sorted_perm[:10], 1):
        print(f"  {i:<6} {feat:<20} {metrics['importance']:>12.6f} {metrics['std']:>12.6f}")
    
    # 3. 特征相关性测试
    print(f"\n{'='*60}")
    print("5.3 特征相关性测试")
    print(f"{'='*60}")
    
    corr_results = validator.test_feature_correlation(X_train, threshold=0.8)
    
    if corr_results:
        print(f"\n  发现 {len(corr_results)} 对高相关特征 (|r| > 0.8):")
        print(f"  {'特征1':<20} {'特征2':<20} {'相关系数':>12}")
        print(f"  {'-'*52}")
        
        for pair in corr_results[:10]:  # 只显示前10对
            print(f"  {pair['feature1']:<20} {pair['feature2']:<20} {pair['correlation']:>12.4f}")
    else:
        print(f"\n  ✓ 未发现高相关特征对")
    
    # 4. VIF多重共线性测试
    print(f"\n{'='*60}")
    print("5.4 VIF多重共线性测试")
    print(f"{'='*60}")
    
    vif_results = validator.test_vif_multicollinearity(X_train, threshold=10.0)
    
    high_vif = {k: v for k, v in vif_results.items() if v > 10.0}
    
    if high_vif:
        print(f"\n  发现 {len(high_vif)} 个高VIF特征 (VIF > 10):")
        print(f"  {'特征':<20} {'VIF':>12}")
        print(f"  {'-'*32}")
        
        sorted_vif = sorted(high_vif.items(), key=lambda x: x[1], reverse=True)
        for feat, vif in sorted_vif[:10]:  # 只显示前10个
            print(f"  {feat:<20} {vif:>12.2f}")
    else:
        print(f"\n  ✓ 所有特征VIF < 10")
    
    # 5. 生成完整验证报告
    print(f"\n{'='*60}")
    print("5.5 生成完整验证报告")
    print(f"{'='*60}")
    
    report = validator.generate_validation_report(
        X_train, y_train, model, X_val, y_val
    )
    
    print(f"\n✓ 验证报告生成完成")
    print(f"  包含 {len(report)} 个部分:")
    for key in report.keys():
        print(f"    - {key}")
    
    # 6. 特征移除建议
    print(f"\n{'='*60}")
    print("5.6 特征移除建议")
    print(f"{'='*60}")
    
    suggestions = validator.suggest_feature_removal(report)
    
    if suggestions:
        print(f"\n  建议移除 {len(suggestions)} 个特征:")
        print(f"  {'特征':<20} {'原因':<40}")
        print(f"  {'-'*60}")
        
        for feat, reason in suggestions.items():
            print(f"  {feat:<20} {reason:<40}")
    else:
        print(f"\n  ✓ 所有特征质量良好,无需移除")
    
    return report


def save_scaler_demo(scaler: FeatureScaler, save_dir: str = "models/scalers"):
    """
    演示保存和加载归一化器
    
    Args:
        scaler: 归一化器
        save_dir: 保存目录
    """
    print(f"\n{'='*60}")
    print("步骤6: 保存和加载归一化器")
    print(f"{'='*60}")
    
    # 保存
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    scaler.save(str(save_path / "feature_scaler.pkl"))
    print(f"✓ 归一化器已保存到: {save_path / 'feature_scaler.pkl'}")
    
    # 加载
    loaded_scaler = FeatureScaler.load(str(save_path / "feature_scaler.pkl"))
    print(f"✓ 归一化器已加载")
    
    # 验证
    print(f"\n  验证加载的归一化器:")
    print(f"    特征组数量: {len(loaded_scaler.feature_groups)}")
    print(f"    归一化器类型: {loaded_scaler.scaler_types}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("完整数据管道演示")
    print("="*60)
    print("\n本示例演示从原始数据到特征验证的完整流程:")
    print("  1. 生成模拟数据")
    print("  2. 数据清洗")
    print("  3. 特征计算")
    print("  4. 特征归一化")
    print("  5. 特征验证")
    print("  6. 保存归一化器")
    
    # 步骤1: 生成数据
    raw_data = generate_sample_data(n_samples=1000)
    
    # 步骤2: 数据清洗
    cleaned_data = step1_data_cleaning(raw_data)
    
    # 步骤3: 特征计算
    features_df = step2_feature_calculation(cleaned_data)
    
    # 步骤4: 特征归一化
    train_df, val_df, test_df, scaler = step3_feature_normalization(features_df)
    
    # 步骤5: 特征验证
    report = step4_feature_validation(train_df, val_df, test_df)
    
    # 步骤6: 保存归一化器
    save_scaler_demo(scaler)
    
    # 总结
    print(f"\n{'='*60}")
    print("完整数据管道演示完成")
    print(f"{'='*60}")
    print("\n✓ 所有步骤执行成功!")
    print("\n数据管道总结:")
    print(f"  原始数据: {len(raw_data)} 条")
    print(f"  清洗后: {len(cleaned_data)} 条")
    print(f"  特征维度: {len(features_df.columns)} 维")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(val_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    print("\n下一步:")
    print("  - 使用真实市场数据替换模拟数据")
    print("  - 根据验证报告优化特征工程")
    print("  - 将数据输入TS2Vec模型进行训练")


if __name__ == "__main__":
    main()