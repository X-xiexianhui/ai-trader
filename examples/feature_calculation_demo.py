"""
特征计算演示脚本

展示如何使用FeatureCalculator计算27维手工特征
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.features import FeatureCalculator
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data(n=1000):
    """创建示例OHLC数据"""
    logger.info(f"创建{n}条示例数据...")
    
    np.random.seed(42)
    
    # 创建时间索引（5分钟K线）
    start_date = datetime(2023, 1, 1, 9, 30)
    dates = [start_date + timedelta(minutes=5*i) for i in range(n)]
    
    # 生成模拟价格数据（随机游走）
    returns = np.random.randn(n) * 0.01  # 1%标准差
    close = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLC
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_price = close + np.random.randn(n) * 0.3
    
    # 生成成交量
    volume = np.random.lognormal(10, 0.5, n).astype(int)
    
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=pd.DatetimeIndex(dates))
    
    logger.info(f"数据创建完成: {len(df)}行")
    return df


def demonstrate_feature_calculation():
    """演示特征计算流程"""
    print("=" * 80)
    print("特征计算演示")
    print("=" * 80)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    df = create_sample_data(n=1000)
    print(f"   原始数据形状: {df.shape}")
    print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"\n   前5行数据:")
    print(df.head())
    
    # 2. 初始化特征计算器
    print("\n2. 初始化特征计算器...")
    calculator = FeatureCalculator()
    
    # 3. 计算各组特征
    print("\n3. 计算各组特征...")
    
    # 3.1 价格与收益特征
    print("\n   3.1 计算价格与收益特征（5维）...")
    df_with_price = calculator.calculate_price_return_features(df.copy())
    price_features = ['ret_1', 'ret_5', 'ret_20', 'price_slope_20', 'C_div_MA20']
    print(f"       特征: {price_features}")
    print(f"       示例值:")
    print(df_with_price[price_features].iloc[25:30])
    
    # 3.2 波动率特征
    print("\n   3.2 计算波动率特征（5维）...")
    df_with_vol = calculator.calculate_volatility_features(df.copy())
    vol_features = ['ATR14_norm', 'vol_20', 'range_20_norm', 'BB_width_norm', 'parkinson_vol']
    print(f"       特征: {vol_features}")
    print(f"       示例值:")
    print(df_with_vol[vol_features].iloc[25:30])
    
    # 3.3 技术指标特征
    print("\n   3.3 计算技术指标特征（4维）...")
    df_with_tech = calculator.calculate_technical_indicators(df.copy())
    tech_features = ['EMA20', 'stoch', 'MACD', 'VWAP']
    print(f"       特征: {tech_features}")
    print(f"       示例值:")
    print(df_with_tech[tech_features].iloc[25:30])
    
    # 3.4 成交量特征
    print("\n   3.4 计算成交量特征（4维）...")
    df_with_volume = calculator.calculate_volume_features(df.copy())
    volume_features = ['volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20']
    print(f"       特征: {volume_features}")
    print(f"       示例值:")
    print(df_with_volume[volume_features].iloc[25:30])
    
    # 3.5 K线形态特征
    print("\n   3.5 计算K线形态特征（7维）...")
    df_with_candle = calculator.calculate_candlestick_features(df.copy())
    candle_features = ['pos_in_range_20', 'dist_to_HH20_norm', 'dist_to_LL20_norm',
                       'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG']
    print(f"       特征: {candle_features}")
    print(f"       示例值:")
    print(df_with_candle[candle_features].iloc[25:30])
    
    # 3.6 时间周期特征
    print("\n   3.6 计算时间周期特征（2维）...")
    df_with_time = calculator.calculate_time_features(df.copy())
    time_features = ['sin_tod', 'cos_tod']
    print(f"       特征: {time_features}")
    print(f"       示例值:")
    print(df_with_time[time_features].iloc[25:30])
    
    # 4. 计算所有特征
    print("\n4. 计算所有27维特征...")
    calculator_all = FeatureCalculator()
    df_all_features = calculator_all.calculate_all_features(df)
    
    print(f"   计算完成!")
    print(f"   特征数量: {len(calculator_all.feature_names)}")
    print(f"   数据形状: {df_all_features.shape}")
    print(f"   有效数据行数: {len(df_all_features)} (原始: {len(df)})")
    
    # 5. 查看特征分组
    print("\n5. 特征分组信息:")
    feature_groups = calculator_all.get_feature_groups()
    for group_name, features in feature_groups.items():
        print(f"   {group_name}: {len(features)}个特征")
        print(f"      {features}")
    
    # 6. 特征统计信息
    print("\n6. 特征统计信息:")
    stats = df_all_features[calculator_all.feature_names].describe()
    print(stats.T[['mean', 'std', 'min', 'max']])
    
    # 7. 检查数据质量
    print("\n7. 数据质量检查:")
    print(f"   缺失值数量: {df_all_features[calculator_all.feature_names].isna().sum().sum()}")
    print(f"   无穷值数量: {np.isinf(df_all_features[calculator_all.feature_names]).sum().sum()}")
    
    # 8. 保存结果（可选）
    output_file = 'data/processed/features_demo.csv'
    os.makedirs('data/processed', exist_ok=True)
    df_all_features.to_csv(output_file)
    print(f"\n8. 结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    
    return df_all_features


if __name__ == '__main__':
    try:
        df_result = demonstrate_feature_calculation()
        print(f"\n✓ 特征计算演示成功完成")
        print(f"  最终数据形状: {df_result.shape}")
        print(f"  特征列表: {df_result.columns.tolist()}")
    except Exception as e:
        logger.error(f"演示过程中出错: {e}", exc_info=True)
        print(f"\n✗ 演示失败: {e}")