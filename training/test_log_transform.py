"""
测试对数变换后的parkinson_vol特征

验证：
1. 特征计算是否正确
2. 对数变换是否生效
3. 分布是否得到改善
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from scipy import stats

from src.features.feature_calculator import FeatureCalculator

def main():
    print("=" * 80)
    print("测试对数变换后的parkinson_vol特征")
    print("=" * 80)
    
    # 1. 加载原始数据
    print("\n步骤1: 加载MES_F原始数据")
    data_path = Path("data/processed/MES_F.parquet")
    
    if not data_path.exists():
        print(f"错误: 文件不存在 {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"✓ 数据加载成功: {len(df)} 条记录")
    
    # 2. 计算特征（包含对数变换后的parkinson_vol）
    print("\n步骤2: 计算特征（包含对数变换）")
    feature_calculator = FeatureCalculator()
    df_with_features = feature_calculator.calculate_all_features(df)
    print(f"✓ 特征计算完成")
    print(f"  有效数据: {len(df_with_features)} 条")
    
    # 3. 提取对数变换后的parkinson_vol
    print("\n步骤3: 分析对数变换后的parkinson_vol")
    log_parkinson_vol = df_with_features['parkinson_vol'].dropna()
    
    print(f"\n对数变换后的parkinson_vol统计:")
    print(f"  样本数: {len(log_parkinson_vol)}")
    print(f"  均值: {log_parkinson_vol.mean():.6f}")
    print(f"  标准差: {log_parkinson_vol.std():.6f}")
    print(f"  最小值: {log_parkinson_vol.min():.6f}")
    print(f"  25%分位数: {log_parkinson_vol.quantile(0.25):.6f}")
    print(f"  50%分位数: {log_parkinson_vol.quantile(0.50):.6f}")
    print(f"  75%分位数: {log_parkinson_vol.quantile(0.75):.6f}")
    print(f"  最大值: {log_parkinson_vol.max():.6f}")
    print(f"  偏度: {stats.skew(log_parkinson_vol):.4f}")
    print(f"  峰度: {stats.kurtosis(log_parkinson_vol):.4f}")
    
    # 4. 验证对数变换
    print("\n步骤4: 验证对数变换")
    
    # 检查值的范围（对数变换后应该是负值，因为原始值很小）
    if log_parkinson_vol.max() < 0:
        print("✓ 对数变换已生效（所有值为负数，符合预期）")
    else:
        print("✗ 警告：存在非负值，可能对数变换未正确应用")
    
    # 检查偏度改善
    skewness = stats.skew(log_parkinson_vol)
    if abs(skewness) < 1.0:
        print(f"✓ 偏度改善明显（{skewness:.4f}，接近对称分布）")
    elif abs(skewness) < 2.0:
        print(f"✓ 偏度有所改善（{skewness:.4f}）")
    else:
        print(f"⚠ 偏度仍较大（{skewness:.4f}）")
    
    # 5. 对比原始值和对数变换后的值
    print("\n步骤5: 计算原始parkinson_vol用于对比")
    
    # 重新计算原始parkinson_vol（未对数变换）
    with np.errstate(divide='ignore', invalid='ignore'):
        high_low_ratio = df_with_features['High'] / df_with_features['Low']
        log_ratio = np.log(high_low_ratio)
        original_parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * log_ratio ** 2)
    original_parkinson_vol = original_parkinson_vol.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"\n原始parkinson_vol统计（未对数变换）:")
    print(f"  样本数: {len(original_parkinson_vol)}")
    print(f"  均值: {original_parkinson_vol.mean():.6f}")
    print(f"  标准差: {original_parkinson_vol.std():.6f}")
    print(f"  偏度: {stats.skew(original_parkinson_vol):.4f}")
    print(f"  峰度: {stats.kurtosis(original_parkinson_vol):.4f}")
    
    # 6. 验证对数关系
    print("\n步骤6: 验证对数关系")
    epsilon = 1e-10
    expected_log = np.log(original_parkinson_vol + epsilon)
    
    # 取相同索引的数据进行比较
    common_idx = log_parkinson_vol.index.intersection(expected_log.index)
    actual_log = log_parkinson_vol.loc[common_idx]
    expected_log = expected_log.loc[common_idx]
    
    # 计算差异
    diff = (actual_log - expected_log).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"  最大差异: {max_diff:.10f}")
    print(f"  平均差异: {mean_diff:.10f}")
    
    if max_diff < 1e-6:
        print("✓ 对数变换公式验证通过")
    else:
        print("✗ 警告：对数变换可能存在问题")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 7. 总结
    print("\n总结:")
    print(f"  原始parkinson_vol偏度: {stats.skew(original_parkinson_vol):.4f} (右偏)")
    print(f"  对数变换后偏度: {stats.skew(log_parkinson_vol):.4f} (改善)")
    print(f"  偏度改善幅度: {abs(stats.skew(original_parkinson_vol) - stats.skew(log_parkinson_vol)):.4f}")
    
    if abs(stats.skew(log_parkinson_vol)) < abs(stats.skew(original_parkinson_vol)):
        print("\n✓ 对数变换成功改善了分布的偏度")
    else:
        print("\n⚠ 对数变换未能改善偏度")


if __name__ == "__main__":
    main()