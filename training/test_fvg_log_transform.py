"""
测试对数变换后的FVG特征

验证：
1. FVG特征计算是否正确
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
    print("测试对数变换后的FVG特征")
    print("=" * 80)
    
    # 1. 加载原始数据
    print("\n步骤1: 加载MES_F原始数据")
    data_path = Path("data/processed/MES_F.parquet")
    
    if not data_path.exists():
        print(f"错误: 文件不存在 {data_path}")
        return
    
    df = pd.read_parquet(data_path)
    print(f"✓ 数据加载成功: {len(df)} 条记录")
    
    # 2. 计算特征（包含对数变换后的FVG）
    print("\n步骤2: 计算特征（包含对数变换）")
    feature_calculator = FeatureCalculator()
    df_with_features = feature_calculator.calculate_all_features(df)
    print(f"✓ 特征计算完成")
    print(f"  有效数据: {len(df_with_features)} 条")
    
    # 3. 提取对数变换后的FVG
    print("\n步骤3: 分析对数变换后的FVG")
    log_fvg = df_with_features['FVG'].dropna()
    
    # 统计零值和非零值
    zero_count = (log_fvg == np.log(1e-10)).sum()
    nonzero_count = (log_fvg != np.log(1e-10)).sum()
    
    print(f"\n对数变换后的FVG统计:")
    print(f"  样本数: {len(log_fvg)}")
    print(f"  零值数量: {zero_count} ({zero_count/len(log_fvg)*100:.2f}%)")
    print(f"  非零值数量: {nonzero_count} ({nonzero_count/len(log_fvg)*100:.2f}%)")
    print(f"  均值: {log_fvg.mean():.6f}")
    print(f"  标准差: {log_fvg.std():.6f}")
    print(f"  最小值: {log_fvg.min():.6f}")
    print(f"  25%分位数: {log_fvg.quantile(0.25):.6f}")
    print(f"  50%分位数: {log_fvg.quantile(0.50):.6f}")
    print(f"  75%分位数: {log_fvg.quantile(0.75):.6f}")
    print(f"  最大值: {log_fvg.max():.6f}")
    print(f"  偏度: {stats.skew(log_fvg):.4f}")
    print(f"  峰度: {stats.kurtosis(log_fvg):.4f}")
    
    # 4. 验证对数变换
    print("\n步骤4: 验证对数变换")
    
    # 检查非零值的分布
    log_fvg_nonzero = log_fvg[log_fvg != np.log(1e-10)]
    if len(log_fvg_nonzero) > 0:
        print(f"\n非零值统计:")
        print(f"  样本数: {len(log_fvg_nonzero)}")
        print(f"  均值: {log_fvg_nonzero.mean():.6f}")
        print(f"  标准差: {log_fvg_nonzero.std():.6f}")
        print(f"  偏度: {stats.skew(log_fvg_nonzero):.4f}")
        print(f"  峰度: {stats.kurtosis(log_fvg_nonzero):.4f}")
        
        # 检查偏度改善
        skewness = stats.skew(log_fvg_nonzero)
        if abs(skewness) < 0.5:
            print(f"✓ 偏度改善显著（{skewness:.4f}，接近对称分布）")
        elif abs(skewness) < 1.0:
            print(f"✓ 偏度有所改善（{skewness:.4f}）")
        else:
            print(f"⚠ 偏度仍较大（{skewness:.4f}）")
    
    # 5. 计算原始FVG用于对比
    print("\n步骤5: 计算原始FVG用于对比")
    
    # 重新计算原始FVG（未对数变换）
    original_fvg = pd.Series(0.0, index=df_with_features.index)
    
    for i in range(2, len(df_with_features)):
        try:
            high_1 = df_with_features['High'].iloc[i-2]
            low_1 = df_with_features['Low'].iloc[i-2]
            high_3 = df_with_features['High'].iloc[i]
            low_3 = df_with_features['Low'].iloc[i]
            close_current = df_with_features['Close'].iloc[i]
            
            if np.isnan([high_1, low_1, high_3, low_3, close_current]).any():
                continue
            if close_current <= 0:
                continue
            
            if high_1 < low_3:
                gap_size = low_3 - high_1
                fvg_strength = gap_size / close_current
                original_fvg.iloc[i] = fvg_strength
            elif low_1 > high_3:
                gap_size = low_1 - high_3
                fvg_strength = -gap_size / close_current
                original_fvg.iloc[i] = fvg_strength
        except:
            continue
    
    original_fvg_nonzero = original_fvg[original_fvg != 0]
    
    print(f"\n原始FVG统计（仅非零值）:")
    print(f"  样本数: {len(original_fvg_nonzero)}")
    print(f"  均值: {original_fvg_nonzero.mean():.6f}")
    print(f"  标准差: {original_fvg_nonzero.std():.6f}")
    print(f"  偏度: {stats.skew(original_fvg_nonzero):.4f}")
    print(f"  峰度: {stats.kurtosis(original_fvg_nonzero):.4f}")
    
    # 6. 验证对数关系
    print("\n步骤6: 验证对数关系")
    epsilon = 1e-10
    
    # 对原始非零FVG进行对数变换
    expected_log = np.sign(original_fvg_nonzero) * np.log(np.abs(original_fvg_nonzero) + epsilon)
    
    # 获取对应的对数变换后的值（排除零值）
    nonzero_indices = original_fvg[original_fvg != 0].index
    actual_log = log_fvg.loc[nonzero_indices]
    
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
    print(f"  原始FVG偏度（非零）: {stats.skew(original_fvg_nonzero):.4f}")
    print(f"  对数变换后偏度（非零）: {stats.skew(log_fvg_nonzero):.4f}")
    print(f"  偏度改善幅度: {abs(stats.skew(original_fvg_nonzero) - stats.skew(log_fvg_nonzero)):.4f}")
    
    print(f"\n  原始FVG峰度（非零）: {stats.kurtosis(original_fvg_nonzero):.4f}")
    print(f"  对数变换后峰度（非零）: {stats.kurtosis(log_fvg_nonzero):.4f}")
    print(f"  峰度改善幅度: {abs(stats.kurtosis(original_fvg_nonzero) - stats.kurtosis(log_fvg_nonzero)):.4f}")
    
    if abs(stats.skew(log_fvg_nonzero)) < abs(stats.skew(original_fvg_nonzero)):
        print("\n✓ 对数变换成功改善了FVG的偏度")
    else:
        print("\n⚠ 对数变换未能改善偏度")
    
    if abs(stats.kurtosis(log_fvg_nonzero)) < abs(stats.kurtosis(original_fvg_nonzero)):
        print("✓ 对数变换成功改善了FVG的峰度")
    else:
        print("⚠ 对数变换未能改善峰度")


if __name__ == "__main__":
    main()