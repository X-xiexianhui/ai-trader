"""
数据清洗模块使用示例

演示如何使用DataCleaner类进行OHLC数据清洗
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.cleaning import DataCleaner


def create_sample_data():
    """创建示例数据（包含各种质量问题）"""
    print("创建示例数据...")
    
    # 生成100根5分钟K线
    dates = pd.date_range('2023-01-01 09:00', periods=100, freq='5T')
    np.random.seed(42)
    
    # 生成价格序列
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    
    data = {
        'Open': close_prices * np.random.uniform(0.99, 1.00, 100),
        'High': close_prices * np.random.uniform(1.00, 1.02, 100),
        'Low': close_prices * np.random.uniform(0.98, 0.99, 100),
        'Close': close_prices,
        'Volume': np.random.uniform(1000, 5000, 100)
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # 人为添加一些数据质量问题
    print("\n添加数据质量问题:")
    
    # 1. 添加缺失值
    df.loc[df.index[10:13], 'Close'] = np.nan
    print(f"  - 在索引10-12添加了3个缺失值")
    
    # 2. 添加长期缺失段
    df.loc[df.index[30:38], ['Open', 'High', 'Low', 'Close']] = np.nan
    print(f"  - 在索引30-37添加了8个连续缺失值（将被删除）")
    
    # 3. 添加尖峰异常
    df.loc[df.index[50], 'Close'] = df.loc[df.index[50], 'Close'] * 2.5
    df.loc[df.index[50], 'High'] = df.loc[df.index[50], 'Close'] * 1.01
    print(f"  - 在索引50添加了尖峰异常（价格突然翻倍）")
    
    # 4. 破坏OHLC一致性
    df.loc[df.index[60], 'High'] = df.loc[df.index[60], 'Close'] * 0.95
    print(f"  - 在索引60破坏了OHLC一致性（High < Close）")
    
    # 5. 添加成交量缺失
    df.loc[df.index[70:73], 'Volume'] = np.nan
    print(f"  - 在索引70-72添加了成交量缺失值")
    
    print(f"\n原始数据: {len(df)}行")
    print(f"缺失值统计:")
    print(df.isna().sum())
    
    return df


def example_1_basic_usage():
    """示例1: 基本使用"""
    print("\n" + "="*60)
    print("示例1: 基本使用 - 单步处理")
    print("="*60)
    
    df = create_sample_data()
    
    # 创建清洗器
    cleaner = DataCleaner(
        max_consecutive_missing=5,
        interpolation_limit=3,
        sigma_threshold=3.0
    )
    
    # 1. 处理缺失值
    print("\n步骤1: 处理缺失值")
    df_cleaned, report = cleaner.handle_missing_values(df)
    print(f"  删除了{report['removed_rows']}行")
    print(f"  插值了{sum(report['interpolated_points'].values())}个点")
    
    # 2. 处理异常值
    print("\n步骤2: 处理异常值")
    df_cleaned, report = cleaner.detect_and_handle_outliers(df_cleaned)
    print(f"  检测到{report['total_outliers']}个异常值")
    print(f"  修正了{report['spike_outliers']}个尖峰")
    print(f"  保留了{report['gap_outliers']}个跳空")
    
    # 3. 时间对齐
    print("\n步骤3: 时间对齐")
    df_cleaned, report = cleaner.align_time_and_timezone(
        df_cleaned,
        target_timezone='UTC',
        interval_minutes=5
    )
    print(f"  时区: {report['original_timezone']} → {report['target_timezone']}")
    print(f"  行数: {report['rows_before']} → {report['rows_after']}")
    
    # 4. 质量验证
    print("\n步骤4: 质量验证")
    is_valid, report = cleaner.validate_data_quality(df_cleaned)
    print(f"  验证结果: {'通过✓' if is_valid else '失败✗'}")
    print(f"  问题数: {report['summary']['total_issues']}")
    print(f"  警告数: {report['summary']['total_warnings']}")
    
    return df_cleaned


def example_2_pipeline():
    """示例2: 使用完整流程"""
    print("\n" + "="*60)
    print("示例2: 使用完整清洗流程")
    print("="*60)
    
    df = create_sample_data()
    
    # 创建清洗器并运行完整流程
    cleaner = DataCleaner(
        max_consecutive_missing=5,
        interpolation_limit=3,
        sigma_threshold=3.0
    )
    
    # 一键清洗
    df_cleaned, full_report = cleaner.clean_pipeline(
        df,
        target_timezone='UTC',
        trading_hours=None,  # 24小时交易
        sigma_threshold=2.5  # 使用更严格的异常值检测
    )
    
    print(f"\n清洗完成!")
    print(f"  原始数据: {full_report['missing_values']['total_rows']}行")
    print(f"  最终数据: {len(df_cleaned)}行")
    print(f"  总耗时: {full_report['total_processing_time']:.3f}秒")
    print(f"  验证结果: {'通过✓' if full_report['final_validation'] else '失败✗'}")
    
    # 打印详细报告
    print("\n" + "-"*60)
    cleaner.print_report_summary()
    
    return df_cleaned


def example_3_custom_parameters():
    """示例3: 自定义参数"""
    print("\n" + "="*60)
    print("示例3: 自定义参数")
    print("="*60)
    
    df = create_sample_data()
    
    # 使用严格的清洗参数
    strict_cleaner = DataCleaner(
        max_consecutive_missing=3,  # 更严格：只允许3个连续缺失
        interpolation_limit=2,      # 更保守：只插值2个点
        sigma_threshold=2.0         # 更敏感：2σ就算异常
    )
    
    print("\n使用严格参数清洗...")
    df_strict, report = strict_cleaner.clean_pipeline(df)
    
    # 使用宽松的清洗参数
    loose_cleaner = DataCleaner(
        max_consecutive_missing=10,  # 更宽松：允许10个连续缺失
        interpolation_limit=5,       # 更激进：插值5个点
        sigma_threshold=4.0          # 更不敏感：4σ才算异常
    )
    
    print("\n使用宽松参数清洗...")
    df_loose, report = loose_cleaner.clean_pipeline(df)
    
    print(f"\n结果对比:")
    print(f"  严格清洗: {len(df_strict)}行")
    print(f"  宽松清洗: {len(df_loose)}行")
    
    return df_strict, df_loose


def example_4_trading_hours():
    """示例4: 交易时段过滤"""
    print("\n" + "="*60)
    print("示例4: 交易时段过滤")
    print("="*60)
    
    # 创建24小时数据
    dates = pd.date_range('2023-01-01', periods=288, freq='5T', tz='UTC')  # 24小时
    data = {
        'Open': np.random.uniform(100, 110, 288),
        'High': np.random.uniform(110, 120, 288),
        'Low': np.random.uniform(90, 100, 288),
        'Close': np.random.uniform(100, 110, 288),
        'Volume': np.random.uniform(1000, 5000, 288)
    }
    df = pd.DataFrame(data, index=dates)
    
    print(f"原始数据: {len(df)}行（24小时）")
    
    # 只保留美股交易时段（9:30-16:00 ET）
    cleaner = DataCleaner()
    df_cleaned, report = cleaner.align_time_and_timezone(
        df,
        target_timezone='America/New_York',
        trading_hours=(9, 16)  # 9:00-16:00
    )
    
    print(f"过滤后数据: {len(df_cleaned)}行（交易时段）")
    print(f"过滤掉: {report['filtered_non_trading']}行（非交易时段）")
    print(f"时区: {report['target_timezone']}")
    
    return df_cleaned


def example_5_error_handling():
    """示例5: 错误处理"""
    print("\n" + "="*60)
    print("示例5: 错误处理")
    print("="*60)
    
    cleaner = DataCleaner()
    
    # 测试1: 空DataFrame
    print("\n测试1: 空DataFrame")
    try:
        cleaner.handle_missing_values(pd.DataFrame())
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    # 测试2: 缺少必需列
    print("\n测试2: 缺少必需列")
    try:
        df = pd.DataFrame({'Open': [100], 'High': [105]})
        cleaner.handle_missing_values(df)
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    # 测试3: 无效参数
    print("\n测试3: 无效参数")
    try:
        DataCleaner(max_consecutive_missing=0)
    except ValueError as e:
        print(f"  捕获异常: {e}")
    
    print("\n所有错误都被正确处理✓")


def example_6_performance_test():
    """示例6: 性能测试"""
    print("\n" + "="*60)
    print("示例6: 性能测试（10万条数据）")
    print("="*60)
    
    import time
    
    # 创建10万条数据
    print("生成10万条数据...")
    dates = pd.date_range('2023-01-01', periods=100000, freq='1T', tz='UTC')
    close_prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.0001, 100000)))
    
    data = {
        'Open': close_prices * 0.999,
        'High': close_prices * 1.001,
        'Low': close_prices * 0.998,
        'Close': close_prices,
        'Volume': np.random.uniform(1000, 5000, 100000)
    }
    df = pd.DataFrame(data, index=dates)
    
    print(f"数据规模: {len(df)}行 × {len(df.columns)}列")
    
    # 测试性能
    cleaner = DataCleaner()
    
    print("\n开始清洗...")
    start_time = time.time()
    df_cleaned, report = cleaner.clean_pipeline(df)
    elapsed_time = time.time() - start_time
    
    print(f"\n性能结果:")
    print(f"  处理时间: {elapsed_time:.3f}秒")
    print(f"  处理速度: {len(df)/elapsed_time:.0f}行/秒")
    print(f"  性能要求: {'通过✓' if elapsed_time < 1.0 else '未达标✗'} (要求<1秒)")
    
    return df_cleaned


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("数据清洗模块使用示例")
    print("="*60)
    
    # 运行各个示例
    example_1_basic_usage()
    example_2_pipeline()
    example_3_custom_parameters()
    example_4_trading_hours()
    example_5_error_handling()
    example_6_performance_test()
    
    print("\n" + "="*60)
    print("所有示例运行完成!")
    print("="*60)


if __name__ == '__main__':
    main()