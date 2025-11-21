"""
MES 5分钟K线数据处理Demo

功能：
1. 获取MES（微型标普500期货）5分钟K线数据（最近2年）
2. 分批次下载（每批次59天），清洗后保存到本地
3. 执行步骤1-4：数据清洗、归一化、特征验证、核心特征选择

使用方法：
    python training/demo_mes_pipeline.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import pickle
import time

from src.data.downloader import DataDownloader
from src.data.storage import DataStorage
from src.features.data_cleaner import DataCleaner
from src.features.feature_calculator import FeatureCalculator
from src.features.feature_scaler import FeatureScaler
from src.features.feature_validator import FeatureValidator
from src.utils.logger import setup_logger
from sklearn.linear_model import LinearRegression


def calculate_future_return(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """计算未来收益率"""
    future_close = df['Close'].shift(-periods)
    current_close = df['Close']
    future_return = np.log(future_close / current_close)
    return future_return


def split_train_test(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
    """按时间顺序划分训练集和测试集"""
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def download_and_clean_data_in_batches(
    symbol: str,
    interval: str,
    days_back: int = 59,  # yfinance 5分钟数据限制为60天
    batch_days: int = 59,
    storage_path: str = "data/processed"
) -> pd.DataFrame:
    """
    分批次下载并清洗数据
    
    注意：yfinance的5分钟数据只能获取最近60天的数据
    
    Args:
        symbol: 品种代码
        interval: 时间周期
        days_back: 回溯天数（默认59天，5分钟数据最多60天）
        batch_days: 每批次天数（默认59天）
        storage_path: 存储路径
        
    Returns:
        清洗后的完整数据
    """
    storage = DataStorage(base_path=storage_path)
    
    # 检查本地是否已有数据
    local_data = storage.load_parquet(symbol.replace('=', '_'))
    
    if local_data is not None and not local_data.empty:
        print(f"✓ 从本地加载数据: {len(local_data)} 条记录")
        print(f"  时间范围: {local_data.index[0]} 到 {local_data.index[-1]}")
        return local_data
    
    print(f"\n本地无数据，开始下载最近{days_back}天的数据...")
    print(f"注意：yfinance的5分钟数据只能获取最近60天")
    
    # 初始化下载器和清洗器
    downloader = DataDownloader(max_retries=3, retry_delay=5)
    cleaner = DataCleaner(
        max_consecutive_missing=5,
        sigma_threshold=3.0
    )
    
    # 计算批次
    end_date = datetime.now()
    all_data = []
    batch_count = (days_back + batch_days - 1) // batch_days  # 向上取整
    
    print(f"总批次数: {batch_count}")
    print("")
    
    for i in range(batch_count):
        batch_end = end_date - timedelta(days=i * batch_days)
        batch_start = batch_end - timedelta(days=batch_days - 1)
        
        print(f"批次 {i+1}/{batch_count}: {batch_start.strftime('%Y-%m-%d')} 到 {batch_end.strftime('%Y-%m-%d')}")
        
        try:
            # 下载数据
            batch_df = downloader.download(
                symbol=symbol,
                start_date=batch_start.strftime('%Y-%m-%d'),
                end_date=batch_end.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if batch_df is None or batch_df.empty:
                print(f"  ⚠ 批次 {i+1} 下载失败或无数据")
                continue
            
            # 标准化列名
            batch_df.columns = [col.capitalize() for col in batch_df.columns]
            
            # 清洗数据
            cleaned_batch, _ = cleaner.clean_pipeline(
                df=batch_df,
                target_timezone='UTC',
                trading_hours=None
            )
            
            if not cleaned_batch.empty:
                all_data.append(cleaned_batch)
                print(f"  ✓ 下载并清洗: {len(cleaned_batch)} 条记录")
            else:
                print(f"  ⚠ 批次 {i+1} 清洗后无数据")
            
            # 避免请求过快
            if i < batch_count - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"  ✗ 批次 {i+1} 处理失败: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("所有批次下载失败，无法获取数据")
    
    # 合并所有批次数据
    print(f"\n合并 {len(all_data)} 个批次的数据...")
    combined_df = pd.concat(all_data, axis=0)
    
    # 去重并排序
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df = combined_df.sort_index()
    
    print(f"✓ 合并完成: {len(combined_df)} 条记录")
    print(f"  时间范围: {combined_df.index[0]} 到 {combined_df.index[-1]}")
    
    # 保存到本地
    print(f"\n保存清洗后的数据到本地...")
    save_symbol = symbol.replace('=', '_')  # 文件名不能包含=
    success = storage.save_parquet(combined_df, save_symbol, compression='snappy')
    
    if success:
        print(f"✓ 数据已保存到: {storage_path}/{save_symbol}.parquet")
    else:
        print(f"⚠ 数据保存失败")
    
    return combined_df


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="demo_mes",
        log_file="demo_mes_pipeline.log",
        log_level="INFO"
    )
    
    print("=" * 80)
    print("MES 5分钟K线数据处理Demo")
    print("=" * 80)
    
    # 配置
    symbol = "MES=F"  # 微型标普500期货
    interval = "5m"
    days_back = 59  # yfinance 5分钟数据限制为60天
    
    print(f"\n品种: {symbol}")
    print(f"周期: {interval}")
    print(f"数据范围: 最近{days_back}天（yfinance 5分钟数据限制）")
    
    # ========================================================================
    # 步骤1: 数据获取和清洗
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤1: 数据获取和清洗")
    print("=" * 80)
    
    # 1.1 分批次下载并清洗数据（如果本地不存在）
    print("\n1.1 获取MES数据...")
    try:
        cleaned_df = download_and_clean_data_in_batches(
            symbol=symbol,
            interval=interval,
            days_back=days_back,
            batch_days=59,
            storage_path="data/processed"
        )
    except Exception as e:
        print(f"✗ 数据获取失败: {str(e)}")
        return
    
    if cleaned_df is None or cleaned_df.empty:
        print(f"✗ 无法获取有效数据")
        return
    
    print(f"\n✓ 数据准备完成: {len(cleaned_df)} 条记录")
    print(f"  时间范围: {cleaned_df.index[0]} 到 {cleaned_df.index[-1]}")
    
    # 1.3 计算手工特征
    print("\n1.3 计算27维手工特征...")
    feature_calculator = FeatureCalculator()
    df_with_features = feature_calculator.calculate_all_features(cleaned_df)
    
    feature_names = feature_calculator.get_feature_names()
    feature_groups = feature_calculator.get_feature_groups()
    
    print(f"✓ 特征计算完成")
    print(f"  数据行数: {len(df_with_features)}")
    print(f"  特征数量: {len(feature_names)}")
    print(f"  特征列表: {feature_names}")
    
    # ========================================================================
    # 步骤2: 特征归一化
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤2: 特征归一化")
    print("=" * 80)
    
    # 2.1 划分训练集和测试集
    print("\n2.1 划分训练集和测试集...")
    train_df, test_df = split_train_test(df_with_features, test_ratio=0.2)
    
    print(f"训练集: {len(train_df)} 条 ({train_df.index[0]} 到 {train_df.index[-1]})")
    print(f"测试集: {len(test_df)} 条 ({test_df.index[0]} 到 {test_df.index[-1]})")
    
    # 2.2 特征归一化
    print("\n2.2 特征归一化...")
    X_train = train_df[feature_names].copy()
    X_test = test_df[feature_names].copy()
    
    feature_scaler = FeatureScaler()
    feature_scaler.fit(X_train, feature_groups)
    
    X_train_scaled = feature_scaler.transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    train_normalized = train_df.copy()
    train_normalized[feature_names] = X_train_scaled
    
    test_normalized = test_df.copy()
    test_normalized[feature_names] = X_test_scaled
    
    print(f"✓ 归一化完成")
    print(f"  训练集特征范围: [{X_train_scaled.min().min():.4f}, {X_train_scaled.max().max():.4f}]")
    print(f"  测试集特征范围: [{X_test_scaled.min().min():.4f}, {X_test_scaled.max().max():.4f}]")
    
    # ========================================================================
    # 步骤3: 手工特征验证
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤3: 手工特征验证")
    print("=" * 80)
    
    # 3.1 计算目标变量
    print("\n3.1 计算目标变量...")
    y_train = calculate_future_return(train_normalized, periods=1)
    y_test = calculate_future_return(test_normalized, periods=1)
    
    # 提取有效样本
    train_valid_mask = y_train.notna()
    X_train_valid = X_train_scaled[train_valid_mask]
    y_train_valid = y_train[train_valid_mask]
    
    print(f"有效训练样本: {len(X_train_valid)}")
    
    # 3.2 特征验证
    print("\n3.2 执行特征验证测试...")
    validator = FeatureValidator()
    
    # 单特征信息量测试
    print("\n  3.2.1 单特征信息量测试...")
    info_results = validator.test_single_feature_information(
        X=X_train_valid,
        y=y_train_valid,
        top_n=10
    )
    print(f"  ✓ 完成")
    
    # 置换重要性测试
    print("\n  3.2.2 置换重要性测试...")
    model = LinearRegression()
    model.fit(X_train_valid, y_train_valid)
    
    perm_results = validator.test_permutation_importance(
        model=model,
        X_val=X_train_valid,
        y_val=y_train_valid,
        n_repeats=30,  # 减少重复次数以加快速度
        random_state=42
    )
    print(f"  ✓ 完成")
    
    # 特征相关性检测
    print("\n  3.2.3 特征相关性检测...")
    corr_matrix, high_corr_pairs = validator.test_feature_correlation(
        X=X_train_valid,
        threshold=0.85,
        plot=False  # Demo中不生成图片
    )
    print(f"  ✓ 完成，发现 {len(high_corr_pairs)} 对高相关特征")
    
    # VIF多重共线性检测
    print("\n  3.2.4 VIF多重共线性检测...")
    try:
        vif_results = validator.test_vif_multicollinearity(
            X=X_train_valid,
            threshold=10.0
        )
        print(f"  ✓ 完成")
    except ModuleNotFoundError as e:
        print(f"  ⚠ 跳过（缺少statsmodels模块）")
        vif_results = pd.DataFrame()  # 空DataFrame
    except Exception as e:
        print(f"  ⚠ 跳过（错误: {e}）")
        vif_results = pd.DataFrame()
    
    # ========================================================================
    # 步骤4: 核心特征选择
    # ========================================================================
    print("\n" + "=" * 80)
    print("步骤4: 核心特征选择")
    print("=" * 80)
    
    print("\n4.1 基于验证结果选择核心特征...")
    
    # 初始化候选特征
    candidate_features = set(feature_names)
    
    # 基于置换重要性筛选
    significant_features = perm_results[
        (perm_results['is_significant']) & 
        (perm_results['importance'] > 0.0001)
    ]['feature'].tolist()
    
    candidate_features = candidate_features.intersection(set(significant_features))
    print(f"  置换重要性筛选后: {len(candidate_features)} 个特征")
    
    # 移除高相关特征
    features_to_remove = set()
    for feat1, feat2, corr in high_corr_pairs:
        if corr > 0.85 and feat1 in candidate_features and feat2 in candidate_features:
            imp1 = perm_results[perm_results['feature'] == feat1]['importance'].values
            imp2 = perm_results[perm_results['feature'] == feat2]['importance'].values
            
            if len(imp1) > 0 and len(imp2) > 0:
                if imp1[0] < imp2[0]:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
    
    candidate_features = candidate_features - features_to_remove
    print(f"  移除高相关特征后: {len(candidate_features)} 个特征")
    
    # 移除高VIF特征
    if not vif_results.empty and 'VIF' in vif_results.columns:
        high_vif_features = vif_results[vif_results['VIF'] > 10.0].sort_values('VIF', ascending=False)['feature'].tolist()
        
        for feat in high_vif_features:
            if feat in candidate_features and len(candidate_features) > 10:
                candidate_features.discard(feat)
        
        print(f"  移除高VIF特征后: {len(candidate_features)} 个特征")
    else:
        print(f"  跳过VIF筛选（VIF测试未执行）")
    
    # 如果特征太少，补充信息量最高的
    if len(candidate_features) < 10:
        top_features = info_results.head(20)['feature'].tolist()
        for feat in top_features:
            if len(candidate_features) >= 10:
                break
            if feat in feature_names:
                candidate_features.add(feat)
        print(f"  补充信息量高的特征后: {len(candidate_features)} 个特征")
    
    # 如果特征太多，保留重要性最高的
    if len(candidate_features) > 20:
        candidate_importance = perm_results[
            perm_results['feature'].isin(candidate_features)
        ].sort_values('importance', ascending=False)
        
        candidate_features = set(candidate_importance.head(20)['feature'].tolist())
        print(f"  限制最大特征数后: {len(candidate_features)} 个特征")
    
    core_features = sorted(list(candidate_features))
    
    print(f"\n✓ 核心特征选择完成: {len(core_features)} 个特征")
    print(f"  核心特征列表: {core_features}")
    
    # 4.2 保存核心特征到JSON
    print("\n4.2 保存核心特征到JSON...")
    
    output_dir = Path("training/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    core_features_json = {
        'metadata': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'interval': interval,
            'description': 'MES Demo核心特征'
        },
        'all_features': feature_names,
        'core_features': {
            symbol: core_features
        }
    }
    
    json_path = output_dir / "mes_core_features.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(core_features_json, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 核心特征已保存: {json_path}")
    
    # ========================================================================
    # 生成Demo报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("生成Demo报告")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MES 5分钟K线数据处理Demo报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("数据信息:")
    report_lines.append("-" * 80)
    report_lines.append(f"品种: {symbol}")
    report_lines.append(f"周期: {interval}")
    report_lines.append(f"数据范围: 最近{days_back}天")
    report_lines.append(f"清洗后数据: {len(cleaned_df)} 条")
    report_lines.append(f"特征数据: {len(df_with_features)} 条")
    report_lines.append(f"训练集: {len(train_df)} 条")
    report_lines.append(f"测试集: {len(test_df)} 条")
    report_lines.append("")
    
    report_lines.append("特征信息:")
    report_lines.append("-" * 80)
    report_lines.append(f"所有手工特征: {len(feature_names)} 个")
    report_lines.append(f"核心特征: {len(core_features)} 个")
    report_lines.append("")
    
    # ========================================================================
    # 详细的4个检验结果
    # ========================================================================
    report_lines.append("=" * 80)
    report_lines.append("特征验证详细结果")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 检验1: 单特征信息量测试
    report_lines.append("1. 单特征信息量测试 (R²和互信息)")
    report_lines.append("-" * 80)
    report_lines.append("评估每个特征单独预测目标变量的能力")
    report_lines.append("")
    report_lines.append("前10个最有信息量的特征:")
    for idx, row in info_results.head(10).iterrows():
        report_lines.append(
            f"  {row['feature']:25s} - R²={row['r2_score']:.6f}, "
            f"MI={row['mutual_info']:.6f}, 综合={row['combined_score']:.6f}"
        )
    report_lines.append("")
    
    # 检验2: 置换重要性测试
    report_lines.append("2. 置换重要性测试 (统计显著性)")
    report_lines.append("-" * 80)
    report_lines.append("通过置换特征评估其对模型性能的贡献")
    report_lines.append("")
    significant_perm_features = perm_results[perm_results['is_significant']]
    significant_count = len(significant_perm_features)
    report_lines.append(f"显著特征数量: {significant_count}/{len(feature_names)} (p<0.05)")
    report_lines.append("")
    report_lines.append("前10个最重要的特征:")
    for idx, row in perm_results.head(10).iterrows():
        sig_mark = " ***" if row['is_significant'] else ""
        report_lines.append(
            f"  {row['feature']:25s} - 重要性={row['importance']:.6f}, "
            f"p值={row['p_value']:.4f}{sig_mark}"
        )
    report_lines.append("")
    
    # 检验3: 特征相关性检测
    report_lines.append("3. 特征相关性检测 (Pearson相关)")
    report_lines.append("-" * 80)
    report_lines.append("识别高度相关的特征对，避免冗余")
    report_lines.append("")
    report_lines.append(f"高相关特征对数量: {len(high_corr_pairs)} (|ρ|>0.85)")
    if high_corr_pairs:
        report_lines.append("")
        report_lines.append("高相关特征对:")
        for feat1, feat2, corr in high_corr_pairs[:10]:
            report_lines.append(f"  {feat1:25s} <-> {feat2:25s} : ρ={corr:.4f}")
    else:
        report_lines.append("未发现高度相关的特征对")
    report_lines.append("")
    
    # 检验4: VIF多重共线性检测
    report_lines.append("4. VIF多重共线性检测 (方差膨胀因子)")
    report_lines.append("-" * 80)
    report_lines.append("检测特征间的多重共线性问题")
    report_lines.append("")
    
    if not vif_results.empty and 'VIF' in vif_results.columns:
        high_vif = vif_results[vif_results['VIF'] > 10.0]
        report_lines.append(f"高VIF特征数量: {len(high_vif)}/{len(feature_names)} (VIF>10)")
        
        if len(high_vif) > 0:
            report_lines.append("")
            report_lines.append("高VIF特征:")
            for idx, row in high_vif.iterrows():
                report_lines.append(f"  {row['feature']:25s} - VIF={row['VIF']:.2f}")
        else:
            report_lines.append("未发现多重共线性问题")
        
        report_lines.append("")
        report_lines.append("所有特征的VIF值:")
        for idx, row in vif_results.sort_values('VIF', ascending=False).head(15).iterrows():
            status = "⚠" if row['VIF'] > 10.0 else "✓"
            report_lines.append(f"  {status} {row['feature']:25s} - VIF={row['VIF']:.2f}")
    else:
        report_lines.append("VIF测试未执行（缺少statsmodels模块）")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 核心特征列表
    report_lines.append("核心特征选择结果:")
    report_lines.append("-" * 80)
    report_lines.append(f"从{len(feature_names)}个特征中选择了{len(core_features)}个核心特征")
    report_lines.append("")
    report_lines.append("核心特征列表:")
    for i, feat in enumerate(core_features, 1):
        # 获取特征的详细信息
        feat_row = perm_results[perm_results['feature'] == feat]
        info_row = info_results[info_results['feature'] == feat]
        
        if not feat_row.empty and not info_row.empty:
            imp = feat_row['importance'].values[0]
            p_val = feat_row['p_value'].values[0]
            r2 = info_row['r2_score'].values[0]
            mi = info_row['mutual_info'].values[0]
            
            report_lines.append(
                f"  {i:2d}. {feat:25s} - 重要性={imp:.6f}, p={p_val:.4f}, "
                f"R²={r2:.6f}, MI={mi:.6f}"
            )
        else:
            report_lines.append(f"  {i:2d}. {feat}")
    
    report_lines.append("")
    report_lines.append("选择标准:")
    report_lines.append("  ✓ 置换重要性显著 (p<0.05)")
    report_lines.append("  ✓ 重要性 > 0.0001")
    report_lines.append("  ✓ 移除高相关特征 (|ρ|>0.85)")
    report_lines.append("  ✓ 移除高VIF特征 (VIF>10)")
    report_lines.append("  ✓ 特征数量控制在10-20个")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("Demo完成！")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = output_dir / "mes_demo_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n报告已保存: {report_path}")
    
    # 输出摘要
    print("\n" + "=" * 80)
    print("Demo执行摘要")
    print("=" * 80)
    print(f"✓ 步骤1: 数据获取和清洗 - 完成")
    print(f"  - 清洗后数据: {len(cleaned_df)} 条")
    print(f"  - 计算特征: {len(feature_names)} 个")
    print(f"\n✓ 步骤2: 特征归一化 - 完成")
    print(f"  - 训练集: {len(train_df)} 条")
    print(f"  - 测试集: {len(test_df)} 条")
    print(f"\n✓ 步骤3: 手工特征验证 - 完成")
    print(f"  - 显著特征: {significant_count}/{len(feature_names)} 个")
    print(f"\n✓ 步骤4: 核心特征选择 - 完成")
    print(f"  - 核心特征: {len(core_features)}/{len(feature_names)} 个")
    print(f"  - 已保存到: {json_path}")
    print("\n" + "=" * 80)
    print("所有步骤执行成功！")
    print("=" * 80)


if __name__ == "__main__":
    main()