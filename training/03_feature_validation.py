"""
手工特征验证脚本

功能：
1. 加载归一化后的训练数据
2. 计算目标变量（未来收益）
3. 执行特征验证测试：
   - 单特征信息量测试
   - 置换重要性测试
   - 特征相关性检测
   - VIF多重共线性检测
4. 生成验证报告
5. 保留所有27个手工特征用于后续验证

使用方法：
    python training/03_feature_validation.py
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
from sklearn.linear_model import LinearRegression

from src.data.storage import DataStorage
from src.features.feature_calculator import FeatureCalculator
from src.features.feature_validator import FeatureValidator
from src.utils.logger import setup_logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def calculate_future_return(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """
    计算未来收益率作为目标变量
    
    Args:
        df: 包含Close列的DataFrame
        periods: 未来周期数
        
    Returns:
        未来收益率Series
    """
    future_close = df['Close'].shift(-periods)
    current_close = df['Close']
    
    # 计算对数收益率
    future_return = np.log(future_close / current_close)
    
    return future_return


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="feature_validation",
        log_file="03_feature_validation.log",
        log_level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("开始手工特征验证流程")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("\n步骤1: 加载配置文件")
    config = load_config()
    
    data_config = config['data']
    symbols = data_config['symbols']
    
    logger.info(f"品种: {symbols}")
    
    # 2. 加载归一化后的训练数据
    logger.info("\n步骤2: 加载归一化后的训练数据")
    processed_storage = DataStorage(base_path='data/processed')
    
    train_data_dict = {}
    for symbol in symbols:
        logger.info(f"\n加载 {symbol}...")
        
        df = processed_storage.load_parquet(f"{symbol}_train_normalized")
        
        if df is not None and not df.empty:
            train_data_dict[symbol] = df
            logger.info(f"✓ {symbol} 加载成功: {len(df)} 条记录")
        else:
            logger.warning(f"✗ {symbol} 加载失败")
    
    if not train_data_dict:
        logger.error("没有成功加载任何数据，退出")
        return
    
    # 3. 获取特征名称
    logger.info("\n步骤3: 获取特征名称")
    feature_calculator = FeatureCalculator()
    
    # 手工计算一次以获取特征名称
    sample_df = list(train_data_dict.values())[0]
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    sample_ohlcv = sample_df[ohlcv_cols].head(100)
    feature_calculator.calculate_all_features(sample_ohlcv)
    
    feature_names = feature_calculator.get_feature_names()
    feature_groups = feature_calculator.get_feature_groups()
    
    logger.info(f"手工特征数量: {len(feature_names)}")
    logger.info(f"特征列表: {feature_names}")
    
    # 4. 计算目标变量（未来收益）
    logger.info("\n步骤4: 计算目标变量")
    target_dict = {}
    
    for symbol, df in train_data_dict.items():
        # 计算未来1期收益率
        future_return = calculate_future_return(df, periods=1)
        target_dict[symbol] = future_return
        
        valid_count = future_return.notna().sum()
        logger.info(f"{symbol}: {valid_count} 个有效目标值")
    
    # 5. 特征验证
    logger.info("\n步骤5: 执行特征验证测试")
    
    validation_results = {}
    
    for symbol in train_data_dict.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"验证 {symbol}")
        logger.info(f"{'=' * 80}")
        
        df = train_data_dict[symbol]
        y = target_dict[symbol]
        
        # 提取特征和有效目标
        X = df[feature_names].copy()
        
        # 删除目标为NaN的行
        valid_mask = y.notna()
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        logger.info(f"有效样本数: {len(X_valid)}")
        
        # 创建验证器
        validator = FeatureValidator()
        
        # 5.1 单特征信息量测试
        logger.info("\n5.1 单特征信息量测试")
        try:
            info_results = validator.test_single_feature_information(
                X=X_valid,
                y=y_valid,
                top_n=10
            )
            logger.info("✓ 单特征信息量测试完成")
        except Exception as e:
            logger.error(f"✗ 单特征信息量测试失败: {e}")
            info_results = None
        
        # 5.2 置换重要性测试（使用简单线性回归模型）
        logger.info("\n5.2 置换重要性测试")
        try:
            # 训练一个简单的线性回归模型
            model = LinearRegression()
            model.fit(X_valid, y_valid)
            
            perm_results = validator.test_permutation_importance(
                model=model,
                X_val=X_valid,
                y_val=y_valid,
                n_repeats=50,  # 减少重复次数以加快速度
                random_state=42
            )
            logger.info("✓ 置换重要性测试完成")
        except Exception as e:
            logger.error(f"✗ 置换重要性测试失败: {e}")
            perm_results = None
        
        # 5.3 特征相关性检测
        logger.info("\n5.3 特征相关性检测")
        try:
            corr_matrix, high_corr_pairs = validator.test_feature_correlation(
                X=X_valid,
                threshold=0.85,
                plot=True,
                figsize=(14, 12)
            )
            logger.info("✓ 特征相关性检测完成")
        except Exception as e:
            logger.error(f"✗ 特征相关性检测失败: {e}")
            corr_matrix = None
            high_corr_pairs = []
        
        # 5.4 VIF多重共线性检测
        logger.info("\n5.4 VIF多重共线性检测")
        try:
            vif_results = validator.test_vif_multicollinearity(
                X=X_valid,
                threshold=10.0
            )
            logger.info("✓ VIF多重共线性检测完成")
        except Exception as e:
            logger.error(f"✗ VIF多重共线性检测失败: {e}")
            vif_results = None
        
        # 保存验证结果
        validation_results[symbol] = {
            'info_results': info_results,
            'perm_results': perm_results,
            'corr_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs,
            'vif_results': vif_results,
            'validator': validator
        }
        
        # 生成该品种的验证报告
        report_path = f"training/output/03_validation_{symbol}_report.txt"
        validator.generate_validation_report(output_path=report_path)
        logger.info(f"✓ {symbol} 验证报告已保存: {report_path}")
    
    # 6. 生成综合验证报告
    logger.info("\n步骤6: 生成综合验证报告")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("手工特征验证综合报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("验证说明:")
    report_lines.append("-" * 80)
    report_lines.append("本次验证保留所有27个手工特征，用于评估特征质量。")
    report_lines.append("验证包括以下测试:")
    report_lines.append("  1. 单特征信息量测试 - 评估每个特征的预测能力")
    report_lines.append("  2. 置换重要性测试 - 评估特征对模型性能的贡献")
    report_lines.append("  3. 特征相关性检测 - 识别高度相关的特征对")
    report_lines.append("  4. VIF多重共线性检测 - 检测特征间的多重共线性")
    report_lines.append("")
    
    # 汇总各品种的验证结果
    for symbol in train_data_dict.keys():
        report_lines.append(f"\n品种: {symbol}")
        report_lines.append("-" * 80)
        
        results = validation_results[symbol]
        
        # 单特征信息量
        if results['info_results'] is not None:
            info_df = results['info_results']
            report_lines.append("\n前10个最有信息量的特征:")
            for idx, row in info_df.head(10).iterrows():
                report_lines.append(
                    f"  {row['feature']:20s} - R²={row['r2_score']:.4f}, MI={row['mutual_info']:.4f}"
                )
        
        # 置换重要性
        if results['perm_results'] is not None:
            perm_df = results['perm_results']
            significant_features = perm_df[perm_df['is_significant']]
            report_lines.append(f"\n显著特征数量: {len(significant_features)}/{len(perm_df)} (p<0.05)")
            
            report_lines.append("\n前10个最重要的特征:")
            for idx, row in perm_df.head(10).iterrows():
                sig_mark = "***" if row['is_significant'] else ""
                report_lines.append(
                    f"  {row['feature']:20s} - 重要性={row['importance']:.6f} {sig_mark}"
                )
        
        # 高相关特征对
        if results['high_corr_pairs']:
            report_lines.append(f"\n高度相关特征对数量: {len(results['high_corr_pairs'])}")
            report_lines.append("前5对:")
            for feat1, feat2, corr in results['high_corr_pairs'][:5]:
                report_lines.append(f"  {feat1} <-> {feat2}: {corr:.4f}")
        else:
            report_lines.append("\n未发现高度相关特征对 (|ρ|>0.85)")
        
        # VIF结果
        if results['vif_results'] is not None:
            vif_df = results['vif_results']
            high_vif = vif_df[vif_df['has_multicollinearity']]
            report_lines.append(f"\n高VIF特征数量: {len(high_vif)}/{len(vif_df)} (VIF>10)")
            
            if len(high_vif) > 0:
                report_lines.append("高VIF特征:")
                for idx, row in high_vif.iterrows():
                    report_lines.append(f"  {row['feature']:20s} - VIF={row['VIF']:.2f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("验证结论:")
    report_lines.append("-" * 80)
    report_lines.append("所有27个手工特征已完成验证。")
    report_lines.append("详细的验证结果请查看各品种的独立报告。")
    report_lines.append("这些特征将在后续的特征选择中用于识别核心特征。")
    report_lines.append("=" * 80)
    
    # 保存综合报告
    report_path = "training/output/03_validation_summary.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n综合报告已保存: {report_path}")
    
    # 7. 保存特征验证结果（用于后续特征选择）
    logger.info("\n步骤7: 保存特征验证结果")
    
    import pickle
    
    validation_summary = {
        'feature_names': feature_names,
        'feature_groups': feature_groups,
        'validation_results': {}
    }
    
    for symbol, results in validation_results.items():
        validation_summary['validation_results'][symbol] = {
            'info_results': results['info_results'],
            'perm_results': results['perm_results'],
            'high_corr_pairs': results['high_corr_pairs'],
            'vif_results': results['vif_results']
        }
    
    summary_path = "training/output/03_validation_results.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(validation_summary, f)
    
    logger.info(f"验证结果已保存: {summary_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("手工特征验证流程完成！")
    logger.info("所有27个手工特征已保留用于后续验证")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()