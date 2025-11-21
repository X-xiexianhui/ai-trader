"""
完整的特征验证脚本 - 整合版

功能：
1. 读取原始MES数据
2. 数据清洗
3. 计算26维手工特征
4. 特征归一化
5. 执行完整的4项特征验证测试：
   - 测试1: 单特征信息量测试
   - 测试2: 特征相关性检测
   - 测试3: VIF多重共线性检测
   - 测试4: 置换重要性测试（需要模型）
6. 生成详细的验证报告
7. 分析并建议核心特征

使用方法:
    python training/validate_features.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.data_cleaner import DataCleaner
from src.features.feature_calculator import FeatureCalculator
from src.features.feature_scaler import FeatureScaler
from src.features.feature_validator import FeatureValidator
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('validate_features', log_dir='training/output', log_file='validate_features.log')


def load_raw_data(data_path: str) -> pd.DataFrame:
    """加载原始CSV数据"""
    logger.info("=" * 80)
    logger.info("步骤1: 加载原始数据")
    logger.info("=" * 80)
    logger.info(f"数据路径: {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 标准化列名
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_mapping)
    
    logger.info(f"数据加载完成: {len(df)}行 × {len(df.columns)}列")
    logger.info(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗数据"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 数据清洗")
    logger.info("=" * 80)
    
    cleaner = DataCleaner()
    df_clean, _ = cleaner.clean_pipeline(df)
    
    logger.info(f"数据清洗完成: {len(df_clean)}行")
    
    return df_clean


def calculate_features(df: pd.DataFrame) -> tuple:
    """计算26维特征"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 计算26维特征")
    logger.info("=" * 80)
    
    calculator = FeatureCalculator()
    df_features = calculator.calculate_all_features(df)
    
    feature_names = calculator.get_feature_names()
    feature_groups = calculator.get_feature_groups()
    
    logger.info(f"特征计算完成: {len(feature_names)}个特征，{len(df_features)}行数据")
    
    return df_features, feature_names, feature_groups


def normalize_features(df: pd.DataFrame, feature_groups: dict) -> tuple:
    """归一化特征"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤4: 特征归一化")
    logger.info("=" * 80)
    
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    X = df[all_features].copy()
    
    scaler = FeatureScaler()
    X_normalized = scaler.fit_transform(X, feature_groups)
    
    df_normalized = df.copy()
    df_normalized[all_features] = X_normalized
    
    logger.info(f"特征归一化完成: {len(all_features)}个特征")
    
    return df_normalized, scaler


def prepare_target_variable(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """准备目标变量"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤5: 准备目标变量")
    logger.info("=" * 80)
    logger.info(f"预测时间窗口: {horizon}个周期")
    
    future_close = df['Close'].shift(-horizon)
    current_close = df['Close']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.log(future_close / current_close)
    
    y = y.replace([np.inf, -np.inf], np.nan)
    valid_idx = ~y.isna()
    y = y[valid_idx]
    
    logger.info(f"目标变量准备完成: {len(y)}个有效样本")
    
    return y


def run_complete_validation(X: pd.DataFrame, y: pd.Series, feature_groups: dict) -> dict:
    """
    执行完整的4项特征验证测试
    """
    logger.info("\n" + "=" * 80)
    logger.info("步骤6: 完整特征验证（4项测试）")
    logger.info("=" * 80)
    
    # 对齐数据
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    
    # 删除NaN
    valid_idx = ~(X_aligned.isna().any(axis=1) | y_aligned.isna())
    X_clean = X_aligned[valid_idx]
    y_clean = y_aligned[valid_idx]
    
    logger.info(f"有效样本数: {len(X_clean)}")
    
    # ========================================================================
    # 测试1: 单特征信息量测试
    # ========================================================================
    logger.info("\n测试1: 单特征信息量测试...")
    
    feature_info = []
    for col in X_clean.columns:
        X_col = X_clean[[col]].values
        y_val = y_clean.values
        
        try:
            lr = LinearRegression()
            lr.fit(X_col, y_val)
            y_pred = lr.predict(X_col)
            r2 = r2_score(y_val, y_pred)
        except:
            r2 = 0.0
        
        try:
            mi = mutual_info_regression(X_col, y_val, random_state=42)[0]
        except:
            mi = 0.0
        
        # 找到特征所属的组
        group = None
        for group_name, features in feature_groups.items():
            if col in features:
                group = group_name
                break
        
        feature_info.append({
            'feature': col,
            'group': group,
            'r2_score': r2,
            'mutual_info': mi,
            'combined_score': r2 + mi
        })
    
    info_df = pd.DataFrame(feature_info)
    info_df = info_df.sort_values('combined_score', ascending=False)
    
    logger.info(f"完成: 计算了{len(info_df)}个特征的信息量")
    
    # ========================================================================
    # 测试2: 特征相关性检测
    # ========================================================================
    logger.info("\n测试2: 特征相关性检测...")
    
    corr_matrix = X_clean.corr()
    
    # 找出所有高相关对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.7:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    corr_pairs_df = pd.DataFrame(high_corr_pairs)
    if len(corr_pairs_df) > 0:
        corr_pairs_df = corr_pairs_df.sort_values('abs_correlation', ascending=False)
    
    logger.info(f"完成: 发现{len(corr_pairs_df)}对高相关特征 (|ρ| > 0.7)")
    
    # ========================================================================
    # 测试3: VIF多重共线性检测
    # ========================================================================
    logger.info("\n测试3: VIF多重共线性检测...")
    
    vif_data = []
    for i, col in enumerate(X_clean.columns):
        try:
            vif = variance_inflation_factor(X_clean.values, i)
            vif_data.append({
                'feature': col,
                'VIF': vif,
                'has_multicollinearity': vif > 10.0
            })
        except Exception as e:
            logger.warning(f"  计算{col}的VIF时出错: {e}")
            vif_data.append({
                'feature': col,
                'VIF': np.nan,
                'has_multicollinearity': False
            })
    
    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)
    
    high_vif_count = vif_df['has_multicollinearity'].sum()
    logger.info(f"完成: {high_vif_count}个特征存在多重共线性 (VIF > 10)")
    
    # ========================================================================
    # 测试4: 置换重要性测试
    # ========================================================================
    logger.info("\n测试4: 置换重要性测试...")
    logger.info("  注意: 此测试需要训练好的模型，当前跳过")
    
    return {
        'info_df': info_df,
        'corr_matrix': corr_matrix,
        'corr_pairs_df': corr_pairs_df,
        'vif_df': vif_df,
        'X_clean': X_clean,
        'y_clean': y_clean
    }


def generate_complete_report(validation_results: dict, feature_groups: dict, output_dir: str = 'training/output'):
    """生成完整的验证报告"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤7: 生成完整验证报告")
    logger.info("=" * 80)
    
    info_df = validation_results['info_df']
    corr_matrix = validation_results['corr_matrix']
    corr_pairs_df = validation_results['corr_pairs_df']
    vif_df = validation_results['vif_df']
    
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append("完整特征验证报告")
    report_lines.append("=" * 120)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"样本数量: {len(validation_results['X_clean'])}")
    report_lines.append(f"特征数量: {len(info_df)}")
    report_lines.append("")
    
    # ========================================================================
    # 测试1: 单特征信息量测试
    # ========================================================================
    report_lines.append("=" * 120)
    report_lines.append("测试1: 单特征信息量测试")
    report_lines.append("=" * 120)
    report_lines.append("")
    report_lines.append(f"{'排名':<6} {'特征名':<25} {'组别':<15} {'R²得分':<12} {'互信息':<12} {'综合得分':<12}")
    report_lines.append("-" * 120)
    
    for idx, row in info_df.iterrows():
        rank = list(info_df.index).index(idx) + 1
        report_lines.append(
            f"{rank:<6} {row['feature']:<25} {row['group']:<15} "
            f"{row['r2_score']:<12.6f} {row['mutual_info']:<12.6f} {row['combined_score']:<12.6f}"
        )
    
    report_lines.append("")
    
    # ========================================================================
    # 测试2: 特征相关性检测
    # ========================================================================
    report_lines.append("=" * 120)
    report_lines.append("测试2: 特征相关性检测")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    # 2.1 完整相关矩阵
    report_lines.append("2.1 完整相关矩阵")
    report_lines.append("-" * 120)
    report_lines.append("")
    corr_str = corr_matrix.to_string()
    report_lines.append(corr_str)
    report_lines.append("")
    
    # 2.2 高相关特征对
    report_lines.append("2.2 高相关特征对 (|ρ| > 0.7)")
    report_lines.append("-" * 120)
    
    if len(corr_pairs_df) > 0:
        report_lines.append(f"{'序号':<6} {'特征1':<25} {'特征2':<25} {'相关系数':<15}")
        report_lines.append("-" * 120)
        
        for idx, row in corr_pairs_df.iterrows():
            seq = list(corr_pairs_df.index).index(idx) + 1
            report_lines.append(
                f"{seq:<6} {row['feature1']:<25} {row['feature2']:<25} {row['correlation']:<15.6f}"
            )
        
        report_lines.append("")
        report_lines.append(f"总计: {len(corr_pairs_df)}对高相关特征")
    else:
        report_lines.append("未发现高相关特征对")
    
    report_lines.append("")
    
    # ========================================================================
    # 测试3: VIF多重共线性检测
    # ========================================================================
    report_lines.append("=" * 120)
    report_lines.append("测试3: VIF多重共线性检测")
    report_lines.append("=" * 120)
    report_lines.append("")
    report_lines.append(f"{'排名':<6} {'特征名':<25} {'VIF值':<15} {'存在共线性':<15}")
    report_lines.append("-" * 120)
    
    for idx, row in vif_df.iterrows():
        rank = list(vif_df.index).index(idx) + 1
        vif_str = f"{row['VIF']:.2f}" if not np.isnan(row['VIF']) and not np.isinf(row['VIF']) else "inf"
        multicollinearity = "是" if row['has_multicollinearity'] else "否"
        report_lines.append(
            f"{rank:<6} {row['feature']:<25} {vif_str:<15} {multicollinearity:<15}"
        )
    
    report_lines.append("")
    high_vif_count = vif_df['has_multicollinearity'].sum()
    report_lines.append(f"存在多重共线性的特征数量: {high_vif_count}/{len(vif_df)} (VIF > 10)")
    report_lines.append("")
    
    # ========================================================================
    # 测试4: 置换重要性测试
    # ========================================================================
    report_lines.append("=" * 120)
    report_lines.append("测试4: 置换重要性测试")
    report_lines.append("=" * 120)
    report_lines.append("")
    report_lines.append("注意: 置换重要性测试需要训练好的模型。")
    report_lines.append("建议在模型训练完成后，使用以下代码进行测试：")
    report_lines.append("")
    report_lines.append("```python")
    report_lines.append("from src.features.feature_validator import FeatureValidator")
    report_lines.append("")
    report_lines.append("validator = FeatureValidator()")
    report_lines.append("perm_importance = validator.test_permutation_importance(")
    report_lines.append("    model=trained_model,")
    report_lines.append("    X_val=X_validation,")
    report_lines.append("    y_val=y_validation,")
    report_lines.append("    n_repeats=100")
    report_lines.append(")")
    report_lines.append("```")
    report_lines.append("")
    
    # ========================================================================
    # 核心特征推荐
    # ========================================================================
    report_lines.append("=" * 120)
    report_lines.append("核心特征推荐")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    # 基于VIF和相关性分析，建议移除的特征
    features_to_remove = set()
    
    # 移除VIF > 100的特征
    for idx, row in vif_df.iterrows():
        if not np.isnan(row['VIF']) and not np.isinf(row['VIF']) and row['VIF'] > 100:
            features_to_remove.add(row['feature'])
    
    # 移除高相关对中信息量较低的
    if len(corr_pairs_df) > 0:
        for idx, row in corr_pairs_df.iterrows():
            if row['abs_correlation'] > 0.95:
                feat1_score = info_df[info_df['feature'] == row['feature1']]['combined_score'].values[0]
                feat2_score = info_df[info_df['feature'] == row['feature2']]['combined_score'].values[0]
                if feat1_score < feat2_score:
                    features_to_remove.add(row['feature1'])
                else:
                    features_to_remove.add(row['feature2'])
    
    # 核心特征列表
    all_features = info_df['feature'].tolist()
    core_features = [f for f in all_features if f not in features_to_remove]
    
    report_lines.append(f"原始特征数量: {len(all_features)}")
    report_lines.append(f"建议移除特征: {len(features_to_remove)}")
    report_lines.append(f"核心特征数量: {len(core_features)}")
    report_lines.append(f"特征保留率: {len(core_features)/len(all_features)*100:.1f}%")
    report_lines.append("")
    
    if features_to_remove:
        report_lines.append("建议移除的特征:")
        for feat in sorted(features_to_remove):
            report_lines.append(f"  - {feat}")
        report_lines.append("")
    
    report_lines.append("核心特征列表（按组别）:")
    for group_name, features in feature_groups.items():
        group_core = [f for f in features if f in core_features]
        if group_core:
            report_lines.append(f"\n{group_name}组 ({len(group_core)}个):")
            for feat in group_core:
                score = info_df[info_df['feature'] == feat]['combined_score'].values[0]
                report_lines.append(f"  - {feat} (得分: {score:.6f})")
    
    report_lines.append("")
    report_lines.append("=" * 120)
    report_lines.append("报告生成完成")
    report_lines.append("=" * 120)
    
    # 保存报告
    output_path = f'{output_dir}/complete_validation_report.txt'
    report_text = "\n".join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"完整验证报告已保存: {output_path}")
    
    # 保存CSV文件
    info_df.to_csv(f'{output_dir}/test1_feature_information.csv', index=False)
    corr_matrix.to_csv(f'{output_dir}/test2_correlation_matrix.csv')
    if len(corr_pairs_df) > 0:
        corr_pairs_df.to_csv(f'{output_dir}/test2_high_correlation_pairs.csv', index=False)
    vif_df.to_csv(f'{output_dir}/test3_vif_results.csv', index=False)
    
    logger.info("详细数据CSV文件已保存")
    
    return core_features, features_to_remove


def main():
    """主函数"""
    try:
        logger.info("\n" + "=" * 80)
        logger.info("完整特征验证流程开始")
        logger.info("=" * 80)
        
        # 1. 加载原始数据
        data_path = 'data/raw/MES_5m_20251121_233039.csv'
        df_raw = load_raw_data(data_path)
        
        # 2. 数据清洗
        df_clean = clean_data(df_raw)
        
        # 3. 计算特征
        df_features, feature_names, feature_groups = calculate_features(df_clean)
        
        # 4. 归一化特征
        df_normalized, scaler = normalize_features(df_features, feature_groups)
        
        # 保存归一化后的数据
        output_dir = Path('training/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_normalized.to_csv('training/output/mes_features_normalized.csv')
        df_normalized.to_parquet('training/output/mes_features_normalized.parquet')
        scaler.save('training/output/scalers')
        
        logger.info(f"\n归一化数据已保存")
        
        # 5. 准备目标变量
        y = prepare_target_variable(df_normalized, horizon=5)
        
        # 6. 执行完整验证
        all_features = []
        for features in feature_groups.values():
            all_features.extend(features)
        X = df_normalized[all_features]
        
        validation_results = run_complete_validation(X, y, feature_groups)
        
        # 7. 生成完整报告
        core_features, features_to_remove = generate_complete_report(
            validation_results, feature_groups
        )
        
        # 最终总结
        logger.info("\n" + "=" * 80)
        logger.info("✓ 完整特征验证流程完成！")
        logger.info("=" * 80)
        logger.info("\n生成的文件:")
        logger.info("  1. training/output/mes_features_normalized.csv - 归一化特征数据")
        logger.info("  2. training/output/mes_features_normalized.parquet - 归一化特征数据(Parquet)")
        logger.info("  3. training/output/scalers/ - 特征归一化器")
        logger.info("  4. training/output/complete_validation_report.txt - 完整验证报告")
        logger.info("  5. training/output/test1_feature_information.csv - 特征信息量数据")
        logger.info("  6. training/output/test2_correlation_matrix.csv - 相关矩阵")
        logger.info("  7. training/output/test2_high_correlation_pairs.csv - 高相关特征对")
        logger.info("  8. training/output/test3_vif_results.csv - VIF检测结果")
        logger.info("  9. feature_correlation_heatmap.png - 特征相关性热力图")
        logger.info(f"\n核心特征数量: {len(core_features)}/{len(all_features)}")
        logger.info(f"建议移除特征: {len(features_to_remove)}个")
        logger.info("\n请查看完整验证报告了解详细的特征分析结果！")
        
    except Exception as e:
        logger.error(f"\n✗ 特征验证失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()