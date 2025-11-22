"""
MES期货数据特征工程和验证脚本

功能：
1. 读取清洗后的数据
2. 计算23维手工特征
3. 归一化特征
4. 执行特征验证测试
5. 生成详细特征检测报告
6. 给出核心特征保留建议
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import warnings

from src.features.feature_calculator import FeatureCalculator
from src.features.feature_scaler import FeatureScaler
from src.features.feature_validator import FeatureValidator

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'feature_engineering.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_cleaned_data(data_path: Path) -> pd.DataFrame:
    """
    加载清洗后的数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        清洗后的DataFrame
    """
    logger.info(f"加载清洗后的数据: {data_path}")
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    logger.info(f"数据加载完成: {len(df)}行, {len(df.columns)}列")
    logger.info(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    logger.info(f"列名: {df.columns.tolist()}")
    
    return df


def calculate_features(df: pd.DataFrame) -> tuple:
    """
    计算23维手工特征
    
    Args:
        df: 清洗后的OHLCV数据
        
    Returns:
        (带特征的DataFrame, 特征计算器实例)
    """
    logger.info("=" * 80)
    logger.info("开始计算23维手工特征")
    logger.info("=" * 80)
    
    calculator = FeatureCalculator()
    
    # 计算所有特征
    df_with_features = calculator.calculate_all_features(df)
    
    # 获取特征名称和分组
    feature_names = calculator.get_feature_names()
    feature_groups = calculator.get_feature_groups()
    
    logger.info(f"\n特征计算完成:")
    logger.info(f"  - 总特征数: {len(feature_names)}")
    logger.info(f"  - 数据行数: {len(df_with_features)}")
    logger.info(f"  - 特征分组:")
    for group_name, features in feature_groups.items():
        logger.info(f"    * {group_name}: {len(features)}个特征")
        logger.info(f"      {features}")
    
    return df_with_features, calculator


def normalize_features(df: pd.DataFrame, feature_groups: dict) -> tuple:
    """
    归一化特征
    
    Args:
        df: 带特征的DataFrame
        feature_groups: 特征分组字典
        
    Returns:
        (归一化后的DataFrame, 特征归一化器实例)
    """
    logger.info("=" * 80)
    logger.info("开始特征归一化")
    logger.info("=" * 80)
    
    # 提取特征列
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    X = df[all_features].copy()
    
    # 创建并拟合归一化器
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X, feature_groups)
    
    # 创建归一化后的完整DataFrame
    df_normalized = df.copy()
    df_normalized[all_features] = X_scaled
    
    logger.info(f"\n归一化完成:")
    logger.info(f"  - 使用的归一化器:")
    for group_name, group_scaler in scaler.scalers.items():
        scaler_type = type(group_scaler).__name__
        logger.info(f"    * {group_name}: {scaler_type}")
    
    # 保存归一化器
    scaler_dir = project_root / 'models' / 'scalers'
    scaler.save(scaler_dir)
    logger.info(f"  - 归一化器已保存到: {scaler_dir}")
    
    return df_normalized, scaler


def validate_features(df: pd.DataFrame, feature_groups: dict) -> dict:
    """
    执行特征验证测试
    
    Args:
        df: 归一化后的DataFrame
        feature_groups: 特征分组字典
        
    Returns:
        验证结果字典
    """
    logger.info("=" * 80)
    logger.info("开始特征验证测试")
    logger.info("=" * 80)
    
    # 提取特征列
    all_features = []
    for features in feature_groups.values():
        all_features.extend(features)
    
    X = df[all_features].copy()
    
    # 创建目标变量（未来1周期收益率）
    y = df['Close'].pct_change(1).shift(-1)
    
    # 删除NaN
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    logger.info(f"有效样本数: {len(X_valid)}")
    
    # 创建验证器
    validator = FeatureValidator()
    
    # 1. 单特征信息量测试
    logger.info("\n1. 单特征信息量测试")
    logger.info("-" * 80)
    single_feature_info = validator.test_single_feature_information(X_valid, y_valid, top_n=15)
    
    # 2. 特征相关性检测
    logger.info("\n2. 特征相关性检测")
    logger.info("-" * 80)
    corr_matrix, high_corr_pairs = validator.test_feature_correlation(
        X_valid, 
        threshold=0.85, 
        plot=True
    )
    
    # 3. VIF多重共线性检测
    logger.info("\n3. VIF多重共线性检测")
    logger.info("-" * 80)
    vif_results = validator.test_vif_multicollinearity(X_valid, threshold=10.0)
    
    # 4. 置换重要性测试（需要训练一个简单模型）
    logger.info("\n4. 置换重要性测试")
    logger.info("-" * 80)
    logger.info("训练简单线性模型用于置换测试...")
    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 训练Ridge回归模型
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # 执行置换重要性测试
    permutation_importance = validator.test_permutation_importance(
        model, X_val, y_val, n_repeats=50, random_state=42
    )
    
    return validator.validation_results


def generate_detailed_report(validation_results: dict, output_dir: Path) -> None:
    """
    生成详细的特征检测报告
    
    Args:
        validation_results: 验证结果字典
        output_dir: 输出目录
    """
    logger.info("=" * 80)
    logger.info("生成详细特征检测报告")
    logger.info("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成文本报告
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("MES期货数据特征工程验证报告")
    report_lines.append("=" * 100)
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1.1 单特征信息量测试
    if 'single_feature_info' in validation_results:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("1. 单特征信息量测试")
        report_lines.append("=" * 100)
        report_lines.append("\n说明: 评估每个特征单独对目标变量的预测能力")
        report_lines.append("  - R² Score: 线性回归R²得分，范围[0,1]，越高越好")
        report_lines.append("  - Mutual Info: 互信息得分，衡量非线性关系，越高越好")
        report_lines.append("  - Combined Score: R² + Mutual Info 综合得分")
        report_lines.append("")
        
        df = validation_results['single_feature_info']
        report_lines.append("前15个最重要的特征:")
        report_lines.append("-" * 100)
        report_lines.append(df.head(15).to_string(index=False))
        report_lines.append("")
        
        # 统计信息
        report_lines.append("统计信息:")
        report_lines.append(f"  - 平均R²: {df['r2_score'].mean():.6f}")
        report_lines.append(f"  - 平均互信息: {df['mutual_info'].mean():.6f}")
        report_lines.append(f"  - R²>0.001的特征数: {(df['r2_score'] > 0.001).sum()}/{len(df)}")
        report_lines.append("")
    
    # 1.2 置换重要性测试
    if 'permutation_importance' in validation_results:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("2. 置换重要性测试")
        report_lines.append("=" * 100)
        report_lines.append("\n说明: 通过随机打乱特征值评估特征对模型性能的贡献")
        report_lines.append("  - Importance: 置换后性能下降幅度，越高越重要")
        report_lines.append("  - Std: 重要性的标准差")
        report_lines.append("  - P-value: 显著性检验p值，<0.05表示显著")
        report_lines.append("")
        
        df = validation_results['permutation_importance']
        significant_df = df[df['is_significant']]
        
        report_lines.append(f"显著特征 (p<0.05): {len(significant_df)}/{len(df)}")
        report_lines.append("-" * 100)
        report_lines.append(significant_df.head(15).to_string(index=False))
        report_lines.append("")
        
        # 不显著的特征
        non_significant_df = df[~df['is_significant']]
        if len(non_significant_df) > 0:
            report_lines.append(f"\n不显著特征 (p>=0.05): {len(non_significant_df)}")
            report_lines.append("-" * 100)
            report_lines.append(non_significant_df.to_string(index=False))
            report_lines.append("")
    
    # 1.3 特征相关性检测
    if 'high_corr_pairs' in validation_results:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("3. 特征相关性检测")
        report_lines.append("=" * 100)
        report_lines.append("\n说明: 识别高度相关的特征对，避免冗余")
        report_lines.append("  - 阈值: |ρ| > 0.85")
        report_lines.append("")
        
        pairs = validation_results['high_corr_pairs']
        
        if len(pairs) > 0:
            report_lines.append(f"发现 {len(pairs)} 对高度相关特征:")
            report_lines.append("-" * 100)
            for feat1, feat2, corr in pairs:
                report_lines.append(f"  {feat1:25s} <-> {feat2:25s}  |ρ| = {corr:.4f}")
        else:
            report_lines.append("未发现高度相关的特征对 (|ρ| > 0.85)")
        report_lines.append("")
    
    # 1.4 VIF多重共线性检测
    if 'vif' in validation_results:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("4. VIF多重共线性检测")
        report_lines.append("=" * 100)
        report_lines.append("\n说明: 使用方差膨胀因子检测多重共线性")
        report_lines.append("  - VIF < 5: 无多重共线性")
        report_lines.append("  - 5 <= VIF < 10: 中度多重共线性")
        report_lines.append("  - VIF >= 10: 严重多重共线性")
        report_lines.append("")
        
        df = validation_results['vif']
        high_vif_df = df[df['has_multicollinearity']]
        
        if len(high_vif_df) > 0:
            report_lines.append(f"存在多重共线性的特征 (VIF>=10): {len(high_vif_df)}/{len(df)}")
            report_lines.append("-" * 100)
            report_lines.append(high_vif_df.to_string(index=False))
            report_lines.append("")
        else:
            report_lines.append("未发现严重多重共线性问题 (所有VIF < 10)")
            report_lines.append("")
        
        # 显示所有特征的VIF
        report_lines.append("所有特征的VIF值:")
        report_lines.append("-" * 100)
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
    
    # 保存文本报告
    report_path = output_dir / 'feature_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"文本报告已保存: {report_path}")
    
    # 2. 生成JSON报告（便于程序读取）
    json_report = {}
    
    if 'single_feature_info' in validation_results:
        df = validation_results['single_feature_info']
        json_report['single_feature_info'] = df.to_dict('records')
    
    if 'permutation_importance' in validation_results:
        df = validation_results['permutation_importance']
        json_report['permutation_importance'] = df.to_dict('records')
    
    if 'high_corr_pairs' in validation_results:
        pairs = validation_results['high_corr_pairs']
        json_report['high_corr_pairs'] = [
            {'feature1': f1, 'feature2': f2, 'correlation': float(corr)}
            for f1, f2, corr in pairs
        ]
    
    if 'vif' in validation_results:
        df = validation_results['vif']
        json_report['vif'] = df.to_dict('records')
    
    json_path = output_dir / 'feature_validation_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON报告已保存: {json_path}")


def suggest_core_features(validation_results: dict, output_dir: Path) -> list:
    """
    给出核心特征保留建议
    
    Args:
        validation_results: 验证结果字典
        output_dir: 输出目录
        
    Returns:
        建议保留的核心特征列表
    """
    logger.info("=" * 80)
    logger.info("生成核心特征保留建议")
    logger.info("=" * 80)
    
    # 创建验证器实例并加载结果
    validator = FeatureValidator()
    validator.validation_results = validation_results
    
    # 获取建议移除的特征
    features_to_remove = validator.suggest_feature_removal(
        importance_threshold=0.0001,  # 重要性阈值
        corr_threshold=0.85,          # 相关性阈值
        vif_threshold=10.0            # VIF阈值
    )
    
    # 获取所有特征
    if 'single_feature_info' in validation_results:
        all_features = validation_results['single_feature_info']['feature'].tolist()
    else:
        all_features = []
    
    # 计算保留的特征
    core_features = [f for f in all_features if f not in features_to_remove]
    
    # 生成建议报告
    suggestion_lines = []
    suggestion_lines.append("=" * 100)
    suggestion_lines.append("核心特征保留建议")
    suggestion_lines.append("=" * 100)
    suggestion_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    suggestion_lines.append("")
    
    suggestion_lines.append("建议策略:")
    suggestion_lines.append("  1. 移除重要性过低的特征 (importance < 0.0001)")
    suggestion_lines.append("  2. 移除高度相关特征对中重要性较低的一个 (|ρ| > 0.85)")
    suggestion_lines.append("  3. 移除存在严重多重共线性的特征 (VIF >= 10)")
    suggestion_lines.append("")
    
    suggestion_lines.append(f"原始特征数: {len(all_features)}")
    suggestion_lines.append(f"建议移除: {len(features_to_remove)}")
    suggestion_lines.append(f"建议保留: {len(core_features)}")
    suggestion_lines.append("")
    
    if len(features_to_remove) > 0:
        suggestion_lines.append("建议移除的特征:")
        suggestion_lines.append("-" * 100)
        for feat in sorted(features_to_remove):
            # 查找移除原因
            reasons = []
            
            # 检查重要性
            if 'permutation_importance' in validation_results:
                imp_df = validation_results['permutation_importance']
                imp_row = imp_df[imp_df['feature'] == feat]
                if len(imp_row) > 0:
                    imp_value = imp_row['importance'].values[0]
                    if imp_value < 0.0001:
                        reasons.append(f"低重要性({imp_value:.6f})")
            
            # 检查相关性
            if 'high_corr_pairs' in validation_results:
                for f1, f2, corr in validation_results['high_corr_pairs']:
                    if feat in [f1, f2]:
                        other = f2 if feat == f1 else f1
                        reasons.append(f"与{other}高度相关({corr:.4f})")
                        break
            
            # 检查VIF
            if 'vif' in validation_results:
                vif_df = validation_results['vif']
                vif_row = vif_df[vif_df['feature'] == feat]
                if len(vif_row) > 0:
                    vif_value = vif_row['VIF'].values[0]
                    if vif_value >= 10:
                        reasons.append(f"高VIF({vif_value:.2f})")
            
            reason_str = ", ".join(reasons) if reasons else "未知原因"
            suggestion_lines.append(f"  - {feat:30s}  原因: {reason_str}")
        suggestion_lines.append("")
    
    suggestion_lines.append("建议保留的核心特征:")
    suggestion_lines.append("-" * 100)
    
    # 按重要性排序核心特征
    if 'permutation_importance' in validation_results:
        imp_df = validation_results['permutation_importance']
        core_features_with_imp = []
        for feat in core_features:
            imp_row = imp_df[imp_df['feature'] == feat]
            if len(imp_row) > 0:
                imp_value = imp_row['importance'].values[0]
                is_sig = imp_row['is_significant'].values[0]
                core_features_with_imp.append((feat, imp_value, is_sig))
        
        # 排序
        core_features_with_imp.sort(key=lambda x: x[1], reverse=True)
        
        for feat, imp, is_sig in core_features_with_imp:
            sig_mark = "***" if is_sig else ""
            suggestion_lines.append(f"  - {feat:30s}  重要性: {imp:.6f} {sig_mark}")
    else:
        for feat in sorted(core_features):
            suggestion_lines.append(f"  - {feat}")
    
    suggestion_lines.append("")
    suggestion_lines.append("注: *** 表示统计显著 (p<0.05)")
    suggestion_lines.append("")
    
    # 保存建议报告
    suggestion_path = output_dir / 'core_features_suggestion.txt'
    with open(suggestion_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(suggestion_lines))
    
    logger.info(f"核心特征建议已保存: {suggestion_path}")
    
    # 保存核心特征列表（JSON格式）
    core_features_json = {
        'total_features': len(all_features),
        'features_to_remove': features_to_remove,
        'core_features': core_features,
        'core_features_count': len(core_features)
    }
    
    json_path = output_dir / 'core_features.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(core_features_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"核心特征列表已保存: {json_path}")
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("核心特征保留建议摘要")
    logger.info("=" * 80)
    logger.info(f"原始特征数: {len(all_features)}")
    logger.info(f"建议移除: {len(features_to_remove)}")
    logger.info(f"建议保留: {len(core_features)}")
    logger.info(f"保留比例: {len(core_features)/len(all_features)*100:.1f}%")
    
    return core_features


def main():
    """主函数"""
    logger.info("=" * 100)
    logger.info("MES期货数据特征工程和验证流程")
    logger.info("=" * 100)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # 1. 加载清洗后的数据
    data_path = project_root / 'data' / 'processed' / 'MES_cleaned_5m.csv'
    df = load_cleaned_data(data_path)
    
    # 2. 计算特征
    df_with_features, calculator = calculate_features(df)
    
    # 保存带特征的数据
    features_data_path = project_root / 'data' / 'processed' / 'MES_with_features_5m.csv'
    df_with_features.to_csv(features_data_path)
    logger.info(f"带特征的数据已保存: {features_data_path}")
    
    # 3. 归一化特征
    feature_groups = calculator.get_feature_groups()
    df_normalized, scaler = normalize_features(df_with_features, feature_groups)
    
    # 保存归一化后的数据
    normalized_data_path = project_root / 'data' / 'processed' / 'MES_normalized_5m.csv'
    df_normalized.to_csv(normalized_data_path)
    logger.info(f"归一化后的数据已保存: {normalized_data_path}")
    
    # 4. 执行特征验证
    validation_results = validate_features(df_normalized, feature_groups)
    
    # 5. 生成详细报告
    output_dir = project_root / 'training' / 'output'
    generate_detailed_report(validation_results, output_dir)
    
    # 6. 给出核心特征建议
    core_features = suggest_core_features(validation_results, output_dir)
    
    logger.info("")
    logger.info("=" * 100)
    logger.info("特征工程和验证流程完成")
    logger.info("=" * 100)
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info("生成的文件:")
    logger.info(f"  1. 带特征的数据: {features_data_path}")
    logger.info(f"  2. 归一化后的数据: {normalized_data_path}")
    logger.info(f"  3. 归一化器: {project_root / 'models' / 'scalers'}")
    logger.info(f"  4. 特征验证报告: {output_dir / 'feature_validation_report.txt'}")
    logger.info(f"  5. JSON报告: {output_dir / 'feature_validation_report.json'}")
    logger.info(f"  6. 核心特征建议: {output_dir / 'core_features_suggestion.txt'}")
    logger.info(f"  7. 核心特征列表: {output_dir / 'core_features.json'}")
    logger.info(f"  8. 相关性热力图: feature_correlation_heatmap.png")


if __name__ == '__main__':
    main()