"""
核心特征选择和输出脚本

功能：
1. 加载特征验证结果
2. 基于验证结果选择核心特征
3. 将核心特征列表输出到JSON文件
4. 生成特征选择报告

选择策略：
- 基于置换重要性选择显著特征
- 移除高度相关的冗余特征
- 移除高VIF的多重共线性特征
- 保留信息量最高的特征

使用方法：
    python training/04_core_feature_selection.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Set

from src.utils.logger import setup_logger


def load_validation_results(path: str = "training/output/03_validation_results.pkl") -> dict:
    """加载特征验证结果"""
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


def select_core_features(
    validation_results: dict,
    importance_threshold: float = 0.0001,
    corr_threshold: float = 0.85,
    vif_threshold: float = 10.0,
    min_features: int = 10,
    max_features: int = 20
) -> Dict[str, List[str]]:
    """
    基于验证结果选择核心特征
    
    Args:
        validation_results: 验证结果字典
        importance_threshold: 重要性阈值
        corr_threshold: 相关性阈值
        vif_threshold: VIF阈值
        min_features: 最小特征数
        max_features: 最大特征数
        
    Returns:
        每个品种的核心特征列表
    """
    all_feature_names = validation_results['feature_names']
    symbol_results = validation_results['validation_results']
    
    core_features_dict = {}
    
    for symbol, results in symbol_results.items():
        logger.info(f"\n选择 {symbol} 的核心特征...")
        
        # 初始化特征集合
        candidate_features = set(all_feature_names)
        
        # 1. 基于置换重要性筛选
        if results['perm_results'] is not None:
            perm_df = results['perm_results']
            
            # 保留显著且重要性大于阈值的特征
            significant_features = perm_df[
                (perm_df['is_significant']) & 
                (perm_df['importance'] > importance_threshold)
            ]['feature'].tolist()
            
            candidate_features = candidate_features.intersection(set(significant_features))
            logger.info(f"  置换重要性筛选后: {len(candidate_features)} 个特征")
        
        # 2. 移除高相关特征对中的一个
        if results['high_corr_pairs']:
            features_to_remove = set()
            
            for feat1, feat2, corr in results['high_corr_pairs']:
                if corr > corr_threshold and feat1 in candidate_features and feat2 in candidate_features:
                    # 比较重要性，移除较低的
                    if results['perm_results'] is not None:
                        perm_df = results['perm_results']
                        imp1 = perm_df[perm_df['feature'] == feat1]['importance'].values
                        imp2 = perm_df[perm_df['feature'] == feat2]['importance'].values
                        
                        if len(imp1) > 0 and len(imp2) > 0:
                            if imp1[0] < imp2[0]:
                                features_to_remove.add(feat1)
                            else:
                                features_to_remove.add(feat2)
                        else:
                            features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat2)
            
            candidate_features = candidate_features - features_to_remove
            logger.info(f"  移除高相关特征后: {len(candidate_features)} 个特征")
        
        # 3. 移除高VIF特征
        if results['vif_results'] is not None:
            vif_df = results['vif_results']
            high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['feature'].tolist()
            
            # 按VIF从高到低排序，逐个移除直到满足条件
            high_vif_features_sorted = vif_df[
                vif_df['VIF'] > vif_threshold
            ].sort_values('VIF', ascending=False)['feature'].tolist()
            
            for feat in high_vif_features_sorted:
                if feat in candidate_features and len(candidate_features) > min_features:
                    candidate_features.discard(feat)
            
            logger.info(f"  移除高VIF特征后: {len(candidate_features)} 个特征")
        
        # 4. 如果特征太少，补充信息量最高的特征
        if len(candidate_features) < min_features and results['info_results'] is not None:
            info_df = results['info_results']
            top_features = info_df.head(max_features)['feature'].tolist()
            
            for feat in top_features:
                if len(candidate_features) >= min_features:
                    break
                if feat in all_feature_names:
                    candidate_features.add(feat)
            
            logger.info(f"  补充信息量高的特征后: {len(candidate_features)} 个特征")
        
        # 5. 如果特征太多，保留重要性最高的
        if len(candidate_features) > max_features and results['perm_results'] is not None:
            perm_df = results['perm_results']
            
            # 获取候选特征的重要性
            candidate_importance = perm_df[
                perm_df['feature'].isin(candidate_features)
            ].sort_values('importance', ascending=False)
            
            candidate_features = set(candidate_importance.head(max_features)['feature'].tolist())
            logger.info(f"  限制最大特征数后: {len(candidate_features)} 个特征")
        
        # 转换为列表并排序
        core_features = sorted(list(candidate_features))
        core_features_dict[symbol] = core_features
        
        logger.info(f"  最终核心特征: {len(core_features)} 个")
        logger.info(f"  特征列表: {core_features}")
    
    return core_features_dict


def save_core_features_to_json(
    core_features_dict: Dict[str, List[str]],
    output_path: str = "training/output/core_features.json"
) -> None:
    """
    将核心特征保存到JSON文件
    
    Args:
        core_features_dict: 核心特征字典
        output_path: 输出路径
    """
    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 准备JSON数据
    json_data = {
        'metadata': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '基于特征验证结果选择的核心特征',
            'total_symbols': len(core_features_dict)
        },
        'core_features': core_features_dict
    }
    
    # 保存为JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"核心特征已保存到: {output_path}")


def generate_selection_report(
    validation_results: dict,
    core_features_dict: Dict[str, List[str]],
    output_path: str = "training/output/04_feature_selection_report.txt"
) -> None:
    """
    生成特征选择报告
    
    Args:
        validation_results: 验证结果
        core_features_dict: 核心特征字典
        output_path: 输出路径
    """
    all_feature_names = validation_results['feature_names']
    feature_groups = validation_results['feature_groups']
    symbol_results = validation_results['validation_results']
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("核心特征选择报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("选择策略:")
    report_lines.append("-" * 80)
    report_lines.append("1. 基于置换重要性选择显著特征 (p<0.05)")
    report_lines.append("2. 移除高度相关的冗余特征 (|ρ|>0.85)")
    report_lines.append("3. 移除高VIF的多重共线性特征 (VIF>10)")
    report_lines.append("4. 保留信息量最高的特征")
    report_lines.append("5. 控制特征数量在10-20个之间")
    report_lines.append("")
    
    # 统计信息
    report_lines.append("选择结果统计:")
    report_lines.append("-" * 80)
    report_lines.append(f"原始特征数量: {len(all_feature_names)}")
    
    for symbol, core_features in core_features_dict.items():
        report_lines.append(f"{symbol}: {len(core_features)} 个核心特征")
    
    report_lines.append("")
    
    # 各品种的详细信息
    for symbol, core_features in core_features_dict.items():
        report_lines.append(f"\n品种: {symbol}")
        report_lines.append("-" * 80)
        
        report_lines.append(f"\n核心特征列表 ({len(core_features)} 个):")
        for i, feat in enumerate(core_features, 1):
            # 找到特征所属的组
            feat_group = None
            for group_name, group_features in feature_groups.items():
                if feat in group_features:
                    feat_group = group_name
                    break
            
            # 获取特征的重要性信息
            results = symbol_results[symbol]
            importance_info = ""
            
            if results['perm_results'] is not None:
                perm_df = results['perm_results']
                feat_row = perm_df[perm_df['feature'] == feat]
                if not feat_row.empty:
                    imp = feat_row['importance'].values[0]
                    p_val = feat_row['p_value'].values[0]
                    importance_info = f"重要性={imp:.6f}, p={p_val:.4f}"
            
            report_lines.append(f"  {i:2d}. {feat:25s} [{feat_group:15s}] {importance_info}")
        
        # 被移除的特征
        removed_features = set(all_feature_names) - set(core_features)
        report_lines.append(f"\n被移除的特征 ({len(removed_features)} 个):")
        
        for feat in sorted(removed_features):
            # 找到移除原因
            reasons = []
            
            results = symbol_results[symbol]
            
            # 检查是否因为重要性低
            if results['perm_results'] is not None:
                perm_df = results['perm_results']
                feat_row = perm_df[perm_df['feature'] == feat]
                if not feat_row.empty:
                    if not feat_row['is_significant'].values[0]:
                        reasons.append("不显著")
                    elif feat_row['importance'].values[0] <= 0.0001:
                        reasons.append("重要性低")
            
            # 检查是否因为高相关
            if results['high_corr_pairs']:
                for feat1, feat2, corr in results['high_corr_pairs']:
                    if feat == feat1 or feat == feat2:
                        reasons.append(f"高相关(ρ={corr:.2f})")
                        break
            
            # 检查是否因为高VIF
            if results['vif_results'] is not None:
                vif_df = results['vif_results']
                feat_row = vif_df[vif_df['feature'] == feat]
                if not feat_row.empty and feat_row['has_multicollinearity'].values[0]:
                    vif_val = feat_row['VIF'].values[0]
                    reasons.append(f"高VIF({vif_val:.1f})")
            
            reason_str = ", ".join(reasons) if reasons else "其他"
            report_lines.append(f"  - {feat:25s} [{reason_str}]")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("特征选择完成")
    report_lines.append("核心特征已保存到JSON文件，可用于模型训练")
    report_lines.append("=" * 80)
    
    # 保存报告
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"特征选择报告已保存: {output_path}")


def main():
    """主函数"""
    global logger
    
    # 设置日志
    logger = setup_logger(
        name="core_feature_selection",
        log_file="04_core_feature_selection.log",
        log_level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("开始核心特征选择流程")
    logger.info("=" * 80)
    
    # 1. 加载特征验证结果
    logger.info("\n步骤1: 加载特征验证结果")
    
    validation_results_path = "training/output/03_validation_results.pkl"
    
    if not Path(validation_results_path).exists():
        logger.error(f"验证结果文件不存在: {validation_results_path}")
        logger.error("请先运行 03_feature_validation.py")
        return
    
    validation_results = load_validation_results(validation_results_path)
    
    all_feature_names = validation_results['feature_names']
    logger.info(f"原始特征数量: {len(all_feature_names)}")
    logger.info(f"品种数量: {len(validation_results['validation_results'])}")
    
    # 2. 选择核心特征
    logger.info("\n步骤2: 选择核心特征")
    
    core_features_dict = select_core_features(
        validation_results=validation_results,
        importance_threshold=0.0001,
        corr_threshold=0.85,
        vif_threshold=10.0,
        min_features=10,
        max_features=20
    )
    
    # 3. 保存核心特征到JSON
    logger.info("\n步骤3: 保存核心特征到JSON文件")
    
    json_output_path = "training/output/core_features.json"
    save_core_features_to_json(core_features_dict, json_output_path)
    
    # 4. 生成特征选择报告
    logger.info("\n步骤4: 生成特征选择报告")
    
    report_output_path = "training/output/04_feature_selection_report.txt"
    generate_selection_report(
        validation_results=validation_results,
        core_features_dict=core_features_dict,
        output_path=report_output_path
    )
    
    # 5. 输出摘要
    logger.info("\n" + "=" * 80)
    logger.info("核心特征选择摘要")
    logger.info("=" * 80)
    
    for symbol, core_features in core_features_dict.items():
        logger.info(f"{symbol}: {len(core_features)}/{len(all_feature_names)} 个特征")
        logger.info(f"  核心特征: {core_features}")
    
    logger.info("\n" + "=" * 80)
    logger.info("核心特征选择流程完成！")
    logger.info(f"核心特征已保存到: {json_output_path}")
    logger.info("这些特征将用于模型训练")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()