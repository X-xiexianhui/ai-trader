"""
消融实验框架
评估特征组对模型性能的贡献

任务6.2.1-6.2.2实现:
1. 消融实验框架
2. 特征组贡献度分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    任务6.2.1: 消融实验框架
    
    通过逐组移除特征评估其贡献度
    """
    
    def __init__(
        self,
        feature_groups: Dict[str, List[str]],
        output_dir: str = 'results/ablation'
    ):
        """
        初始化消融实验框架
        
        Args:
            feature_groups: 特征组定义，格式: {组名: [特征列表]}
            output_dir: 输出目录
        """
        self.feature_groups = feature_groups
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[Dict] = []
        self.baseline_metrics: Dict = {}
        
        logger.info(f"消融实验框架初始化，特征组数: {len(feature_groups)}")
        for group_name, features in feature_groups.items():
            logger.info(f"  {group_name}: {len(features)} 个特征")
    
    def run_ablation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        train_func: Callable,
        eval_func: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        运行消融实验
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征
            y_val: 验证集标签
            train_func: 训练函数，接收(X_train, y_train, **kwargs)返回模型
            eval_func: 评估函数，接收(model, X_val, y_val, **kwargs)返回指标字典
            **kwargs: 传递给训练和评估函数的额外参数
            
        Returns:
            实验结果列表
        """
        logger.info("开始消融实验...")
        
        self.results = []
        
        # 1. 全量基线
        logger.info("\n" + "="*60)
        logger.info("运行全量基线（使用所有特征）")
        logger.info("="*60)
        
        try:
            baseline_model = train_func(X_train, y_train, **kwargs)
            baseline_metrics = eval_func(baseline_model, X_val, y_val, **kwargs)
            self.baseline_metrics = baseline_metrics
            
            self.results.append({
                'experiment': 'baseline',
                'removed_group': None,
                'features_used': list(X_train.columns),
                'n_features': len(X_train.columns),
                'metrics': baseline_metrics
            })
            
            logger.info(f"基线指标: {baseline_metrics}")
            
        except Exception as e:
            logger.error(f"基线实验失败: {e}")
            raise
        
        # 2. 逐组移除特征
        for group_name, group_features in self.feature_groups.items():
            logger.info("\n" + "="*60)
            logger.info(f"移除特征组: {group_name}")
            logger.info("="*60)
            
            # 检查特征是否存在
            available_features = [f for f in group_features if f in X_train.columns]
            if not available_features:
                logger.warning(f"特征组 {group_name} 中没有可用特征，跳过")
                continue
            
            logger.info(f"移除 {len(available_features)} 个特征")
            
            # 移除该组特征
            remaining_features = [f for f in X_train.columns if f not in available_features]
            
            if not remaining_features:
                logger.warning(f"移除 {group_name} 后没有剩余特征，跳过")
                continue
            
            X_train_ablated = X_train[remaining_features]
            X_val_ablated = X_val[remaining_features]
            
            # 训练和评估
            try:
                model = train_func(X_train_ablated, y_train, **kwargs)
                metrics = eval_func(model, X_val_ablated, y_val, **kwargs)
                
                self.results.append({
                    'experiment': f'remove_{group_name}',
                    'removed_group': group_name,
                    'removed_features': available_features,
                    'features_used': remaining_features,
                    'n_features': len(remaining_features),
                    'metrics': metrics
                })
                
                logger.info(f"移除后指标: {metrics}")
                
            except Exception as e:
                logger.error(f"移除 {group_name} 的实验失败: {e}")
                continue
        
        logger.info(f"\n消融实验完成，共 {len(self.results)} 个实验")
        
        return self.results
    
    def calculate_contributions(self) -> pd.DataFrame:
        """
        计算各特征组的贡献度
        
        Returns:
            贡献度DataFrame
        """
        if not self.results or not self.baseline_metrics:
            logger.warning("没有实验结果")
            return pd.DataFrame()
        
        contributions = []
        
        for result in self.results:
            if result['removed_group'] is None:
                continue  # 跳过基线
            
            group_name = result['removed_group']
            metrics = result['metrics']
            
            # 计算性能下降（贡献度）
            contribution = {
                'feature_group': group_name,
                'n_features': len(result['removed_features'])
            }
            
            for metric_name, baseline_value in self.baseline_metrics.items():
                if metric_name in metrics:
                    ablated_value = metrics[metric_name]
                    
                    # 性能下降 = 基线 - 移除后
                    # 正值表示该组特征有正贡献
                    performance_drop = baseline_value - ablated_value
                    
                    # 相对贡献度（百分比）
                    if baseline_value != 0:
                        relative_contribution = (performance_drop / abs(baseline_value)) * 100
                    else:
                        relative_contribution = 0
                    
                    contribution[f'{metric_name}_drop'] = performance_drop
                    contribution[f'{metric_name}_relative'] = relative_contribution
            
            contributions.append(contribution)
        
        contributions_df = pd.DataFrame(contributions)
        
        logger.info(f"计算了 {len(contributions_df)} 个特征组的贡献度")
        
        return contributions_df
    
    def get_results(self) -> List[Dict]:
        """获取实验结果"""
        return self.results
    
    def save_results(self, filename: str = 'ablation_results.json') -> str:
        """
        保存实验结果
        
        Args:
            filename: 文件名
            
        Returns:
            文件路径
        """
        if not self.results:
            logger.warning("没有结果可保存")
            return ""
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存: {filepath}")
        
        return str(filepath)


class FeatureGroupAnalyzer:
    """
    任务6.2.2: 特征组贡献度分析
    
    分析各特征组对模型性能的贡献
    """
    
    def __init__(self, ablation_results: List[Dict]):
        """
        初始化特征组分析器
        
        Args:
            ablation_results: 消融实验结果
        """
        self.results = ablation_results
        self.baseline_metrics = {}
        self.contributions_df = None
        
        # 提取基线指标
        for result in ablation_results:
            if result.get('removed_group') is None:
                self.baseline_metrics = result.get('metrics', {})
                break
        
        logger.info(f"特征组分析器初始化，共 {len(ablation_results)} 个实验")
    
    def analyze_absolute_contribution(self) -> pd.DataFrame:
        """
        分析绝对贡献度
        
        Returns:
            绝对贡献度DataFrame
        """
        contributions = []
        
        for result in self.results:
            if result.get('removed_group') is None:
                continue
            
            group_name = result['removed_group']
            metrics = result['metrics']
            
            contribution = {
                'feature_group': group_name,
                'n_features': len(result.get('removed_features', []))
            }
            
            # 计算每个指标的绝对贡献
            for metric_name, baseline_value in self.baseline_metrics.items():
                if metric_name in metrics:
                    ablated_value = metrics[metric_name]
                    absolute_contribution = baseline_value - ablated_value
                    contribution[metric_name] = absolute_contribution
            
            contributions.append(contribution)
        
        contributions_df = pd.DataFrame(contributions)
        
        logger.info("绝对贡献度分析完成")
        
        return contributions_df
    
    def analyze_relative_contribution(self) -> pd.DataFrame:
        """
        分析相对贡献度（百分比）
        
        Returns:
            相对贡献度DataFrame
        """
        contributions = []
        
        for result in self.results:
            if result.get('removed_group') is None:
                continue
            
            group_name = result['removed_group']
            metrics = result['metrics']
            
            contribution = {
                'feature_group': group_name,
                'n_features': len(result.get('removed_features', []))
            }
            
            # 计算每个指标的相对贡献
            for metric_name, baseline_value in self.baseline_metrics.items():
                if metric_name in metrics:
                    ablated_value = metrics[metric_name]
                    
                    if baseline_value != 0:
                        relative_contribution = ((baseline_value - ablated_value) / 
                                                abs(baseline_value)) * 100
                    else:
                        relative_contribution = 0
                    
                    contribution[f'{metric_name}_pct'] = relative_contribution
            
            contributions.append(contribution)
        
        contributions_df = pd.DataFrame(contributions)
        
        logger.info("相对贡献度分析完成")
        
        return contributions_df
    
    def rank_feature_groups(
        self,
        metric: str = 'sharpe_ratio',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        对特征组进行排名
        
        Args:
            metric: 用于排名的指标
            ascending: 是否升序
            
        Returns:
            排名DataFrame
        """
        if self.contributions_df is None:
            self.contributions_df = self.analyze_absolute_contribution()
        
        if metric not in self.contributions_df.columns:
            logger.warning(f"指标 {metric} 不存在")
            return pd.DataFrame()
        
        ranked_df = self.contributions_df.sort_values(
            metric, 
            ascending=ascending
        ).reset_index(drop=True)
        
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        logger.info(f"特征组排名完成（基于 {metric}）")
        
        return ranked_df
    
    def identify_redundant_features(
        self,
        threshold: float = 0.01
    ) -> List[str]:
        """
        识别冗余特征组
        
        Args:
            threshold: 贡献度阈值，低于此值认为冗余
            
        Returns:
            冗余特征组列表
        """
        if self.contributions_df is None:
            self.contributions_df = self.analyze_absolute_contribution()
        
        redundant_groups = []
        
        # 检查所有指标的贡献度
        metric_columns = [col for col in self.contributions_df.columns 
                         if col not in ['feature_group', 'n_features']]
        
        for _, row in self.contributions_df.iterrows():
            group_name = row['feature_group']
            
            # 如果所有指标的贡献度都很小，认为是冗余的
            all_small = all(abs(row[col]) < threshold for col in metric_columns)
            
            if all_small:
                redundant_groups.append(group_name)
        
        logger.info(f"识别出 {len(redundant_groups)} 个冗余特征组")
        
        return redundant_groups
    
    def plot_contribution_heatmap(
        self,
        filename: str = 'contribution_heatmap.png',
        output_dir: str = 'results/ablation'
    ) -> str:
        """
        绘制贡献度热力图
        
        Args:
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        if self.contributions_df is None:
            self.contributions_df = self.analyze_absolute_contribution()
        
        # 准备数据
        metric_columns = [col for col in self.contributions_df.columns 
                         if col not in ['feature_group', 'n_features']]
        
        if not metric_columns:
            logger.warning("没有指标数据可绘制")
            return ""
        
        plot_data = self.contributions_df.set_index('feature_group')[metric_columns]
        
        # 绘制热力图
        plt.figure(figsize=(12, max(6, len(plot_data) * 0.5)))
        sns.heatmap(
            plot_data,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': '贡献度'},
            linewidths=0.5
        )
        
        plt.title('特征组贡献度热力图\n(正值=有贡献，负值=有害)')
        plt.xlabel('指标')
        plt.ylabel('特征组')
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"贡献度热力图已保存: {filepath}")
        
        return str(filepath)
    
    def plot_contribution_bars(
        self,
        metric: str = 'sharpe_ratio',
        filename: str = 'contribution_bars.png',
        output_dir: str = 'results/ablation'
    ) -> str:
        """
        绘制贡献度柱状图
        
        Args:
            metric: 要绘制的指标
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            图片路径
        """
        if self.contributions_df is None:
            self.contributions_df = self.analyze_absolute_contribution()
        
        if metric not in self.contributions_df.columns:
            logger.warning(f"指标 {metric} 不存在")
            return ""
        
        # 排序
        plot_data = self.contributions_df.sort_values(metric, ascending=True)
        
        # 绘制柱状图
        plt.figure(figsize=(10, max(6, len(plot_data) * 0.4)))
        
        colors = ['green' if x > 0 else 'red' for x in plot_data[metric]]
        
        plt.barh(plot_data['feature_group'], plot_data[metric], color=colors, alpha=0.7)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        plt.xlabel(f'{metric} 贡献度')
        plt.ylabel('特征组')
        plt.title(f'特征组对 {metric} 的贡献度\n(正值=有贡献，负值=有害)')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"贡献度柱状图已保存: {filepath}")
        
        return str(filepath)
    
    def generate_report(
        self,
        filename: str = 'feature_contribution_report.txt',
        output_dir: str = 'results/ablation'
    ) -> str:
        """
        生成特征贡献度分析报告
        
        Args:
            filename: 文件名
            output_dir: 输出目录
            
        Returns:
            报告文件路径
        """
        if self.contributions_df is None:
            self.contributions_df = self.analyze_absolute_contribution()
        
        relative_df = self.analyze_relative_contribution()
        redundant_groups = self.identify_redundant_features()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("特征组贡献度分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 基线指标
        report_lines.append("基线指标（使用所有特征）:")
        report_lines.append("-" * 80)
        for metric, value in self.baseline_metrics.items():
            report_lines.append(f"  {metric}: {value:.4f}")
        report_lines.append("")
        
        # 绝对贡献度
        report_lines.append("绝对贡献度:")
        report_lines.append("-" * 80)
        report_lines.append(self.contributions_df.to_string())
        report_lines.append("")
        
        # 相对贡献度
        report_lines.append("相对贡献度（%）:")
        report_lines.append("-" * 80)
        report_lines.append(relative_df.to_string())
        report_lines.append("")
        
        # 冗余特征组
        report_lines.append("冗余特征组:")
        report_lines.append("-" * 80)
        if redundant_groups:
            for group in redundant_groups:
                report_lines.append(f"  - {group}")
        else:
            report_lines.append("  未发现冗余特征组")
        report_lines.append("")
        
        # 优化建议
        report_lines.append("优化建议:")
        report_lines.append("-" * 80)
        if redundant_groups:
            report_lines.append(f"  1. 考虑移除以下冗余特征组: {', '.join(redundant_groups)}")
        
        # 找出贡献最大的特征组
        metric_columns = [col for col in self.contributions_df.columns 
                         if col not in ['feature_group', 'n_features']]
        if metric_columns:
            top_group = self.contributions_df.loc[
                self.contributions_df[metric_columns[0]].idxmax(), 
                'feature_group'
            ]
            report_lines.append(f"  2. 重点关注贡献最大的特征组: {top_group}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # 保存报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"特征贡献度分析报告已保存: {filepath}")
        
        return str(filepath)