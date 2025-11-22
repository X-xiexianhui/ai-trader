"""
SHAP值分析模块 - 使用SHAP解释模型预测

本模块提供基于SHAP (SHapley Additive exPlanations) 的模型可解释性分析，包括：
1. 全局特征重要性分析
2. 单样本预测解释
3. 特征依赖关系分析
4. 特征交互效应分析
5. 可视化报告生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import shap
import warnings

# 配置日志
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    SHAP值分析器 - 提供模型可解释性分析
    
    功能：
    1. 计算SHAP值
    2. 全局特征重要性分析
    3. 单样本预测解释
    4. 特征依赖关系分析
    5. 特征交互效应分析
    6. 生成可视化报告
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        初始化SHAP分析器
        
        Args:
            model: 训练好的模型
            model_type: 模型类型，支持：
                - 'tree': 树模型（XGBoost, LightGBM, CatBoost, RandomForest等）
                - 'linear': 线性模型（LinearRegression, Ridge, Lasso等）
                - 'deep': 深度学习模型（需要提供predict函数）
                - 'kernel': 通用模型（使用KernelExplainer，较慢）
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self.base_value = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            feature_names: Optional[List[str]] = None,
            background_samples: int = 100) -> 'SHAPAnalyzer':
        """
        拟合SHAP解释器
        
        Args:
            X: 训练数据或背景数据
            feature_names: 特征名称列表
            background_samples: 背景样本数量（用于KernelExplainer）
            
        Returns:
            self
        """
        logger.info(f"初始化SHAP解释器 (模型类型: {self.model_type})...")
        
        # 处理特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # 根据模型类型创建解释器
        if self.model_type == 'tree':
            # 树模型使用TreeExplainer（最快）
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("使用TreeExplainer")
            
        elif self.model_type == 'linear':
            # 线性模型使用LinearExplainer
            self.explainer = shap.LinearExplainer(self.model, X_array)
            logger.info("使用LinearExplainer")
            
        elif self.model_type == 'deep':
            # 深度学习模型使用DeepExplainer
            # 需要背景数据
            background = X_array[:background_samples]
            self.explainer = shap.DeepExplainer(self.model, background)
            logger.info(f"使用DeepExplainer (背景样本: {background_samples})")
            
        elif self.model_type == 'kernel':
            # 通用模型使用KernelExplainer（最慢但最通用）
            background = shap.sample(X_array, background_samples)
            self.explainer = shap.KernelExplainer(self.model.predict, background)
            logger.info(f"使用KernelExplainer (背景样本: {background_samples})")
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        logger.info("SHAP解释器初始化完成")
        
        return self
    
    def calculate_shap_values(self, X: Union[pd.DataFrame, np.ndarray],
                             check_additivity: bool = False) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            X: 待解释的数据
            check_additivity: 是否检查可加性（调试用）
            
        Returns:
            SHAP值数组
        """
        if self.explainer is None:
            raise ValueError("请先调用fit()方法初始化解释器")
        
        logger.info(f"计算SHAP值 (样本数: {len(X)})...")
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 计算SHAP值
        if self.model_type == 'tree':
            shap_values = self.explainer.shap_values(X_array, check_additivity=check_additivity)
        else:
            shap_values = self.explainer.shap_values(X_array)
        
        # 处理多输出情况（分类模型）
        if isinstance(shap_values, list):
            # 对于二分类，通常取第二个类别的SHAP值
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        self.shap_values = shap_values
        
        # 获取基准值
        if hasattr(self.explainer, 'expected_value'):
            self.base_value = self.explainer.expected_value
            if isinstance(self.base_value, (list, np.ndarray)):
                self.base_value = self.base_value[1] if len(self.base_value) == 2 else self.base_value[0]
        
        logger.info(f"SHAP值计算完成 (shape: {shap_values.shape})")
        
        return shap_values
    
    def get_global_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取全局特征重要性
        
        Args:
            top_n: 返回前N个最重要的特征
            
        Returns:
            特征重要性DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info("计算全局特征重要性...")
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'importance_rank': range(1, len(self.feature_names) + 1)
        })
        
        # 排序
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        importance_df['importance_rank'] = range(1, len(importance_df) + 1)
        
        logger.info(f"前{min(top_n, len(importance_df))}个最重要特征:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['importance_rank']}. {row['feature']}: {row['mean_abs_shap']:.6f}")
        
        return importance_df.head(top_n)
    
    def explain_prediction(self, X: Union[pd.DataFrame, np.ndarray],
                          sample_idx: int = 0) -> Dict:
        """
        解释单个样本的预测
        
        Args:
            X: 数据
            sample_idx: 样本索引
            
        Returns:
            解释结果字典
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info(f"解释样本 {sample_idx} 的预测...")
        
        # 获取样本的SHAP值
        sample_shap = self.shap_values[sample_idx]
        
        # 获取样本的特征值
        if isinstance(X, pd.DataFrame):
            sample_features = X.iloc[sample_idx].values
        else:
            sample_features = X[sample_idx]
        
        # 创建解释DataFrame
        explanation_df = pd.DataFrame({
            'feature': self.feature_names,
            'feature_value': sample_features,
            'shap_value': sample_shap,
            'abs_shap_value': np.abs(sample_shap)
        })
        
        # 排序
        explanation_df = explanation_df.sort_values('abs_shap_value', ascending=False)
        
        # 计算预测值
        prediction = self.base_value + sample_shap.sum()
        
        result = {
            'sample_idx': sample_idx,
            'base_value': self.base_value,
            'prediction': prediction,
            'shap_sum': sample_shap.sum(),
            'explanation': explanation_df
        }
        
        logger.info(f"  基准值: {self.base_value:.6f}")
        logger.info(f"  SHAP总和: {sample_shap.sum():.6f}")
        logger.info(f"  预测值: {prediction:.6f}")
        
        return result
    
    def plot_summary(self, X: Union[pd.DataFrame, np.ndarray],
                    plot_type: str = 'dot',
                    max_display: int = 20,
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None) -> None:
        """
        绘制SHAP摘要图
        
        Args:
            X: 数据
            plot_type: 图表类型 ('dot', 'bar', 'violin')
            max_display: 最多显示的特征数
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info(f"绘制SHAP摘要图 (类型: {plot_type})...")
        
        plt.figure(figsize=figsize)
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 绘制摘要图
        shap.summary_plot(
            self.shap_values,
            X_array,
            feature_names=self.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"摘要图已保存: {save_path}")
        
        plt.close()
    
    def plot_waterfall(self, X: Union[pd.DataFrame, np.ndarray],
                      sample_idx: int = 0,
                      max_display: int = 20,
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None) -> None:
        """
        绘制单样本的瀑布图
        
        Args:
            X: 数据
            sample_idx: 样本索引
            max_display: 最多显示的特征数
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info(f"绘制样本 {sample_idx} 的瀑布图...")
        
        plt.figure(figsize=figsize)
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 创建Explanation对象
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.base_value,
            data=X_array[sample_idx],
            feature_names=self.feature_names
        )
        
        # 绘制瀑布图
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"瀑布图已保存: {save_path}")
        
        plt.close()
    
    def plot_dependence(self, X: Union[pd.DataFrame, np.ndarray],
                       feature: str,
                       interaction_feature: Optional[str] = 'auto',
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None) -> None:
        """
        绘制特征依赖图
        
        Args:
            X: 数据
            feature: 目标特征
            interaction_feature: 交互特征 ('auto'自动选择)
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info(f"绘制特征 {feature} 的依赖图...")
        
        plt.figure(figsize=figsize)
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # 绘制依赖图
        shap.dependence_plot(
            feature,
            self.shap_values,
            X_array,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"依赖图已保存: {save_path}")
        
        plt.close()
    
    def plot_force(self, X: Union[pd.DataFrame, np.ndarray],
                  sample_idx: int = 0,
                  matplotlib: bool = True,
                  figsize: Tuple[int, int] = (20, 3),
                  save_path: Optional[str] = None) -> None:
        """
        绘制单样本的力图
        
        Args:
            X: 数据
            sample_idx: 样本索引
            matplotlib: 是否使用matplotlib（否则使用JavaScript）
            figsize: 图形大小
            save_path: 保存路径
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info(f"绘制样本 {sample_idx} 的力图...")
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if matplotlib:
            plt.figure(figsize=figsize)
            
            # 使用matplotlib绘制
            shap.force_plot(
                self.base_value,
                self.shap_values[sample_idx],
                X_array[sample_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"力图已保存: {save_path}")
            
            plt.close()
        else:
            # 使用JavaScript绘制（交互式）
            force_plot = shap.force_plot(
                self.base_value,
                self.shap_values[sample_idx],
                X_array[sample_idx],
                feature_names=self.feature_names
            )
            
            if save_path:
                shap.save_html(save_path, force_plot)
                logger.info(f"交互式力图已保存: {save_path}")
    
    def analyze_feature_interactions(self, X: Union[pd.DataFrame, np.ndarray],
                                    top_n: int = 10) -> pd.DataFrame:
        """
        分析特征交互效应
        
        Args:
            X: 数据
            top_n: 返回前N个最强的交互
            
        Returns:
            特征交互DataFrame
        """
        if self.shap_values is None:
            raise ValueError("请先调用calculate_shap_values()方法")
        
        logger.info("分析特征交互效应...")
        
        # 转换为数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        interactions = []
        n_features = len(self.feature_names)
        
        # 计算所有特征对的交互强度
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # 使用SHAP值的相关性作为交互强度的代理
                interaction_strength = np.abs(np.corrcoef(
                    self.shap_values[:, i],
                    self.shap_values[:, j]
                )[0, 1])
                
                interactions.append({
                    'feature1': self.feature_names[i],
                    'feature2': self.feature_names[j],
                    'interaction_strength': interaction_strength
                })
        
        # 转换为DataFrame并排序
        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('interaction_strength', ascending=False)
        
        logger.info(f"前{top_n}个最强的特征交互:")
        for idx, row in interactions_df.head(top_n).iterrows():
            logger.info(f"  {row['feature1']} <-> {row['feature2']}: {row['interaction_strength']:.4f}")
        
        return interactions_df.head(top_n)
    
    def generate_report(self, X: Union[pd.DataFrame, np.ndarray],
                       output_dir: Union[str, Path],
                       report_name: str = 'shap_analysis_report',
                       n_samples_to_explain: int = 5) -> None:
        """
        生成完整的SHAP分析报告
        
        Args:
            X: 数据
            output_dir: 输出目录
            report_name: 报告名称
            n_samples_to_explain: 解释的样本数量
        """
        logger.info("=" * 80)
        logger.info("生成SHAP分析报告")
        logger.info("=" * 80)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 全局特征重要性
        importance_df = self.get_global_importance(top_n=20)
        importance_df.to_csv(output_dir / f'{report_name}_importance.csv', index=False)
        
        # 2. 绘制摘要图
        self.plot_summary(
            X,
            plot_type='dot',
            save_path=output_dir / f'{report_name}_summary_dot.png'
        )
        
        self.plot_summary(
            X,
            plot_type='bar',
            save_path=output_dir / f'{report_name}_summary_bar.png'
        )
        
        # 3. 解释样本
        sample_indices = np.linspace(0, len(X) - 1, n_samples_to_explain, dtype=int)
        
        for idx in sample_indices:
            # 瀑布图
            self.plot_waterfall(
                X,
                sample_idx=idx,
                save_path=output_dir / f'{report_name}_waterfall_sample_{idx}.png'
            )
            
            # 力图
            self.plot_force(
                X,
                sample_idx=idx,
                save_path=output_dir / f'{report_name}_force_sample_{idx}.png'
            )
        
        # 4. 特征依赖图（前5个最重要特征）
        top_features = importance_df.head(5)['feature'].tolist()
        
        for feature in top_features:
            self.plot_dependence(
                X,
                feature=feature,
                save_path=output_dir / f'{report_name}_dependence_{feature}.png'
            )
        
        # 5. 特征交互分析
        interactions_df = self.analyze_feature_interactions(X, top_n=20)
        interactions_df.to_csv(output_dir / f'{report_name}_interactions.csv', index=False)
        
        # 6. 生成文本报告
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("SHAP值分析报告")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        report_lines.append("1. 全局特征重要性 (Top 20)")
        report_lines.append("-" * 100)
        report_lines.append(importance_df.to_string(index=False))
        report_lines.append("")
        
        report_lines.append("2. 特征交互效应 (Top 20)")
        report_lines.append("-" * 100)
        report_lines.append(interactions_df.to_string(index=False))
        report_lines.append("")
        
        report_lines.append("3. 样本解释")
        report_lines.append("-" * 100)
        for idx in sample_indices:
            explanation = self.explain_prediction(X, sample_idx=idx)
            report_lines.append(f"\n样本 {idx}:")
            report_lines.append(f"  基准值: {explanation['base_value']:.6f}")
            report_lines.append(f"  预测值: {explanation['prediction']:.6f}")
            report_lines.append(f"  前5个最重要特征:")
            for _, row in explanation['explanation'].head(5).iterrows():
                report_lines.append(
                    f"    {row['feature']:25s} = {row['feature_value']:10.4f}  "
                    f"SHAP = {row['shap_value']:10.6f}"
                )
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        # 保存文本报告
        report_path = output_dir / f'{report_name}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"SHAP分析报告已保存到: {output_dir}")
        logger.info(f"  - 特征重要性: {report_name}_importance.csv")
        logger.info(f"  - 摘要图: {report_name}_summary_*.png")
        logger.info(f"  - 样本解释: {report_name}_waterfall_*.png, {report_name}_force_*.png")
        logger.info(f"  - 依赖图: {report_name}_dependence_*.png")
        logger.info(f"  - 特征交互: {report_name}_interactions.csv")
        logger.info(f"  - 文本报告: {report_name}.txt")