"""
特征验证模块 - 评估特征质量和重要性

任务1.4.1-1.4.4实现:
1. 单特征信息量测试
2. 置换重要性测试
3. 特征相关性检测
4. VIF多重共线性检测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    特征验证器 - 评估特征质量和重要性
    """
    
    def __init__(self):
        self.validation_results = {}
        
    def test_single_feature_information(self, 
                                       X: pd.DataFrame, 
                                       y: pd.Series,
                                       top_n: int = 10) -> pd.DataFrame:
        """
        任务1.4.1: 单特征信息量测试
        
        使用线性回归R²和互信息评估每个特征的信息量
        
        Args:
            X: 特征DataFrame
            y: 目标变量（未来收益）
            top_n: 返回前N个最重要的特征
            
        Returns:
            特征信息量排名表
        """
        logger.info("开始单特征信息量测试...")
        
        results = []
        
        for col in X.columns:
            # 跳过包含NaN的特征
            if X[col].isna().any() or y.isna().any():
                valid_idx = ~(X[col].isna() | y.isna())
                X_col = X[col][valid_idx].values.reshape(-1, 1)
                y_valid = y[valid_idx].values
            else:
                X_col = X[col].values.reshape(-1, 1)
                y_valid = y.values
            
            if len(X_col) < 10:  # 数据太少，跳过
                continue
            
            # 1. 线性回归R²
            try:
                lr = LinearRegression()
                lr.fit(X_col, y_valid)
                y_pred = lr.predict(X_col)
                r2 = r2_score(y_valid, y_pred)
            except:
                r2 = 0.0
            
            # 2. 互信息
            try:
                mi = mutual_info_regression(X_col, y_valid, random_state=42)[0]
            except:
                mi = 0.0
            
            results.append({
                'feature': col,
                'r2_score': r2,
                'mutual_info': mi,
                'combined_score': r2 + mi  # 综合得分
            })
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('combined_score', ascending=False)
        
        self.validation_results['single_feature_info'] = results_df
        
        logger.info(f"单特征信息量测试完成，前{top_n}个特征:")
        for idx, row in results_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: R²={row['r2_score']:.4f}, MI={row['mutual_info']:.4f}")
        
        return results_df.head(top_n)
    
    def test_permutation_importance(self,
                                   model,
                                   X_val: pd.DataFrame,
                                   y_val: pd.Series,
                                   n_repeats: int = 100,
                                   random_state: int = 42) -> pd.DataFrame:
        """
        任务1.4.2: 置换重要性测试
        
        通过置换特征评估其对模型性能的贡献
        
        Args:
            model: 训练好的模型
            X_val: 验证集特征
            y_val: 验证集目标
            n_repeats: 置换重复次数
            random_state: 随机种子
            
        Returns:
            特征重要性排名 + 显著性检验结果
        """
        logger.info("开始置换重要性测试...")
        
        # 1. 计算基线性能
        try:
            baseline_score = r2_score(y_val, model.predict(X_val))
        except:
            baseline_score = 0.0
        
        results = []
        np.random.seed(random_state)
        
        for col in X_val.columns:
            importance_scores = []
            
            for _ in range(n_repeats):
                # 置换该特征
                X_permuted = X_val.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col].values)
                
                # 计算性能下降
                try:
                    permuted_score = r2_score(y_val, model.predict(X_permuted))
                    importance = baseline_score - permuted_score
                    importance_scores.append(importance)
                except:
                    importance_scores.append(0.0)
            
            # 计算统计量
            mean_importance = np.mean(importance_scores)
            std_importance = np.std(importance_scores)
            
            # 单侧t检验（检验重要性是否显著大于0）
            if std_importance > 0:
                t_stat = mean_importance / (std_importance / np.sqrt(n_repeats))
                p_value = 1 - stats.t.cdf(t_stat, n_repeats - 1)
            else:
                p_value = 1.0
            
            results.append({
                'feature': col,
                'importance': mean_importance,
                'std': std_importance,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            })
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('importance', ascending=False)
        
        self.validation_results['permutation_importance'] = results_df
        
        # 统计显著特征
        significant_count = results_df['is_significant'].sum()
        logger.info(f"置换重要性测试完成: {significant_count}/{len(results_df)}个特征显著(p<0.05)")
        
        return results_df
    
    def test_feature_correlation(self,
                                X: pd.DataFrame,
                                threshold: float = 0.85,
                                plot: bool = True,
                                figsize: tuple = (12, 10)) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        任务1.4.3: 特征相关性检测
        
        识别高度相关的特征对
        
        Args:
            X: 特征DataFrame
            threshold: 高相关阈值
            plot: 是否绘制热力图
            figsize: 图形大小
            
        Returns:
            相关矩阵, 高度相关特征对列表
        """
        logger.info("开始特征相关性检测...")
        
        # 计算Pearson相关矩阵
        corr_matrix = X.corr()
        
        # 识别高度相关对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        # 排序
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        self.validation_results['correlation_matrix'] = corr_matrix
        self.validation_results['high_corr_pairs'] = high_corr_pairs
        
        logger.info(f"发现{len(high_corr_pairs)}对高度相关特征(|ρ|>{threshold})")
        
        # 绘制热力图
        if plot:
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, 
                       cmap='coolwarm', 
                       center=0,
                       vmin=-1, vmax=1,
                       square=True,
                       linewidths=0.5,
                       cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            logger.info("相关性热力图已保存: feature_correlation_heatmap.png")
            plt.close()
        
        return corr_matrix, high_corr_pairs
    
    def test_vif_multicollinearity(self,
                                  X: pd.DataFrame,
                                  threshold: float = 10.0) -> pd.DataFrame:
        """
        任务1.4.4: VIF多重共线性检测
        
        使用方差膨胀因子检测多重共线性
        
        Args:
            X: 特征DataFrame
            threshold: VIF阈值（通常使用10）
            
        Returns:
            VIF表 + 高VIF特征列表
        """
        logger.info("开始VIF多重共线性检测...")
        
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # 删除包含NaN的行
        X_clean = X.dropna()
        
        if len(X_clean) < len(X.columns) + 1:
            logger.warning("数据量不足，无法计算VIF")
            return pd.DataFrame()
        
        vif_data = []
        
        for i, col in enumerate(X_clean.columns):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_data.append({
                    'feature': col,
                    'VIF': vif,
                    'has_multicollinearity': vif > threshold
                })
            except Exception as e:
                logger.warning(f"计算{col}的VIF时出错: {e}")
                vif_data.append({
                    'feature': col,
                    'VIF': np.nan,
                    'has_multicollinearity': False
                })
        
        # 转换为DataFrame并排序
        vif_df = pd.DataFrame(vif_data)
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        self.validation_results['vif'] = vif_df
        
        # 统计高VIF特征
        high_vif_count = vif_df['has_multicollinearity'].sum()
        logger.info(f"VIF检测完成: {high_vif_count}/{len(vif_df)}个特征存在多重共线性(VIF>{threshold})")
        
        if high_vif_count > 0:
            logger.info("高VIF特征:")
            for idx, row in vif_df[vif_df['has_multicollinearity']].iterrows():
                logger.info(f"  {row['feature']}: VIF={row['VIF']:.2f}")
        
        return vif_df
    
    def generate_validation_report(self, 
                                  output_path: str = 'feature_validation_report.txt') -> Dict:
        """
        生成完整的特征验证报告
        
        Returns:
            验证结果字典
        """
        logger.info("生成特征验证报告...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("特征验证报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 1. 单特征信息量
        if 'single_feature_info' in self.validation_results:
            report_lines.append("1. 单特征信息量测试")
            report_lines.append("-" * 80)
            df = self.validation_results['single_feature_info'].head(10)
            report_lines.append(df.to_string())
            report_lines.append("")
        
        # 2. 置换重要性
        if 'permutation_importance' in self.validation_results:
            report_lines.append("2. 置换重要性测试")
            report_lines.append("-" * 80)
            df = self.validation_results['permutation_importance']
            significant_df = df[df['is_significant']].head(10)
            report_lines.append(significant_df.to_string())
            report_lines.append("")
        
        # 3. 高相关特征对
        if 'high_corr_pairs' in self.validation_results:
            report_lines.append("3. 高度相关特征对")
            report_lines.append("-" * 80)
            pairs = self.validation_results['high_corr_pairs'][:10]
            for feat1, feat2, corr in pairs:
                report_lines.append(f"  {feat1} <-> {feat2}: {corr:.4f}")
            report_lines.append("")
        
        # 4. VIF多重共线性
        if 'vif' in self.validation_results:
            report_lines.append("4. VIF多重共线性检测")
            report_lines.append("-" * 80)
            df = self.validation_results['vif']
            high_vif_df = df[df['has_multicollinearity']]
            if len(high_vif_df) > 0:
                report_lines.append(high_vif_df.to_string())
            else:
                report_lines.append("  未发现多重共线性问题")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        # 保存报告
        report_text = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"特征验证报告已保存: {output_path}")
        
        return self.validation_results
    
    def suggest_feature_removal(self, 
                               importance_threshold: float = 0.001,
                               corr_threshold: float = 0.85,
                               vif_threshold: float = 10.0) -> List[str]:
        """
        基于验证结果建议移除的特征
        
        Args:
            importance_threshold: 重要性阈值
            corr_threshold: 相关性阈值
            vif_threshold: VIF阈值
            
        Returns:
            建议移除的特征列表
        """
        features_to_remove = set()
        
        # 1. 移除低重要性特征
        if 'permutation_importance' in self.validation_results:
            df = self.validation_results['permutation_importance']
            low_importance = df[df['importance'] < importance_threshold]['feature'].tolist()
            features_to_remove.update(low_importance)
            logger.info(f"低重要性特征: {len(low_importance)}个")
        
        # 2. 移除高相关特征对中的一个
        if 'high_corr_pairs' in self.validation_results:
            pairs = self.validation_results['high_corr_pairs']
            for feat1, feat2, corr in pairs:
                if corr > corr_threshold:
                    # 保留重要性更高的特征
                    if 'permutation_importance' in self.validation_results:
                        imp_df = self.validation_results['permutation_importance']
                        imp1 = imp_df[imp_df['feature'] == feat1]['importance'].values
                        imp2 = imp_df[imp_df['feature'] == feat2]['importance'].values
                        
                        if len(imp1) > 0 and len(imp2) > 0:
                            if imp1[0] < imp2[0]:
                                features_to_remove.add(feat1)
                            else:
                                features_to_remove.add(feat2)
                        else:
                            features_to_remove.add(feat2)  # 默认移除第二个
                    else:
                        features_to_remove.add(feat2)
        
        # 3. 移除高VIF特征
        if 'vif' in self.validation_results:
            df = self.validation_results['vif']
            high_vif = df[df['VIF'] > vif_threshold]['feature'].tolist()
            features_to_remove.update(high_vif)
            logger.info(f"高VIF特征: {len(high_vif)}个")
        
        features_to_remove = list(features_to_remove)
        logger.info(f"建议移除{len(features_to_remove)}个特征")
        
        return features_to_remove