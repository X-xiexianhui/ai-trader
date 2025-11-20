"""
特征重要性分析

实现置换重要性分析和消融实验
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
import logging
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PermutationImportance:
    """
    置换重要性分析器
    
    通过随机打乱特征值来评估特征重要性
    """
    
    def __init__(
        self,
        model: any,
        eval_func: Callable,
        n_repeats: int = 100,
        random_state: Optional[int] = None
    ):
        """
        初始化置换重要性分析器
        
        Args:
            model: 训练好的模型
            eval_func: 评估函数，接收(model, X, y)，返回性能指标
            n_repeats: 重复次数
            random_state: 随机种子
        """
        self.model = model
        self.eval_func = eval_func
        self.n_repeats = n_repeats
        self.random_state = random_state
        
        self.importances = {}
        self.baseline_score = None
        
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.info(f"置换重要性分析器初始化，重复次数: {n_repeats}")
    
    def analyze(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        执行置换重要性分析
        
        Args:
            X: 特征数据
            y: 目标变量
            feature_names: 特征名称列表，如果为None则使用X的列名
            
        Returns:
            Dict: 重要性分析结果
        """
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 计算基线性能
        logger.info("计算基线性能...")
        self.baseline_score = self.eval_func(self.model, X, y)
        logger.info(f"基线性能: {self.baseline_score}")
        
        # 对每个特征进行置换测试
        self.importances = {}
        
        for feature_idx, feature_name in enumerate(tqdm(feature_names, desc="分析特征重要性")):
            scores = []
            
            for _ in range(self.n_repeats):
                # 复制数据
                X_permuted = X.copy()
                
                # 随机打乱该特征
                if isinstance(X_permuted, pd.DataFrame):
                    X_permuted.iloc[:, feature_idx] = np.random.permutation(
                        X_permuted.iloc[:, feature_idx].values
                    )
                else:
                    X_permuted[:, feature_idx] = np.random.permutation(
                        X_permuted[:, feature_idx]
                    )
                
                # 评估性能
                score = self.eval_func(self.model, X_permuted, y)
                scores.append(score)
            
            # 计算重要性（基线 - 置换后的平均性能）
            scores = np.array(scores)
            importance = self.baseline_score - scores.mean()
            
            # 统计检验
            t_stat, p_value = stats.ttest_1samp(scores, self.baseline_score)
            
            self.importances[feature_name] = {
                'importance': importance,
                'importance_std': scores.std(),
                'permuted_scores_mean': scores.mean(),
                'permuted_scores_std': scores.std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05
            }
        
        logger.info("置换重要性分析完成")
        return self.importances
    
    def get_ranking(self) -> pd.DataFrame:
        """
        获取特征重要性排名
        
        Returns:
            DataFrame: 排序后的特征重要性
        """
        if not self.importances:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.importances).T
        df = df.sort_values('importance', ascending=False)
        return df
    
    def plot_importances(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个特征
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_ranking().head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
            
            # 绘制条形图
            y_pos = np.arange(len(df))
            ax.barh(y_pos, df['importance'], xerr=df['importance_std'])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df.index)
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'Top {top_n} Feature Importances (Permutation)')
            
            # 标记显著性
            for i, (idx, row) in enumerate(df.iterrows()):
                if row['is_significant']:
                    ax.text(row['importance'], i, ' *', va='center', fontsize=12, color='red')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"特征重要性图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")
    
    def print_summary(self, top_n: int = 20):
        """打印重要性摘要"""
        df = self.get_ranking().head(top_n)
        
        print("\n" + "="*80)
        print("特征重要性分析（置换方法）")
        print("="*80)
        print(f"\n基线性能: {self.baseline_score:.4f}")
        print(f"重复次数: {self.n_repeats}")
        print(f"\nTop {top_n} 特征:")
        print("-" * 80)
        
        for idx, row in df.iterrows():
            sig_marker = "*" if row['is_significant'] else " "
            print(f"{sig_marker} {idx:30s} | "
                  f"重要性: {row['importance']:8.4f} ± {row['importance_std']:.4f} | "
                  f"p值: {row['p_value']:.4f}")
        
        print("\n注: * 表示统计显著 (p < 0.05)")
        print("="*80 + "\n")


class AblationStudy:
    """
    消融实验
    
    通过逐步移除特征组来评估其重要性
    """
    
    def __init__(
        self,
        train_func: Callable,
        eval_func: Callable
    ):
        """
        初始化消融实验
        
        Args:
            train_func: 训练函数，接收(X_train, y_train, X_val, y_val)，返回模型
            eval_func: 评估函数，接收(model, X_test, y_test)，返回性能指标
        """
        self.train_func = train_func
        self.eval_func = eval_func
        
        self.results = {}
        self.baseline_score = None
        
        logger.info("消融实验初始化")
    
    def run_ablation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_groups: Dict[str, List[str]]
    ) -> Dict:
        """
        执行消融实验
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            X_test: 测试特征
            y_test: 测试标签
            feature_groups: 特征组字典，{组名: [特征列表]}
            
        Returns:
            Dict: 消融实验结果
        """
        # 1. 基线：使用所有特征
        logger.info("训练基线模型（所有特征）...")
        baseline_model = self.train_func(X_train, y_train, X_val, y_val)
        self.baseline_score = self.eval_func(baseline_model, X_test, y_test)
        
        self.results['baseline'] = {
            'features': X_train.columns.tolist(),
            'num_features': X_train.shape[1],
            'score': self.baseline_score,
            'score_diff': 0.0,
            'score_diff_pct': 0.0
        }
        
        logger.info(f"基线性能: {self.baseline_score:.4f}")
        
        # 2. 逐个移除特征组
        for group_name, features in feature_groups.items():
            logger.info(f"\n移除特征组: {group_name} ({len(features)}个特征)")
            
            # 移除该组特征
            remaining_features = [f for f in X_train.columns if f not in features]
            
            if not remaining_features:
                logger.warning(f"移除{group_name}后没有剩余特征，跳过")
                continue
            
            # 训练模型
            X_train_ablated = X_train[remaining_features]
            X_val_ablated = X_val[remaining_features]
            X_test_ablated = X_test[remaining_features]
            
            model = self.train_func(X_train_ablated, y_train, X_val_ablated, y_val)
            score = self.eval_func(model, X_test_ablated, y_test)
            
            # 计算性能变化
            score_diff = score - self.baseline_score
            score_diff_pct = (score_diff / self.baseline_score * 100) if self.baseline_score != 0 else 0
            
            self.results[group_name] = {
                'removed_features': features,
                'remaining_features': remaining_features,
                'num_features': len(remaining_features),
                'score': score,
                'score_diff': score_diff,
                'score_diff_pct': score_diff_pct
            }
            
            logger.info(f"性能: {score:.4f} (变化: {score_diff:+.4f}, {score_diff_pct:+.2f}%)")
        
        # 3. 只使用每个特征组（单独测试）
        for group_name, features in feature_groups.items():
            logger.info(f"\n只使用特征组: {group_name}")
            
            # 只保留该组特征
            available_features = [f for f in features if f in X_train.columns]
            
            if not available_features:
                logger.warning(f"{group_name}中没有可用特征，跳过")
                continue
            
            # 训练模型
            X_train_only = X_train[available_features]
            X_val_only = X_val[available_features]
            X_test_only = X_test[available_features]
            
            model = self.train_func(X_train_only, y_train, X_val_only, y_val)
            score = self.eval_func(model, X_test_only, y_test)
            
            # 计算性能变化
            score_diff = score - self.baseline_score
            score_diff_pct = (score_diff / self.baseline_score * 100) if self.baseline_score != 0 else 0
            
            self.results[f'{group_name}_only'] = {
                'features': available_features,
                'num_features': len(available_features),
                'score': score,
                'score_diff': score_diff,
                'score_diff_pct': score_diff_pct
            }
            
            logger.info(f"性能: {score:.4f} (变化: {score_diff:+.4f}, {score_diff_pct:+.2f}%)")
        
        logger.info("\n消融实验完成")
        return self.results
    
    def get_summary(self) -> pd.DataFrame:
        """
        获取消融实验摘要
        
        Returns:
            DataFrame: 摘要表格
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                'experiment': name,
                'num_features': result['num_features'],
                'score': result['score'],
                'score_diff': result['score_diff'],
                'score_diff_pct': result['score_diff_pct']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('score', ascending=False)
        return df
    
    def print_summary(self):
        """打印消融实验摘要"""
        df = self.get_summary()
        
        print("\n" + "="*80)
        print("消融实验摘要")
        print("="*80)
        print(f"\n基线性能: {self.baseline_score:.4f}")
        print("\n实验结果:")
        print("-" * 80)
        print(f"{'实验名称':<30} | {'特征数':>8} | {'性能':>10} | {'变化':>10} | {'变化%':>10}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['experiment']:<30} | "
                  f"{row['num_features']:>8} | "
                  f"{row['score']:>10.4f} | "
                  f"{row['score_diff']:>+10.4f} | "
                  f"{row['score_diff_pct']:>+9.2f}%")
        
        print("="*80 + "\n")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        绘制消融实验结果
        
        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_summary()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 绘制性能对比
            ax1.barh(range(len(df)), df['score'])
            ax1.set_yticks(range(len(df)))
            ax1.set_yticklabels(df['experiment'])
            ax1.set_xlabel('Performance Score')
            ax1.set_title('Ablation Study: Performance Comparison')
            ax1.axvline(self.baseline_score, color='r', linestyle='--', label='Baseline')
            ax1.legend()
            
            # 绘制性能变化
            colors = ['green' if x >= 0 else 'red' for x in df['score_diff']]
            ax2.barh(range(len(df)), df['score_diff'], color=colors)
            ax2.set_yticks(range(len(df)))
            ax2.set_yticklabels(df['experiment'])
            ax2.set_xlabel('Performance Change')
            ax2.set_title('Ablation Study: Performance Change from Baseline')
            ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"消融实验图已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib未安装，无法绘图")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(X.iloc[:, 0] * 2 + X.iloc[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5)
    
    # 简单的模型和评估函数
    class SimpleModel:
        def __init__(self, X, y):
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
    
    def eval_func(model, X, y):
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        return -mse  # 负MSE，越大越好
    
    # 测试置换重要性
    print("\n测试置换重要性分析...")
    model = SimpleModel(X, y)
    perm_importance = PermutationImportance(model, eval_func, n_repeats=50)
    importances = perm_importance.analyze(X, y)
    perm_importance.print_summary(top_n=10)