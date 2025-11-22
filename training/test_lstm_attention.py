"""
LSTM+Attention神经网络测试和评估脚本

功能：
1. 从processed加载测试数据
2. 使用k-means对市场状态向量进行聚类
3. 回归任务评估（波动率和收益率）
   - 均方误差 (MSE)
   - 平均绝对误差 (MAE)
   - R² 分数
4. 分类任务评估（市场状态）
   - 准确率 (Accuracy)
   - 精确度 (Precision)
   - 召回率 (Recall)
   - F1 值
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
import logging
from datetime import datetime
from tqdm import tqdm
import json
import pickle

from src.models.lstm_attention import create_model
from src.models.data_loader import create_dataloaders
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    name='lstm_attention_test',
    log_level='INFO',
    log_dir='logs',
    log_file='lstm_attention_test.log'
)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        output_dir: Path
    ):
        """
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            device: 设备
            output_dir: 输出目录
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储预测结果
        self.state_vectors = []
        self.predictions = {
            'class_logits': [],
            'class_pred': [],
            'volatility': [],
            'returns': []
        }
        self.targets = {
            'market_state': [],
            'volatility': [],
            'returns': []
        }
    
    def extract_predictions(self):
        """提取模型预测结果"""
        logger.info("提取模型预测...")
        
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='提取预测'):
                # 将数据移到设备
                sequences = batch['sequence'].to(self.device)
                market_states = batch['market_state'].cpu().numpy()
                volatilities = batch['volatility'].cpu().numpy()
                returns = batch['returns'].cpu().numpy()
                
                # 前向传播
                outputs = self.model(sequences, return_attention=False)
                
                # 提取市场状态向量
                state_vectors = outputs['state_vector'].cpu().numpy()
                self.state_vectors.append(state_vectors)
                
                # 提取预测
                class_logits = outputs['class_logits'].cpu().numpy()
                class_pred = np.argmax(class_logits, axis=1)
                volatility_pred = outputs['volatility'].cpu().numpy()
                returns_pred = outputs['returns'].cpu().numpy()
                
                self.predictions['class_logits'].append(class_logits)
                self.predictions['class_pred'].append(class_pred)
                self.predictions['volatility'].append(volatility_pred)
                self.predictions['returns'].append(returns_pred)
                
                # 存储目标
                self.targets['market_state'].append(market_states)
                self.targets['volatility'].append(volatilities)
                self.targets['returns'].append(returns)
        
        # 合并所有批次
        self.state_vectors = np.vstack(self.state_vectors)
        self.predictions['class_logits'] = np.vstack(self.predictions['class_logits'])
        self.predictions['class_pred'] = np.concatenate(self.predictions['class_pred'])
        self.predictions['volatility'] = np.vstack(self.predictions['volatility']).flatten()
        self.predictions['returns'] = np.vstack(self.predictions['returns']).flatten()
        
        self.targets['market_state'] = np.concatenate(self.targets['market_state'])
        self.targets['volatility'] = np.vstack(self.targets['volatility']).flatten()
        self.targets['returns'] = np.vstack(self.targets['returns']).flatten()
        
        logger.info(f"提取完成: {len(self.state_vectors)} 个样本")
    
    def cluster_state_vectors(self, n_clusters: int = 8):
        """使用K-means对市场状态向量进行聚类"""
        logger.info(f"\n使用K-means聚类 (n_clusters={n_clusters})...")
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.state_vectors)
        
        # 保存聚类模型
        cluster_path = self.output_dir / 'kmeans_model.pkl'
        with open(cluster_path, 'wb') as f:
            pickle.dump(kmeans, f)
        logger.info(f"聚类模型已保存: {cluster_path}")
        
        # 分析每个聚类的特征
        logger.info("\n聚类分析:")
        cluster_stats = []
        
        for i in range(n_clusters):
            mask = cluster_labels == i
            cluster_size = mask.sum()
            
            # 计算该聚类的平均市场状态分布
            cluster_states = self.targets['market_state'][mask]
            state_dist = np.bincount(cluster_states, minlength=4) / cluster_size
            
            # 计算该聚类的平均波动率和收益率
            avg_vol = self.targets['volatility'][mask].mean()
            avg_ret = self.targets['returns'][mask].mean()
            
            cluster_stats.append({
                'cluster': i,
                'size': int(cluster_size),
                'uptrend_ratio': float(state_dist[0]),
                'downtrend_ratio': float(state_dist[1]),
                'sideways_ratio': float(state_dist[2]),
                'reversal_ratio': float(state_dist[3]),
                'avg_volatility': float(avg_vol),
                'avg_returns': float(avg_ret)
            })
            
            logger.info(f"聚类 {i}: 样本数={cluster_size}")
            logger.info(f"  市场状态分布: 上涨={state_dist[0]:.2%}, 下跌={state_dist[1]:.2%}, "
                       f"震荡={state_dist[2]:.2%}, 反转={state_dist[3]:.2%}")
            logger.info(f"  平均波动率={avg_vol:.4f}, 平均收益率={avg_ret:.4f}")
        
        # 保存聚类统计
        stats_path = self.output_dir / 'cluster_stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
        
        # 可视化聚类（使用t-SNE降维）
        self._visualize_clusters(cluster_labels, n_clusters)
        
        return cluster_labels
    
    def _visualize_clusters(self, cluster_labels: np.ndarray, n_clusters: int):
        """可视化聚类结果"""
        logger.info("生成聚类可视化...")
        
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        state_vectors_2d = tsne.fit_transform(self.state_vectors)
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 子图1: 按聚类标签着色
        scatter1 = axes[0].scatter(
            state_vectors_2d[:, 0],
            state_vectors_2d[:, 1],
            c=cluster_labels,
            cmap='tab10',
            alpha=0.6,
            s=10
        )
        axes[0].set_title('市场状态向量聚类 (K-means)', fontsize=14)
        axes[0].set_xlabel('t-SNE 维度 1')
        axes[0].set_ylabel('t-SNE 维度 2')
        plt.colorbar(scatter1, ax=axes[0], label='聚类标签')
        
        # 子图2: 按真实市场状态着色
        state_names = ['上涨', '下跌', '震荡', '反转']
        colors = ['green', 'red', 'blue', 'orange']
        for state_idx, (name, color) in enumerate(zip(state_names, colors)):
            mask = self.targets['market_state'] == state_idx
            axes[1].scatter(
                state_vectors_2d[mask, 0],
                state_vectors_2d[mask, 1],
                c=color,
                label=name,
                alpha=0.6,
                s=10
            )
        axes[1].set_title('市场状态向量 (真实标签)', fontsize=14)
        axes[1].set_xlabel('t-SNE 维度 1')
        axes[1].set_ylabel('t-SNE 维度 2')
        axes[1].legend()
        
        plt.tight_layout()
        
        # 保存图形
        fig_path = self.output_dir / 'cluster_visualization.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"聚类可视化已保存: {fig_path}")
    
    def evaluate_regression(self):
        """评估回归任务（波动率和收益率）"""
        logger.info("\n" + "=" * 80)
        logger.info("回归任务评估")
        logger.info("=" * 80)
        
        results = {}
        
        # 1. 波动率预测评估
        logger.info("\n1. 波动率预测:")
        vol_mse = mean_squared_error(self.targets['volatility'], self.predictions['volatility'])
        vol_mae = mean_absolute_error(self.targets['volatility'], self.predictions['volatility'])
        vol_r2 = r2_score(self.targets['volatility'], self.predictions['volatility'])
        
        logger.info(f"  均方误差 (MSE): {vol_mse:.6f}")
        logger.info(f"  平均绝对误差 (MAE): {vol_mae:.6f}")
        logger.info(f"  R² 分数: {vol_r2:.6f}")
        
        results['volatility'] = {
            'mse': float(vol_mse),
            'mae': float(vol_mae),
            'r2': float(vol_r2)
        }
        
        # 2. 收益率预测评估
        logger.info("\n2. 收益率预测:")
        ret_mse = mean_squared_error(self.targets['returns'], self.predictions['returns'])
        ret_mae = mean_absolute_error(self.targets['returns'], self.predictions['returns'])
        ret_r2 = r2_score(self.targets['returns'], self.predictions['returns'])
        
        logger.info(f"  均方误差 (MSE): {ret_mse:.6f}")
        logger.info(f"  平均绝对误差 (MAE): {ret_mae:.6f}")
        logger.info(f"  R² 分数: {ret_r2:.6f}")
        
        results['returns'] = {
            'mse': float(ret_mse),
            'mae': float(ret_mae),
            'r2': float(ret_r2)
        }
        
        # 可视化回归结果
        self._visualize_regression()
        
        return results
    
    def _visualize_regression(self):
        """可视化回归结果"""
        logger.info("生成回归结果可视化...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 波动率：散点图
        axes[0, 0].scatter(
            self.targets['volatility'],
            self.predictions['volatility'],
            alpha=0.5,
            s=10
        )
        axes[0, 0].plot(
            [self.targets['volatility'].min(), self.targets['volatility'].max()],
            [self.targets['volatility'].min(), self.targets['volatility'].max()],
            'r--', lw=2
        )
        axes[0, 0].set_xlabel('真实波动率')
        axes[0, 0].set_ylabel('预测波动率')
        axes[0, 0].set_title('波动率预测 vs 真实值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 波动率：残差图
        vol_residuals = self.predictions['volatility'] - self.targets['volatility']
        axes[0, 1].scatter(
            self.predictions['volatility'],
            vol_residuals,
            alpha=0.5,
            s=10
        )
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('预测波动率')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('波动率预测残差')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收益率：散点图
        axes[1, 0].scatter(
            self.targets['returns'],
            self.predictions['returns'],
            alpha=0.5,
            s=10
        )
        axes[1, 0].plot(
            [self.targets['returns'].min(), self.targets['returns'].max()],
            [self.targets['returns'].min(), self.targets['returns'].max()],
            'r--', lw=2
        )
        axes[1, 0].set_xlabel('真实收益率')
        axes[1, 0].set_ylabel('预测收益率')
        axes[1, 0].set_title('收益率预测 vs 真实值')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 收益率：残差图
        ret_residuals = self.predictions['returns'] - self.targets['returns']
        axes[1, 1].scatter(
            self.predictions['returns'],
            ret_residuals,
            alpha=0.5,
            s=10
        )
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('预测收益率')
        axes[1, 1].set_ylabel('残差')
        axes[1, 1].set_title('收益率预测残差')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图形
        fig_path = self.output_dir / 'regression_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"回归结果可视化已保存: {fig_path}")
    
    def evaluate_classification(self):
        """评估分类任务（市场状态）"""
        logger.info("\n" + "=" * 80)
        logger.info("分类任务评估")
        logger.info("=" * 80)
        
        # 计算指标
        accuracy = accuracy_score(self.targets['market_state'], self.predictions['class_pred'])
        precision = precision_score(
            self.targets['market_state'],
            self.predictions['class_pred'],
            average='weighted',
            zero_division=0
        )
        recall = recall_score(
            self.targets['market_state'],
            self.predictions['class_pred'],
            average='weighted',
            zero_division=0
        )
        f1 = f1_score(
            self.targets['market_state'],
            self.predictions['class_pred'],
            average='weighted',
            zero_division=0
        )
        
        logger.info(f"\n准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"精确度 (Precision): {precision:.4f}")
        logger.info(f"召回率 (Recall): {recall:.4f}")
        logger.info(f"F1 值: {f1:.4f}")
        
        # 详细分类报告
        state_names = ['上涨', '下跌', '震荡', '反转']
        logger.info("\n详细分类报告:")
        report = classification_report(
            self.targets['market_state'],
            self.predictions['class_pred'],
            target_names=state_names,
            zero_division=0
        )
        logger.info("\n" + report)
        
        # 混淆矩阵
        cm = confusion_matrix(self.targets['market_state'], self.predictions['class_pred'])
        self._plot_confusion_matrix(cm, state_names)
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'classification_report': report
        }
        
        return results
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """绘制混淆矩阵"""
        logger.info("生成混淆矩阵...")
        
        plt.figure(figsize=(10, 8))
        
        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热力图
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': '比例'}
        )
        
        plt.title('市场状态分类混淆矩阵', fontsize=14)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        # 保存图形
        fig_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"混淆矩阵已保存: {fig_path}")
    
    def generate_report(self, regression_results: dict, classification_results: dict):
        """生成完整评估报告"""
        logger.info("\n生成评估报告...")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_samples': len(self.state_vectors),
            'regression': regression_results,
            'classification': classification_results
        }
        
        # 保存JSON报告
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            # 移除classification_report（太长）
            report_copy = report.copy()
            report_copy['classification'] = {
                k: v for k, v in classification_results.items()
                if k != 'classification_report'
            }
            json.dump(report_copy, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        text_report = []
        text_report.append("=" * 80)
        text_report.append("LSTM+Attention模型评估报告")
        text_report.append("=" * 80)
        text_report.append(f"生成时间: {report['timestamp']}")
        text_report.append(f"测试样本数: {report['test_samples']}")
        text_report.append("")
        
        text_report.append("回归任务评估")
        text_report.append("-" * 80)
        text_report.append("1. 波动率预测:")
        text_report.append(f"   MSE: {regression_results['volatility']['mse']:.6f}")
        text_report.append(f"   MAE: {regression_results['volatility']['mae']:.6f}")
        text_report.append(f"   R²:  {regression_results['volatility']['r2']:.6f}")
        text_report.append("")
        text_report.append("2. 收益率预测:")
        text_report.append(f"   MSE: {regression_results['returns']['mse']:.6f}")
        text_report.append(f"   MAE: {regression_results['returns']['mae']:.6f}")
        text_report.append(f"   R²:  {regression_results['returns']['r2']:.6f}")
        text_report.append("")
        
        text_report.append("分类任务评估")
        text_report.append("-" * 80)
        text_report.append(f"准确率:  {classification_results['accuracy']:.4f}")
        text_report.append(f"精确度:  {classification_results['precision']:.4f}")
        text_report.append(f"召回率:  {classification_results['recall']:.4f}")
        text_report.append(f"F1值:    {classification_results['f1']:.4f}")
        text_report.append("")
        text_report.append("详细分类报告:")
        text_report.append(classification_results['classification_report'])
        
        # 保存文本报告
        text_path = self.output_dir / 'evaluation_report.txt'
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_report))
        
        logger.info(f"评估报告已保存:")
        logger.info(f"  - JSON: {report_path}")
        logger.info(f"  - 文本: {text_path}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("LSTM+Attention神经网络测试和评估")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 路径设置
    model_dir = project_root / 'models' / 'lstm_attention'
    data_dir = project_root / 'data' / 'processed'
    output_dir = project_root / 'training' / 'output' / 'lstm_attention_evaluation'
    
    # 检查模型文件
    model_path = model_dir / 'checkpoint_best.pth'
    config_path = model_dir / 'config.json'
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行 training/train_lstm_attention.py")
        return
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info("\n模型配置:")
    for key, value in config.items():
        if key != 'loss_weights':
            logger.info(f"  {key}: {value}")
    
    # 创建数据加载器
    logger.info("\n创建数据加载器...")
    train_path = data_dir / 'MES_train.csv'
    val_path = data_dir / 'MES_val.csv'
    test_path = data_dir / 'MES_test.csv'
    
    _, _, test_loader, _ = create_dataloaders(
        str(train_path),
        str(val_path),
        str(test_path),
        seq_len=config['seq_len'],
        future_periods=config['future_periods'],
        batch_size=config['batch_size'],
        num_workers=4,
        scaler_type='standard'
    )
    
    # 创建模型
    logger.info("\n加载模型...")
    model = create_model(config).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"模型加载完成 (epoch {checkpoint['epoch']})")
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=output_dir
    )
    
    # 1. 提取预测结果
    evaluator.extract_predictions()
    
    # 2. K-means聚类
    cluster_labels = evaluator.cluster_state_vectors(n_clusters=8)
    
    # 3. 回归任务评估
    regression_results = evaluator.evaluate_regression()
    
    # 4. 分类任务评估
    classification_results = evaluator.evaluate_classification()
    
    # 5. 生成完整报告
    evaluator.generate_report(regression_results, classification_results)
    
    logger.info("\n" + "=" * 80)
    logger.info("评估完成！")
    logger.info("=" * 80)
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {output_dir}")


if __name__ == '__main__':
    main()