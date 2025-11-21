"""
TS2Vec完整训练示例

演示如何使用TS2Vec模块进行端到端训练:
1. 数据准备
2. 模型训练
3. 模型评估
4. 结果可视化
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# 导入TS2Vec模块
from src.models.ts2vec.data_preparation import (
    SlidingWindowGenerator,
    TimeSeriesAugmentation,
    TS2VecDataset
)
from src.models.ts2vec.model import TS2VecModel
from src.models.ts2vec.training import TS2VecTrainer
from src.models.ts2vec.evaluation import TS2VecEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000, seq_len: int = 256) -> np.ndarray:
    """
    生成合成的OHLC数据用于演示
    
    Args:
        n_samples: 样本数量
        seq_len: 序列长度
        
    Returns:
        OHLC数据 [n_samples, 4]
    """
    logger.info(f"生成{n_samples}条合成OHLC数据...")
    
    # 生成价格序列(随机游走)
    returns = np.random.randn(n_samples) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    
    # 生成OHLC
    high = close * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    open_price = close * (1 + np.random.randn(n_samples) * 0.003)
    
    # 确保OHLC一致性
    high = np.maximum(high, np.maximum(open_price, close))
    low = np.minimum(low, np.minimum(open_price, close))
    
    data = np.column_stack([open_price, high, low, close])
    
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"价格范围: [{data.min():.2f}, {data.max():.2f}]")
    
    return data


def prepare_datasets(data: np.ndarray,
                     window_length: int = 256,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15):
    """
    准备训练、验证和测试数据集
    
    Args:
        data: 原始OHLC数据
        window_length: 窗口长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    logger.info("准备数据集...")
    
    # 划分数据
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    logger.info(f"训练集: {len(train_data)}条")
    logger.info(f"验证集: {len(val_data)}条")
    logger.info(f"测试集: {len(test_data)}条")
    
    # 创建数据集
    train_dataset = TS2VecDataset(
        train_data,
        window_length=window_length,
        stride=1
    )
    
    val_dataset = TS2VecDataset(
        val_data,
        window_length=window_length,
        stride=1
    )
    
    test_dataset = TS2VecDataset(
        test_data,
        window_length=window_length,
        stride=1
    )
    
    logger.info(f"训练窗口数: {len(train_dataset)}")
    logger.info(f"验证窗口数: {len(val_dataset)}")
    logger.info(f"测试窗口数: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def train_ts2vec_model(train_dataset,
                       val_dataset,
                       config: dict):
    """
    训练TS2Vec模型
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        config: 训练配置
        
    Returns:
        训练好的模型和训练历史
    """
    logger.info("=" * 80)
    logger.info("开始训练TS2Vec模型")
    logger.info("=" * 80)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    model = TS2VecModel(
        input_dim=4,  # OHLC
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_layers=config['num_layers'],
        temperature=config['temperature']
    )
    
    # 创建训练器
    trainer = TS2VecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_epochs=config['warmup_epochs'],
        patience=config['patience'],
        save_dir=config['save_dir'],
        device=device
    )
    
    # 训练
    history = trainer.train()
    
    # 加载最佳模型
    trainer.load_best_model()
    
    logger.info("训练完成!")
    
    return model, history


def evaluate_model(model,
                   test_dataset,
                   test_data: np.ndarray,
                   config: dict):
    """
    评估TS2Vec模型
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        test_data: 原始测试数据
        config: 评估配置
        
    Returns:
        评估结果
    """
    logger.info("=" * 80)
    logger.info("评估TS2Vec模型")
    logger.info("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建评估器
    evaluator = TS2VecEvaluator(model, device=device)
    
    # 创建测试数据加载器
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 1. 评估embedding质量
    logger.info("\n1. 评估Embedding质量...")
    embedding_quality = evaluator.evaluate_embedding_quality(test_loader)
    
    # 2. 聚类质量评估
    logger.info("\n2. 评估聚类质量...")
    # 准备测试数据
    test_windows = test_dataset.windows[:1000]  # 使用前1000个窗口
    test_tensor = torch.FloatTensor(test_windows)
    
    clustering_results = evaluator.clustering_quality(
        test_tensor,
        n_clusters=5
    )
    
    # 3. t-SNE可视化
    logger.info("\n3. 生成t-SNE可视化...")
    evaluator.tsne_visualization(
        test_tensor,
        save_path='ts2vec_tsne.png',
        n_samples=1000
    )
    
    # 4. 线性探测(需要标签)
    logger.info("\n4. 线性探测评估...")
    # 生成简单的涨跌标签
    future_returns = np.diff(test_data[:, 3])  # 使用收盘价
    labels = (future_returns > 0).astype(int)
    
    # 准备数据
    n_samples = min(1000, len(test_windows) - 5)
    train_size = int(n_samples * 0.7)
    
    train_windows = test_windows[:train_size]
    train_labels = labels[:train_size]
    test_windows_probe = test_windows[train_size:n_samples]
    test_labels = labels[train_size:n_samples]
    
    linear_probing_results = evaluator.linear_probing(
        torch.FloatTensor(train_windows),
        torch.LongTensor(train_labels),
        torch.FloatTensor(test_windows_probe),
        torch.LongTensor(test_labels)
    )
    
    # 5. 生成评估报告
    logger.info("\n5. 生成评估报告...")
    report = evaluator.generate_evaluation_report('ts2vec_evaluation_report.txt')
    
    logger.info("\n评估完成!")
    
    return report


def visualize_training_history(history: dict, save_path: str = 'training_history.png'):
    """
    可视化训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    logger.info("生成训练历史可视化...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1].plot(history['learning_rate'], linewidth=2, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练历史已保存: {save_path}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("TS2Vec完整训练示例")
    logger.info("=" * 80)
    
    # 配置
    config = {
        # 数据配置
        'n_samples': 10000,
        'window_length': 256,
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        
        # 模型配置
        'hidden_dim': 256,
        'output_dim': 128,
        'num_layers': 10,
        'temperature': 0.1,
        
        # 训练配置
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'warmup_epochs': 5,
        'patience': 10,
        'save_dir': 'models/checkpoints/ts2vec'
    }
    
    # 创建保存目录
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # 1. 生成数据
    logger.info("\n步骤1: 生成合成数据")
    data = generate_synthetic_data(n_samples=config['n_samples'])
    
    # 2. 准备数据集
    logger.info("\n步骤2: 准备数据集")
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        data,
        window_length=config['window_length'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio']
    )
    
    # 3. 训练模型
    logger.info("\n步骤3: 训练模型")
    model, history = train_ts2vec_model(train_dataset, val_dataset, config)
    
    # 4. 可视化训练历史
    logger.info("\n步骤4: 可视化训练历史")
    visualize_training_history(history)
    
    # 5. 评估模型
    logger.info("\n步骤5: 评估模型")
    test_data = data[int(len(data) * (config['train_ratio'] + config['val_ratio'])):]
    evaluation_results = evaluate_model(model, test_dataset, test_data, config)
    
    # 6. 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("训练和评估总结")
    logger.info("=" * 80)
    logger.info(f"最终训练损失: {history['train_loss'][-1]:.4f}")
    logger.info(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    
    if 'embedding_quality' in evaluation_results:
        quality = evaluation_results['embedding_quality']
        logger.info(f"正样本相似度: {quality['pos_sim_mean']:.4f}")
        logger.info(f"负样本相似度: {quality['neg_sim_mean']:.4f}")
        logger.info(f"分离度: {quality['separation']:.4f}")
    
    if 'clustering' in evaluation_results:
        clustering = evaluation_results['clustering']
        logger.info(f"轮廓系数: {clustering['silhouette_score']:.4f}")
    
    if 'linear_probing' in evaluation_results:
        probing = evaluation_results['linear_probing']
        logger.info(f"线性探测准确率: {probing['test_accuracy']:.4f}")
    
    logger.info("\n所有任务完成!")
    logger.info("生成的文件:")
    logger.info("  - models/checkpoints/ts2vec/best_model.pt (最佳模型)")
    logger.info("  - training_history.png (训练历史)")
    logger.info("  - ts2vec_tsne.png (t-SNE可视化)")
    logger.info("  - ts2vec_evaluation_report.txt (评估报告)")


if __name__ == '__main__':
    main()