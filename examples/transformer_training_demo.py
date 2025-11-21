"""
Transformer状态建模器完整训练示例

演示如何:
1. 加载预训练的TS2Vec模型
2. 准备数据(OHLC + 手工特征)
3. 特征融合(TS2Vec embeddings + 手工特征)
4. 构建Transformer输入序列
5. 训练Transformer模型(带辅助任务)
6. 评估模型性能
7. 可视化结果

对应任务: Module 3 完整流程演示
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# 导入模块
from src.models.ts2vec.model import TS2VecModel
from src.models.transformer.model import TransformerStateModel
from src.models.transformer.feature_fusion import (
    TS2VecEmbeddingGenerator,
    FeatureFusion,
    SequenceBuilder
)
from src.models.transformer.auxiliary_tasks import (
    TransformerWithAuxiliaryTasks,
    MultiTaskLoss
)
from src.models.transformer.training import (
    TransformerTrainer,
    create_dataloaders
)
from src.models.transformer.evaluation import (
    SupervisedMetrics,
    StateRepresentationQuality,
    AttentionVisualizer
)
from src.features import FeatureCalculator
from src.features.feature_scaler import FeatureNormalizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    生成合成的OHLC数据用于演示
    
    Args:
        n_samples: 样本数量
        
    Returns:
        OHLC DataFrame
    """
    logger.info(f"生成{n_samples}条合成数据...")
    
    np.random.seed(42)
    
    # 生成价格序列(带趋势和噪声)
    trend = np.linspace(100, 120, n_samples)
    noise = np.random.randn(n_samples) * 2
    close = trend + noise
    
    # 生成OHLC
    high = close + np.abs(np.random.randn(n_samples) * 1)
    low = close - np.abs(np.random.randn(n_samples) * 1)
    open_price = close + np.random.randn(n_samples) * 0.5
    volume = np.random.randint(1000, 10000, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    # 添加时间索引
    df.index = pd.date_range('2020-01-01', periods=n_samples, freq='5T')
    
    logger.info(f"数据生成完成: {df.shape}")
    
    return df


def compute_hand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算27维手工特征
    
    Args:
        df: OHLC数据
        
    Returns:
        特征DataFrame
    """
    logger.info("计算手工特征...")
    
    calculator = FeatureCalculator()
    features = calculator.calculate_all_features(df)
    
    logger.info(f"特征计算完成: {features.shape}")
    
    return features


def create_future_returns(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """
    创建未来收益率标签
    
    Args:
        df: OHLC数据
        horizon: 预测时间跨度
        
    Returns:
        未来收益率
    """
    returns = np.log(df['Close'].shift(-horizon) / df['Close'])
    return returns


def prepare_transformer_data(ohlc_df: pd.DataFrame,
                             ts2vec_model: TS2VecModel,
                             device: str = 'cpu') -> tuple:
    """
    准备Transformer训练数据
    
    Args:
        ohlc_df: OHLC数据
        ts2vec_model: 预训练的TS2Vec模型
        device: 设备
        
    Returns:
        (sequences, labels)
    """
    logger.info("准备Transformer数据...")
    
    # 1. 计算手工特征
    hand_features = compute_hand_features(ohlc_df)
    
    # 2. 归一化手工特征
    normalizer = FeatureNormalizer()
    hand_features_norm = normalizer.fit_transform(hand_features)
    
    # 3. 生成TS2Vec embeddings
    embedding_generator = TS2VecEmbeddingGenerator(
        ts2vec_model, device, use_projection=True
    )
    
    # 创建滑动窗口
    from src.models.ts2vec.data_preparation import SlidingWindowGenerator
    window_gen = SlidingWindowGenerator(window_length=256, stride=1)
    windows = window_gen.generate_from_dataframe(
        ohlc_df, columns=['Open', 'High', 'Low', 'Close']
    )
    
    # 生成embeddings
    windows_tensor = torch.FloatTensor(windows)
    embeddings = embedding_generator.generate(windows_tensor)
    
    # 如果embeddings是3D,平均池化
    if len(embeddings.shape) == 3:
        embeddings = embeddings.mean(dim=1)  # [N, 128]
    
    logger.info(f"Embeddings形状: {embeddings.shape}")
    
    # 4. 对齐时间(embeddings可能比原始数据短)
    min_len = min(len(embeddings), len(hand_features_norm))
    embeddings = embeddings[:min_len]
    hand_features_aligned = torch.FloatTensor(hand_features_norm.values[:min_len])
    
    # 5. 特征融合
    feature_fusion = FeatureFusion(
        embedding_dim=128,
        feature_dim=27,
        fusion_method='concat'
    )
    fused_features = feature_fusion.fuse(embeddings, hand_features_aligned)
    
    logger.info(f"融合特征形状: {fused_features.shape}")
    
    # 6. 创建未来收益率标签
    future_returns = create_future_returns(ohlc_df, horizon=5)
    labels = torch.FloatTensor(future_returns.values[:min_len]).unsqueeze(1)
    
    # 7. 构建序列
    sequence_builder = SequenceBuilder(sequence_length=64, stride=1)
    sequences, sequence_labels = sequence_builder.build_sequences(
        fused_features, labels.squeeze()
    )
    
    logger.info(f"序列形状: {sequences.shape}, 标签形状: {sequence_labels.shape}")
    
    return sequences, sequence_labels.unsqueeze(1)


def train_transformer_model(sequences: torch.Tensor,
                            labels: torch.Tensor,
                            device: str = 'cpu') -> TransformerWithAuxiliaryTasks:
    """
    训练Transformer模型
    
    Args:
        sequences: 输入序列 [N, seq_len, input_dim]
        labels: 标签 [N, 1]
        device: 设备
        
    Returns:
        训练好的模型
    """
    logger.info("开始训练Transformer模型...")
    
    # 1. 创建数据加载器
    train_loader, val_loader = create_dataloaders(
        sequences, labels,
        train_ratio=0.8,
        batch_size=64,
        shuffle=True
    )
    
    # 2. 创建模型
    transformer = TransformerStateModel(
        input_dim=155,  # 128 + 27
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    model = TransformerWithAuxiliaryTasks(
        transformer_model=transformer,
        d_model=256,
        num_classes=3
    )
    
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 创建优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )
    
    loss_fn = MultiTaskLoss(
        lambda_reg=0.1,
        lambda_cls=0.05
    )
    
    # 4. 创建训练器
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        max_grad_norm=0.5,
        log_interval=50
    )
    
    # 5. 训练
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        save_dir='models/checkpoints',
        early_stopping_patience=10
    )
    
    logger.info("训练完成!")
    
    return model, history


def evaluate_transformer_model(model: TransformerWithAuxiliaryTasks,
                               sequences: torch.Tensor,
                               labels: torch.Tensor,
                               device: str = 'cpu') -> dict:
    """
    评估Transformer模型
    
    Args:
        model: 训练好的模型
        sequences: 测试序列
        labels: 测试标签
        device: 设备
        
    Returns:
        评估指标
    """
    logger.info("评估模型性能...")
    
    # 创建测试数据加载器
    from torch.utils.data import DataLoader, TensorDataset
    test_dataset = TensorDataset(sequences, labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 1. 监督学习指标
    supervised_metrics = SupervisedMetrics.evaluate_model(
        model, test_loader, device
    )
    
    logger.info("监督学习指标:")
    for key, value in supervised_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # 2. 状态表征质量
    state_metrics = StateRepresentationQuality.evaluate_states(
        model, test_loader, device
    )
    
    logger.info("状态表征质量:")
    for key, value in state_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return {
        **supervised_metrics,
        **state_metrics
    }


def visualize_results(model: TransformerWithAuxiliaryTasks,
                     sequences: torch.Tensor,
                     labels: torch.Tensor,
                     history: dict,
                     device: str = 'cpu') -> None:
    """
    可视化训练结果
    
    Args:
        model: 训练好的模型
        sequences: 测试序列
        labels: 测试标签
        history: 训练历史
        device: 设备
    """
    logger.info("生成可视化...")
    
    # 创建输出目录
    output_dir = Path('outputs/transformer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train']['loss'], label='Train')
    axes[0, 0].plot(history['val']['loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 回归损失
    axes[0, 1].plot(history['train']['reg_loss'], label='Train')
    axes[0, 1].plot(history['val']['reg_loss'], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Regression Loss')
    axes[0, 1].set_title('Regression Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 分类损失
    axes[1, 0].plot(history['train']['cls_loss'], label='Train')
    axes[1, 0].plot(history['val']['cls_loss'], label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Classification Loss')
    axes[1, 0].set_title('Classification Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 方向准确率
    train_dir_acc = [m['direction_accuracy'] for m in history['train']['metrics']]
    val_dir_acc = [m['direction_accuracy'] for m in history['val']['metrics']]
    axes[1, 1].plot(train_dir_acc, label='Train')
    axes[1, 1].plot(val_dir_acc, label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Direction Accuracy')
    axes[1, 1].set_title('Direction Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    logger.info(f"训练曲线已保存: {output_dir / 'training_curves.png'}")
    plt.close()
    
    # 2. t-SNE可视化状态向量
    model.eval()
    with torch.no_grad():
        # 采样部分数据
        sample_size = min(1000, len(sequences))
        sample_sequences = sequences[:sample_size].to(device)
        sample_labels = labels[:sample_size].to(device)
        
        # 获取状态向量
        results = model.predict(sample_sequences)
        states = results['state'].cpu().numpy()
        
        # 创建分类标签
        from src.models.transformer.auxiliary_tasks import create_labels_from_returns
        cls_labels = create_labels_from_returns(sample_labels.squeeze()).cpu().numpy()
        
        # t-SNE可视化
        AttentionVisualizer.plot_tsne_visualization(
            states, cls_labels,
            save_path=output_dir / 'state_tsne.png'
        )
    
    logger.info("可视化完成!")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Transformer状态建模器训练演示")
    logger.info("=" * 80)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 1. 生成数据
    logger.info("\n步骤1: 生成合成数据")
    ohlc_df = generate_synthetic_data(n_samples=10000)
    
    # 2. 加载或创建TS2Vec模型
    logger.info("\n步骤2: 准备TS2Vec模型")
    ts2vec_model = TS2VecModel(
        input_dim=4,
        hidden_dim=64,
        output_dim=128,
        num_layers=10
    )
    # 注意: 实际使用时应加载预训练权重
    # ts2vec_model.load('models/checkpoints/best_ts2vec.pt')
    ts2vec_model.to(device)
    ts2vec_model.eval()
    
    # 3. 准备Transformer数据
    logger.info("\n步骤3: 准备Transformer训练数据")
    sequences, labels = prepare_transformer_data(ohlc_df, ts2vec_model, device)
    
    # 4. 划分数据集
    n_samples = len(sequences)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    test_sequences = sequences[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    
    logger.info(f"数据集划分: train={len(train_sequences)}, "
                f"val={len(val_sequences)}, test={len(test_sequences)}")
    
    # 5. 训练模型
    logger.info("\n步骤4: 训练Transformer模型")
    train_data = torch.cat([train_sequences, val_sequences], dim=0)
    train_labels_all = torch.cat([train_labels, val_labels], dim=0)
    model, history = train_transformer_model(train_data, train_labels_all, device)
    
    # 6. 评估模型
    logger.info("\n步骤5: 评估模型性能")
    metrics = evaluate_transformer_model(model, test_sequences, test_labels, device)
    
    # 7. 可视化结果
    logger.info("\n步骤6: 生成可视化")
    visualize_results(model, test_sequences, test_labels, history, device)
    
    # 8. 总结
    logger.info("\n" + "=" * 80)
    logger.info("训练完成!")
    logger.info("=" * 80)
    logger.info("\n关键指标:")
    logger.info(f"  回归R²: {metrics.get('reg_r2', 0):.4f}")
    logger.info(f"  方向准确率: {metrics.get('reg_direction_accuracy', 0):.4f}")
    logger.info(f"  分类准确率: {metrics.get('cls_accuracy', 0):.4f}")
    logger.info(f"  状态可分离性: {metrics.get('separability_ratio', 0):.4f}")
    
    logger.info("\n输出文件:")
    logger.info("  - models/checkpoints/best_transformer.pt (最佳模型)")
    logger.info("  - outputs/transformer/training_curves.png (训练曲线)")
    logger.info("  - outputs/transformer/state_tsne.png (状态向量可视化)")
    
    logger.info("\n下一步:")
    logger.info("  1. 使用真实数据训练")
    logger.info("  2. 调整超参数优化性能")
    logger.info("  3. 进行PPO强化学习训练")


if __name__ == '__main__':
    main()