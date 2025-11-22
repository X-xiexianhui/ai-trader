"""
评估TS2Vec模型

加载训练好的TS2Vec模型，进行全面的性能评估
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
import logging
import json
from datetime import datetime

from src.models.ts2vec import TS2VecModel, TS2VecEvaluator
from src.models.ts2vec.data_preparation import TS2VecDataset
from src.models.ts2vec.training import OptimizedDataLoader
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    name='ts2vec_evaluation',
    log_level='INFO',
    log_dir='logs',
    log_file='ts2vec_evaluation.log'
)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str, config: dict, device: str) -> TS2VecModel:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        config: 配置字典
        device: 计算设备
        
    Returns:
        加载的模型
    """
    logger.info(f"加载模型: {model_path}")
    
    ts2vec_config = config['ts2vec']
    
    model = TS2VecModel.load(
        model_path,
        device=device,
        input_dim=ts2vec_config['input_dim'],
        hidden_dim=ts2vec_config['hidden_dim'],
        output_dim=ts2vec_config['hidden_dim'] // 2,
        num_layers=ts2vec_config['num_layers'],
        kernel_size=ts2vec_config['kernel_size'],
        dilation_rates=ts2vec_config.get('dilation_rates'),
        temperature=ts2vec_config['temperature']
    )
    
    logger.info("模型加载成功")
    return model


def load_data(data_path: str) -> pd.DataFrame:
    """加载数据"""
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # 重命名列为小写
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"数据列: {df.columns.tolist()}")
    return df


def prepare_evaluation_data(test_df: pd.DataFrame, config: dict):
    """
    准备评估数据
    
    Args:
        test_df: 测试数据
        config: 配置字典
        
    Returns:
        test_dataset
    """
    logger.info("准备评估数据...")
    
    # 提取OHLC数据
    ohlc_columns = ['open', 'high', 'low', 'close']
    test_ohlc = test_df[ohlc_columns].values
    
    # 检查数据质量
    if np.isnan(test_ohlc).any():
        logger.warning("测试集中存在NaN值，将进行填充")
        test_ohlc = pd.DataFrame(test_ohlc).fillna(method='ffill').fillna(method='bfill').values
    
    # 创建测试数据集
    ts2vec_config = config['ts2vec']
    
    test_dataset = TS2VecDataset(
        data=test_ohlc,
        window_length=ts2vec_config['window_length'],
        stride=1,
        augmentation_params={
            'aug_types': ['masking', 'warping', 'scaling'],
            'masking_ratio': ts2vec_config['masking_ratio'],
            'warp_ratio': ts2vec_config['time_warp_ratio'],
            'scale_range': ts2vec_config['magnitude_scale_range']
        }
    )
    
    logger.info(f"测试集样本数: {len(test_dataset)}")
    logger.info(f"测试集时间范围: {test_df.index[0]} 到 {test_df.index[-1]}")
    
    return test_dataset


def create_labels_for_probing(df: pd.DataFrame, window_length: int, stride: int) -> np.ndarray:
    """
    为线性探测创建标签（涨跌方向）
    
    Args:
        df: 原始数据
        window_length: 窗口长度
        stride: 步长
        
    Returns:
        标签数组
    """
    logger.info("创建线性探测标签...")
    
    # 计算未来收益率（窗口结束后的涨跌）
    returns = df['close'].pct_change().shift(-1)
    
    # 生成标签（1=上涨，0=下跌）
    labels = []
    num_windows = (len(df) - window_length) // stride + 1
    
    for i in range(num_windows):
        end_idx = i * stride + window_length
        if end_idx < len(returns):
            # 使用窗口结束后的收益率
            future_return = returns.iloc[end_idx]
            label = 1 if future_return > 0 else 0
            labels.append(label)
        else:
            break
    
    labels = np.array(labels)
    logger.info(f"标签分布: 上涨={np.sum(labels==1)}, 下跌={np.sum(labels==0)}")
    
    return labels


def evaluate_model_comprehensive(model: TS2VecModel,
                                 test_dataset,
                                 labels: np.ndarray,
                                 config: dict,
                                 device: str,
                                 output_dir: str = 'training/output') -> dict:
    """
    全面评估模型
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        labels: 标签数组
        config: 配置字典
        device: 计算设备
        output_dir: 输出目录
        
    Returns:
        评估结果
    """
    logger.info("=" * 80)
    logger.info("开始全面评估TS2Vec模型")
    logger.info("=" * 80)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建评估器
    evaluator = TS2VecEvaluator(model, device=device)
    
    # 创建测试数据加载器
    test_loader = OptimizedDataLoader.create_loader(
        test_dataset,
        batch_size=config['ts2vec']['batch_size'],
        shuffle=False,
        device=device
    )
    
    # 1. 评估Embedding质量
    logger.info("\n1. 评估Embedding质量...")
    embedding_quality = evaluator.evaluate_embedding_quality(
        test_loader,
        target_pos_sim=0.8,
        target_neg_sim=0.3
    )
    
    # 2. 线性探测评估
    logger.info("\n2. 线性探测评估...")
    
    # 准备线性探测数据（从测试集中划分）
    n_test = len(test_dataset)
    n_labels = len(labels)
    
    # 确保标签数量匹配
    if n_test != n_labels:
        logger.warning(f"数据集样本数({n_test})与标签数({n_labels})不匹配，调整标签")
        # 截取或填充标签以匹配数据集大小
        if n_labels > n_test:
            labels = labels[:n_test]
        else:
            # 如果标签不够，使用最后一个标签填充
            labels = np.pad(labels, (0, n_test - n_labels), mode='edge')
        n_labels = len(labels)
    
    n_train_probe = int(0.7 * n_test)
    
    # 获取数据
    train_probe_data = []
    test_probe_data = []
    
    for idx in range(n_train_probe):
        x_i, _ = test_dataset[idx]
        train_probe_data.append(x_i)
    
    for idx in range(n_train_probe, n_test):
        x_i, _ = test_dataset[idx]
        test_probe_data.append(x_i)
    
    train_probe_data = torch.stack(train_probe_data)
    test_probe_data = torch.stack(test_probe_data)
    
    train_probe_labels = torch.tensor(labels[:n_train_probe])
    test_probe_labels = torch.tensor(labels[n_train_probe:n_test])
    
    logger.info(f"线性探测 - 训练样本: {len(train_probe_data)}, 测试样本: {len(test_probe_data)}")
    
    linear_probing_results = evaluator.linear_probing(
        train_probe_data,
        train_probe_labels,
        test_probe_data,
        test_probe_labels,
        target_accuracy=0.55
    )
    
    # 3. 聚类质量评估
    logger.info("\n3. 聚类质量评估...")
    
    # 采样一部分数据进行聚类
    sample_size = min(1000, len(test_dataset))
    sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)
    
    sample_data = []
    for idx in sample_indices:
        x_i, _ = test_dataset[idx]
        sample_data.append(x_i)
    
    sample_data = torch.stack(sample_data)
    
    clustering_results = evaluator.clustering_quality(
        sample_data,
        n_clusters=5,
        target_silhouette=0.3
    )
    
    # 4. t-SNE可视化
    logger.info("\n4. 生成t-SNE可视化...")
    
    # 使用聚类的标签进行着色
    tsne_labels = labels[sample_indices]
    
    evaluator.tsne_visualization(
        sample_data,
        labels=tsne_labels,
        save_path=os.path.join(output_dir, 'ts2vec_tsne.png'),
        n_samples=min(1000, len(sample_data))
    )
    
    # 5. 生成评估报告
    logger.info("\n5. 生成评估报告...")
    
    report_path = os.path.join(output_dir, 'ts2vec_evaluation_report.txt')
    all_results = evaluator.generate_evaluation_report(report_path)
    
    # 6. 额外的可视化：相似度分布
    logger.info("\n6. 生成相似度分布图...")
    plot_similarity_distribution(
        evaluator,
        test_loader,
        device,
        save_path=os.path.join(output_dir, 'ts2vec_similarity_distribution.png')
    )
    
    # 7. Embedding维度分析
    logger.info("\n7. Embedding维度分析...")
    embedding_stats = analyze_embedding_dimensions(
        model,
        sample_data,
        device,
        save_path=os.path.join(output_dir, 'ts2vec_embedding_stats.png')
    )
    
    all_results['embedding_stats'] = embedding_stats
    
    return all_results


def plot_similarity_distribution(evaluator, dataloader, device, save_path: str):
    """绘制正负样本相似度分布"""
    import matplotlib.pyplot as plt
    
    pos_similarities = []
    neg_similarities = []
    
    with torch.no_grad():
        for batch_idx, (x_i, x_j) in enumerate(dataloader):
            if batch_idx >= 50:
                break
            
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            z_i, z_j = evaluator.model(x_i, x_j, return_loss=False)
            
            # 正样本相似度
            pos_sim = torch.cosine_similarity(
                z_i.reshape(-1, z_i.shape[-1]),
                z_j.reshape(-1, z_j.shape[-1]),
                dim=1
            )
            pos_similarities.extend(pos_sim.cpu().numpy())
            
            # 负样本相似度
            batch_size = z_i.shape[0]
            if batch_size > 1:
                perm = torch.randperm(batch_size)
                z_j_shuffled = z_j[perm]
                
                neg_sim = torch.cosine_similarity(
                    z_i.reshape(-1, z_i.shape[-1]),
                    z_j_shuffled.reshape(-1, z_j_shuffled.shape[-1]),
                    dim=1
                )
                neg_similarities.extend(neg_sim.cpu().numpy())
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.hist(pos_similarities, bins=50, alpha=0.6, label='Positive Pairs', color='green')
    plt.hist(neg_similarities, bins=50, alpha=0.6, label='Negative Pairs', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution of Positive and Negative Pairs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"相似度分布图已保存: {save_path}")


def analyze_embedding_dimensions(model, data, device, save_path: str) -> dict:
    """分析embedding各维度的统计特性"""
    import matplotlib.pyplot as plt
    
    with torch.no_grad():
        data = data.to(device)
        embeddings = model.encode(data, return_projection=True)
        embeddings = embeddings.mean(dim=1).cpu().numpy()
    
    # 计算统计量
    mean_per_dim = embeddings.mean(axis=0)
    std_per_dim = embeddings.std(axis=0)
    
    stats = {
        'mean_activation': float(np.mean(np.abs(mean_per_dim))),
        'std_activation': float(np.mean(std_per_dim)),
        'min_std': float(np.min(std_per_dim)),
        'max_std': float(np.max(std_per_dim))
    }
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 均值分布
    axes[0, 0].bar(range(len(mean_per_dim)), mean_per_dim)
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].set_title('Mean Activation per Dimension')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 标准差分布
    axes[0, 1].bar(range(len(std_per_dim)), std_per_dim)
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].set_title('Standard Deviation per Dimension')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Embedding范数分布
    norms = np.linalg.norm(embeddings, axis=1)
    axes[1, 0].hist(norms, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel('L2 Norm')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Embedding Norm Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 相关性矩阵（采样部分维度）
    sample_dims = min(20, embeddings.shape[1])
    corr_matrix = np.corrcoef(embeddings[:, :sample_dims].T)
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('Dimension')
    axes[1, 1].set_title(f'Correlation Matrix (first {sample_dims} dims)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Embedding统计图已保存: {save_path}")
    logger.info(f"  平均激活: {stats['mean_activation']:.4f}")
    logger.info(f"  平均标准差: {stats['std_activation']:.4f}")
    
    return stats


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("TS2Vec模型评估")
    logger.info("=" * 80)
    
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model_path = 'models/checkpoints/ts2vec/ts2vec_final.pth'
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行 training/train_ts2vec.py 训练模型")
        return
    
    model = load_model(model_path, config, device)
    
    # 加载预划分的测试集
    data_dir = config['data']['processed_data_dir']
    test_path = os.path.join(data_dir, 'MES_test.csv')
    
    if not os.path.exists(test_path):
        logger.error("未找到测试集文件!")
        logger.error("请先运行: python training/split_dataset.py")
        return
    
    test_df = load_data(test_path)
    
    # 准备评估数据
    test_dataset = prepare_evaluation_data(test_df, config)
    
    # 创建标签
    labels = create_labels_for_probing(
        test_df,
        config['ts2vec']['window_length'],
        stride=1
    )
    
    # 全面评估
    start_time = datetime.now()
    results = evaluate_model_comprehensive(
        model,
        test_dataset,
        labels,
        config,
        device
    )
    end_time = datetime.now()
    
    eval_time = (end_time - start_time).total_seconds()
    logger.info(f"\n评估耗时: {eval_time:.2f} 秒")
    
    # 保存完整结果为JSON（转换numpy类型）
    def convert_to_serializable(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_serializable = convert_to_serializable(results)
    
    results_json_path = 'training/output/ts2vec_evaluation_results.json'
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f"评估结果已保存: {results_json_path}")
    
    # 打印总结
    logger.info("\n" + "=" * 80)
    logger.info("评估总结")
    logger.info("=" * 80)
    
    if 'embedding_quality' in results:
        eq = results['embedding_quality']
        logger.info(f"Embedding质量: {'✓ 通过' if eq['meets_target'] else '✗ 未通过'}")
        logger.info(f"  正样本相似度: {eq['pos_sim_mean']:.4f}")
        logger.info(f"  负样本相似度: {eq['neg_sim_mean']:.4f}")
    
    if 'linear_probing' in results:
        lp = results['linear_probing']
        logger.info(f"线性探测: {'✓ 通过' if lp['meets_target'] else '✗ 未通过'}")
        logger.info(f"  测试准确率: {lp['test_accuracy']:.4f}")
    
    if 'clustering' in results:
        cl = results['clustering']
        logger.info(f"聚类质量: {'✓ 通过' if cl['meets_target'] else '✗ 未通过'}")
        logger.info(f"  轮廓系数: {cl['silhouette_score']:.4f}")
    
    logger.info("=" * 80)
    logger.info("评估完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()