"""
TS2Vec模型评估脚本

功能：
1. 加载训练好的TS2Vec模型
2. 评估embedding质量
3. 执行线性探测测试
4. 评估聚类质量
5. 生成t-SNE可视化
6. 生成完整评估报告

使用方法:
    python training/evaluate_ts2vec.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.models.ts2vec.model import TS2VecModel
from src.models.ts2vec.data_preparation import TS2VecDataset
from src.models.ts2vec.evaluation import TS2VecEvaluator
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('evaluate_ts2vec', log_dir='training/output', log_file='evaluate_ts2vec.log')


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """加载配置文件"""
    logger.info(f"加载配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(model_path: str, device: str) -> TS2VecModel:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        加载的模型
    """
    logger.info(f"加载训练好的模型: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从检查点中获取配置
    config = checkpoint.get('config', {})
    
    # 创建模型
    model = TS2VecModel(
        input_dim=4,
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 128),
        num_layers=10,
        kernel_size=3,
        temperature=0.1
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("模型加载完成")
    
    return model


def prepare_evaluation_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    准备评估数据
    
    Args:
        df: 处理后的DataFrame
        config: 配置字典
        
    Returns:
        (val_loader, test_data, test_labels)
    """
    logger.info("准备评估数据...")
    
    # 提取OHLC数据
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    ohlc_data = df[ohlc_columns].values
    
    ts2vec_config = config['ts2vec']
    window_length = ts2vec_config['window_length']
    
    # 创建数据集（用于embedding质量评估）
    dataset = TS2VecDataset(
        data=ohlc_data,
        window_length=window_length,
        stride=1,
        augmentation_params={
            'aug_types': ['masking', 'warping', 'scaling'],
            'masking_ratio': 0.2,
            'warp_ratio': 0.05,
            'scale_range': (0.9, 1.1)
        }
    )
    
    # 创建DataLoader
    val_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    # 准备线性探测数据
    # 创建标签：未来5周期收益方向（涨=1，跌=0）
    future_returns = df['Close'].pct_change(5).shift(-5)
    labels = (future_returns > 0).astype(int).values
    
    # 对齐窗口和标签
    num_windows = len(dataset.windows)
    labels = labels[window_length-1:window_length-1+num_windows]
    
    # 移除NaN
    valid_mask = ~np.isnan(labels)
    windows = dataset.windows[valid_mask]
    labels = labels[valid_mask]
    
    # 划分训练/测试集（用于线性探测）
    split_idx = int(len(windows) * 0.8)
    
    train_data = torch.FloatTensor(windows[:split_idx])
    train_labels = torch.LongTensor(labels[:split_idx])
    test_data = torch.FloatTensor(windows[split_idx:])
    test_labels = torch.LongTensor(labels[split_idx:])
    
    logger.info(f"评估数据准备完成:")
    logger.info(f"  验证集批次: {len(val_loader)}")
    logger.info(f"  线性探测训练集: {len(train_data)}")
    logger.info(f"  线性探测测试集: {len(test_data)}")
    
    return val_loader, train_data, train_labels, test_data, test_labels


def main():
    """主函数"""
    try:
        # 检测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用计算设备: {device}")
        
        # 1. 加载配置
        config = load_config('configs/config.yaml')
        
        # 2. 加载训练好的模型
        model_path = 'models/checkpoints/ts2vec/best_model.pt'
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            logger.error("请先运行 train_ts2vec.py 训练模型")
            sys.exit(1)
        
        model = load_trained_model(model_path, device)
        
        # 3. 加载数据
        data_path = 'training/output/mes_features_normalized.parquet'
        if not Path(data_path).exists():
            data_path = 'training/output/mes_features_normalized.csv'
        
        logger.info(f"加载数据: {data_path}")
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # 4. 准备评估数据
        val_loader, train_data, train_labels, test_data, test_labels = prepare_evaluation_data(df, config)
        
        # 5. 创建评估器
        logger.info("=" * 80)
        logger.info("开始TS2Vec模型评估...")
        logger.info("=" * 80)
        
        evaluator = TS2VecEvaluator(model, device=device)
        
        # 6. 评估embedding质量
        logger.info("\n1. 评估embedding质量...")
        logger.info("-" * 80)
        embedding_quality = evaluator.evaluate_embedding_quality(
            val_loader,
            target_pos_sim=0.7,
            target_neg_sim=0.3
        )
        
        # 7. 线性探测评估
        logger.info("\n2. 线性探测评估...")
        logger.info("-" * 80)
        linear_probing = evaluator.linear_probing(
            train_data,
            train_labels,
            test_data,
            test_labels,
            target_accuracy=0.52  # 略高于随机猜测(0.5)
        )
        
        # 8. 聚类质量评估
        logger.info("\n3. 聚类质量评估...")
        logger.info("-" * 80)
        # 使用测试数据的一部分进行聚类
        sample_size = min(1000, len(test_data))
        sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
        sample_data = test_data[sample_indices]
        
        clustering = evaluator.clustering_quality(
            sample_data,
            n_clusters=5,
            target_silhouette=0.2
        )
        
        # 9. t-SNE可视化
        logger.info("\n4. 生成t-SNE可视化...")
        logger.info("-" * 80)
        evaluator.tsne_visualization(
            sample_data,
            labels=test_labels[sample_indices].numpy(),
            save_path='training/output/ts2vec_tsne.png',
            n_samples=500
        )
        
        # 10. 生成评估报告
        logger.info("\n5. 生成评估报告...")
        logger.info("-" * 80)
        report_path = 'training/output/ts2vec_evaluation_report.txt'
        evaluation_results = evaluator.generate_evaluation_report(report_path)
        
        # 11. 打印摘要
        logger.info("\n" + "=" * 80)
        logger.info("评估完成！")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("TS2Vec模型评估摘要")
        print("=" * 80)
        
        print("\n1. Embedding质量:")
        print(f"   正样本相似度: {embedding_quality['pos_sim_mean']:.4f} ± {embedding_quality['pos_sim_std']:.4f}")
        print(f"   负样本相似度: {embedding_quality['neg_sim_mean']:.4f} ± {embedding_quality['neg_sim_std']:.4f}")
        print(f"   分离度: {embedding_quality['separation']:.4f}")
        print(f"   达到目标: {'✓' if embedding_quality['meets_target'] else '✗'}")
        
        print("\n2. 线性探测:")
        print(f"   训练准确率: {linear_probing['train_accuracy']:.4f}")
        print(f"   测试准确率: {linear_probing['test_accuracy']:.4f}")
        if linear_probing['test_auc']:
            print(f"   测试AUC: {linear_probing['test_auc']:.4f}")
        print(f"   达到目标: {'✓' if linear_probing['meets_target'] else '✗'}")
        
        print("\n3. 聚类质量:")
        print(f"   轮廓系数: {clustering['silhouette_score']:.4f}")
        print(f"   簇大小: {clustering['cluster_sizes']}")
        print(f"   达到目标: {'✓' if clustering['meets_target'] else '✗'}")
        
        print("\n输出文件:")
        print(f"   评估报告: {report_path}")
        print(f"   t-SNE可视化: training/output/ts2vec_tsne.png")
        print(f"   日志文件: training/output/evaluate_ts2vec.log")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"\n✗ 评估失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()