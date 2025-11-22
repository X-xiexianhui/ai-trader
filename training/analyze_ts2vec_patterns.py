"""
分析TS2Vec学到的形态
通过聚类和可视化来理解模型自动发现的时间序列模式
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import yaml
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ts2vec.model import DilatedConvEncoder
from src.models.ts2vec.data_preparation import TS2VecDataset, OptimizedDataLoader
from src.utils.logger import setup_logger


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """加载训练好的TS2Vec模型"""
    model = DilatedConvEncoder(
        input_dims=config['model']['input_dims'],
        output_dims=config['model']['output_dims'],
        hidden_dims=config['model']['hidden_dims'],
        depth=config['model']['depth']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def extract_embeddings(model, dataloader, device):
    """提取所有样本的embedding"""
    embeddings = []
    timestamps = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x'].to(device)
            timestamp = batch['timestamp']
            
            # 获取embedding (取时间维度的平均)
            emb = model(x)  # [batch, time, dim]
            emb = emb.mean(dim=1)  # [batch, dim]
            
            embeddings.append(emb.cpu().numpy())
            timestamps.extend(timestamp)
    
    embeddings = np.vstack(embeddings)
    return embeddings, timestamps


def find_optimal_clusters(embeddings, max_k=20):
    """使用肘部法则和轮廓系数找到最优聚类数"""
    inertias = []
    silhouette_scores = []
    db_scores = []
    
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, labels))
        db_scores.append(davies_bouldin_score(embeddings, labels))
    
    # 绘制评估指标
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 肘部法则
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('聚类数 K')
    axes[0].set_ylabel('簇内平方和 (Inertia)')
    axes[0].set_title('肘部法则')
    axes[0].grid(True)
    
    # 轮廓系数 (越高越好)
    axes[1].plot(K_range, silhouette_scores, 'go-')
    axes[1].set_xlabel('聚类数 K')
    axes[1].set_ylabel('轮廓系数')
    axes[1].set_title('轮廓系数 (越高越好)')
    axes[1].grid(True)
    
    # Davies-Bouldin指数 (越低越好)
    axes[2].plot(K_range, db_scores, 'ro-')
    axes[2].set_xlabel('聚类数 K')
    axes[2].set_ylabel('Davies-Bouldin指数')
    axes[2].set_title('Davies-Bouldin指数 (越低越好)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training/output/cluster_evaluation.png', dpi=300, bbox_inches='tight')
    print("已保存聚类评估图: training/output/cluster_evaluation.png")
    
    # 推荐最优K
    best_k_silhouette = K_range[np.argmax(silhouette_scores)]
    best_k_db = K_range[np.argmin(db_scores)]
    
    print(f"\n最优聚类数建议:")
    print(f"  基于轮廓系数: K = {best_k_silhouette} (score: {max(silhouette_scores):.3f})")
    print(f"  基于DB指数: K = {best_k_db} (score: {min(db_scores):.3f})")
    
    return best_k_silhouette


def cluster_embeddings(embeddings, n_clusters):
    """对embeddings进行聚类"""
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    
    # DBSCAN聚类 (基于密度)
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    dbscan_labels = dbscan.fit_predict(embeddings)
    
    print(f"\nKMeans聚类结果:")
    print(f"  聚类数: {n_clusters}")
    for i in range(n_clusters):
        count = np.sum(kmeans_labels == i)
        print(f"  簇 {i}: {count} 个样本 ({count/len(kmeans_labels)*100:.1f}%)")
    
    print(f"\nDBSCAN聚类结果:")
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = np.sum(dbscan_labels == -1)
    print(f"  发现簇数: {n_dbscan_clusters}")
    print(f"  噪声点: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")
    
    return kmeans_labels, dbscan_labels, kmeans


def visualize_clusters(embeddings, labels, method_name, save_path):
    """可视化聚类结果"""
    # t-SNE降维
    print(f"\n执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘制
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # 噪声点用黑色
            color = 'black'
            marker = 'x'
            label_name = '噪声'
        else:
            marker = 'o'
            label_name = f'簇 {label}'
        
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=label_name,
            alpha=0.6,
            s=50,
            marker=marker
        )
    
    plt.xlabel('t-SNE维度 1', fontsize=12)
    plt.ylabel('t-SNE维度 2', fontsize=12)
    plt.title(f'TS2Vec Embedding聚类可视化 ({method_name})', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存聚类可视化: {save_path}")
    
    return embeddings_2d


def analyze_cluster_patterns(df, labels, kmeans, n_clusters):
    """分析每个簇的时间序列特征"""
    print("\n分析各簇的时间序列特征...")
    
    cluster_stats = []
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_data = df[mask]
        
        if len(cluster_data) == 0:
            continue
        
        # 计算统计特征
        stats = {
            'cluster_id': cluster_id,
            'count': len(cluster_data),
            'avg_return': cluster_data['close'].pct_change().mean() * 100,
            'volatility': cluster_data['close'].pct_change().std() * 100,
            'avg_volume': cluster_data['volume'].mean(),
            'avg_range': ((cluster_data['high'] - cluster_data['low']) / cluster_data['close']).mean() * 100,
        }
        
        cluster_stats.append(stats)
    
    # 转换为DataFrame
    stats_df = pd.DataFrame(cluster_stats)
    stats_df = stats_df.sort_values('count', ascending=False)
    
    print("\n各簇统计特征:")
    print(stats_df.to_string(index=False))
    
    # 保存
    stats_df.to_csv('training/output/cluster_statistics.csv', index=False)
    print("\n已保存簇统计: training/output/cluster_statistics.csv")
    
    return stats_df


def visualize_cluster_examples(df, labels, n_clusters, samples_per_cluster=5):
    """可视化每个簇的代表性样本"""
    fig, axes = plt.subplots(n_clusters, samples_per_cluster, 
                            figsize=(20, 4*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_indices = np.where(mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # 随机选择样本
        sample_indices = np.random.choice(
            cluster_indices, 
            min(samples_per_cluster, len(cluster_indices)),
            replace=False
        )
        
        for i, idx in enumerate(sample_indices):
            ax = axes[cluster_id, i]
            
            # 获取窗口数据
            window_size = 256
            start_idx = max(0, idx - window_size)
            end_idx = idx
            
            window_data = df.iloc[start_idx:end_idx]
            
            # 绘制K线图
            ax.plot(window_data.index, window_data['close'], 'b-', linewidth=1)
            ax.fill_between(window_data.index, 
                           window_data['low'], 
                           window_data['high'], 
                           alpha=0.3)
            
            ax.set_title(f'簇 {cluster_id} - 样本 {i+1}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if i == 0:
                ax.set_ylabel('价格', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training/output/cluster_examples.png', dpi=300, bbox_inches='tight')
    print("\n已保存簇样本可视化: training/output/cluster_examples.png")


def compare_with_manual_patterns(df, labels, pattern_detector):
    """比较自动聚类与手动形态检测的结果"""
    from src.features.pattern_detector import PatternDetector
    
    print("\n比较自动聚类与手动形态检测...")
    
    # 检测手动形态
    detector = PatternDetector()
    patterns = detector.detect_patterns(df, window_size=50)
    
    # 创建形态标签
    pattern_labels = ['none'] * len(df)
    for pattern in patterns:
        for idx in range(pattern.start_idx, min(pattern.end_idx + 1, len(df))):
            if pattern.confidence > 0.6:  # 只考虑高置信度形态
                pattern_labels[idx] = pattern.pattern_type.value
    
    # 分析每个簇对应的主要形态
    cluster_pattern_map = {}
    
    for cluster_id in range(len(set(labels))):
        mask = labels == cluster_id
        cluster_patterns = [pattern_labels[i] for i in range(len(labels)) if mask[i]]
        
        # 统计形态分布
        pattern_counts = pd.Series(cluster_patterns).value_counts()
        cluster_pattern_map[cluster_id] = pattern_counts.to_dict()
    
    print("\n各簇对应的主要形态:")
    for cluster_id, pattern_dist in cluster_pattern_map.items():
        print(f"\n簇 {cluster_id}:")
        for pattern, count in sorted(pattern_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = count / sum(pattern_dist.values()) * 100
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    return cluster_pattern_map


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name='pattern_analysis',
        log_file='logs/pattern_analysis.log',
        log_level='INFO'
    )
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    Path('training/output').mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据...")
    df = pd.read_csv('data/processed/MES_test.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 
                            'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    # 创建数据集
    dataset = TS2VecDataset(
        df=df,
        window_length=config['data']['window_length'],
        stride=config['data']['stride']
    )
    
    dataloader = OptimizedDataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # 加载模型
    logger.info("加载训练好的模型...")
    checkpoint_path = 'models/checkpoints/ts2vec/best_model.pt'
    model = load_model(checkpoint_path, config, device)
    
    # 提取embeddings
    logger.info("提取embeddings...")
    embeddings, timestamps = extract_embeddings(model, dataloader, device)
    logger.info(f"提取了 {len(embeddings)} 个embedding，维度: {embeddings.shape[1]}")
    
    # 找到最优聚类数
    logger.info("\n寻找最优聚类数...")
    optimal_k = find_optimal_clusters(embeddings, max_k=20)
    
    # 执行聚类
    logger.info(f"\n使用K={optimal_k}进行聚类...")
    kmeans_labels, dbscan_labels, kmeans_model = cluster_embeddings(embeddings, optimal_k)
    
    # 可视化KMeans聚类
    logger.info("\n可视化KMeans聚类结果...")
    embeddings_2d = visualize_clusters(
        embeddings, 
        kmeans_labels, 
        'KMeans',
        'training/output/kmeans_clusters.png'
    )
    
    # 可视化DBSCAN聚类
    logger.info("\n可视化DBSCAN聚类结果...")
    visualize_clusters(
        embeddings,
        dbscan_labels,
        'DBSCAN',
        'training/output/dbscan_clusters.png'
    )
    
    # 分析簇特征
    cluster_stats = analyze_cluster_patterns(df, kmeans_labels, kmeans_model, optimal_k)
    
    # 可视化簇样本
    logger.info("\n可视化各簇代表性样本...")
    visualize_cluster_examples(df, kmeans_labels, optimal_k, samples_per_cluster=5)
    
    # 与手动形态检测对比
    logger.info("\n与手动形态检测对比...")
    try:
        from src.features.pattern_detector import PatternDetector
        detector = PatternDetector()
        cluster_pattern_map = compare_with_manual_patterns(df, kmeans_labels, detector)
    except Exception as e:
        logger.warning(f"手动形态检测失败: {e}")
    
    logger.info("\n分析完成！")
    logger.info("生成的文件:")
    logger.info("  - training/output/cluster_evaluation.png: 聚类评估指标")
    logger.info("  - training/output/kmeans_clusters.png: KMeans聚类可视化")
    logger.info("  - training/output/dbscan_clusters.png: DBSCAN聚类可视化")
    logger.info("  - training/output/cluster_statistics.csv: 簇统计特征")
    logger.info("  - training/output/cluster_examples.png: 各簇代表性样本")


if __name__ == '__main__':
    main()