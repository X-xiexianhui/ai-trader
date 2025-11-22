"""
训练TS2Vec模型

从data/processed目录读取清洗后的MES数据，训练TS2Vec模型
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
from torch.utils.data import DataLoader, random_split
import yaml
import logging
from datetime import datetime

from src.models.ts2vec.model import TS2VecModel
from src.models.ts2vec.training import TS2VecTrainer, OptimizedDataLoader
from src.models.ts2vec.data_preparation import TS2VecDataset
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    name='ts2vec_training',
    log_level='INFO',
    log_dir='logs',
    log_file='ts2vec_training.log'
)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> pd.DataFrame:
    """
    加载数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        DataFrame
    """
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # 设置时间戳为索引
    df.set_index('date', inplace=True)
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"数据列: {df.columns.tolist()}")
    logger.info(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    return df


def prepare_ts2vec_data(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict) -> tuple:
    """
    准备TS2Vec训练数据
    
    Args:
        train_df: 训练数据
        val_df: 验证数据
        config: 配置字典
        
    Returns:
        (train_dataset, val_dataset)
    """
    logger.info("准备TS2Vec训练数据...")
    
    ts2vec_config = config['ts2vec']
    
    # 处理训练集
    logger.info("处理训练集...")
    train_ohlc = train_df[['open', 'high', 'low', 'close']].values
    if np.isnan(train_ohlc).any():
        logger.warning("训练集中存在NaN值，将进行填充")
        train_ohlc = pd.DataFrame(train_ohlc).fillna(method='ffill').fillna(method='bfill').values
    
    train_dataset = TS2VecDataset(
        data=train_ohlc,
        window_length=ts2vec_config['window_length'],
        stride=1,
        augmentation_params={
            'aug_types': ['masking', 'warping', 'scaling'],
            'masking_ratio': ts2vec_config['masking_ratio'],
            'warp_ratio': ts2vec_config['time_warp_ratio'],
            'scale_range': ts2vec_config['magnitude_scale_range']
        }
    )
    
    # 处理验证集
    logger.info("处理验证集...")
    val_ohlc = val_df[['open', 'high', 'low', 'close']].values
    if np.isnan(val_ohlc).any():
        logger.warning("验证集中存在NaN值，将进行填充")
        val_ohlc = pd.DataFrame(val_ohlc).fillna(method='ffill').fillna(method='bfill').values
    
    val_dataset = TS2VecDataset(
        data=val_ohlc,
        window_length=ts2vec_config['window_length'],
        stride=1,
        augmentation_params={
            'aug_types': ['masking', 'warping', 'scaling'],
            'masking_ratio': ts2vec_config['masking_ratio'],
            'warp_ratio': ts2vec_config['time_warp_ratio'],
            'scale_range': ts2vec_config['magnitude_scale_range']
        }
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    return train_dataset, val_dataset


def create_model(config: dict, device: str) -> TS2VecModel:
    """
    创建TS2Vec模型
    
    Args:
        config: 配置字典
        device: 计算设备
        
    Returns:
        TS2VecModel实例
    """
    logger.info("创建TS2Vec模型...")
    
    ts2vec_config = config['ts2vec']
    
    model = TS2VecModel(
        input_dim=ts2vec_config['input_dim'],
        hidden_dim=ts2vec_config['hidden_dim'],
        output_dim=ts2vec_config['hidden_dim'] // 2,  # 输出维度为隐藏维度的一半
        num_layers=ts2vec_config['num_layers'],
        kernel_size=ts2vec_config['kernel_size'],
        dilation_rates=ts2vec_config.get('dilation_rates'),
        temperature=ts2vec_config['temperature']
    )
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    return model


def train_model(model: TS2VecModel,
                train_dataset,
                val_dataset,
                config: dict,
                device: str) -> dict:
    """
    训练模型
    
    Args:
        model: TS2Vec模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        config: 配置字典
        device: 计算设备
        
    Returns:
        训练历史
    """
    logger.info("开始训练TS2Vec模型...")
    
    ts2vec_config = config['ts2vec']
    
    # 创建数据加载器
    train_loader = OptimizedDataLoader.create_loader(
        train_dataset,
        batch_size=ts2vec_config['batch_size'],
        shuffle=True,
        device=device
    )
    
    val_loader = OptimizedDataLoader.create_loader(
        val_dataset,
        batch_size=ts2vec_config['batch_size'],
        shuffle=False,
        device=device
    )
    
    # 创建训练器
    trainer = TS2VecTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=ts2vec_config['learning_rate'],
        num_epochs=ts2vec_config['num_epochs'],
        warmup_epochs=5,
        patience=ts2vec_config['patience'],
        save_dir='models/checkpoints/ts2vec',
        device=device,
        # 可选的优化参数（默认禁用）
        use_amp=False,  # 混合精度训练
        gradient_accumulation_steps=1,
        use_compile=False,  # torch.compile
        optimizer_type='adam',
        weight_decay=0.0,
        grad_clip_norm=None
    )
    
    # 训练
    history = trainer.train()
    
    # 加载最佳模型
    trainer.load_best_model()
    
    # 保存最终模型
    final_model_path = 'models/checkpoints/ts2vec/ts2vec_final.pth'
    model.save(final_model_path)
    logger.info(f"最终模型已保存: {final_model_path}")
    
    return history


def evaluate_model(model: TS2VecModel,
                   test_dataset,
                   config: dict,
                   device: str) -> dict:
    """
    评估模型
    
    Args:
        model: 训练好的模型
        test_dataset: 测试数据集
        config: 配置字典
        device: 计算设备
        
    Returns:
        评估指标
    """
    logger.info("评估模型...")
    
    model.eval()
    
    ts2vec_config = config['ts2vec']
    
    test_loader = OptimizedDataLoader.create_loader(
        test_dataset,
        batch_size=ts2vec_config['batch_size'],
        shuffle=False,
        device=device
    )
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x_i, x_j in test_loader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            
            loss = model(x_i, x_j, return_loss=True)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    metrics = {
        'test_loss': avg_loss
    }
    
    logger.info(f"测试集损失: {avg_loss:.4f}")
    
    return metrics


def plot_training_history(history: dict, save_path: str = 'training/output/ts2vec_training_history.png'):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率曲线
        axes[1].plot(history['learning_rate'], label='Learning Rate')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练历史图已保存: {save_path}")
        
        plt.close()
    except ImportError:
        logger.warning("matplotlib未安装，跳过绘图")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("TS2Vec模型训练")
    logger.info("=" * 80)
    
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载预划分的数据集
    data_dir = config['data']['processed_data_dir']
    
    train_path = os.path.join(data_dir, 'MES_train.csv')
    val_path = os.path.join(data_dir, 'MES_val.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logger.error("未找到预划分的数据集!")
        logger.error("请先运行: python training/split_dataset.py")
        return
    
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    
    # 准备数据
    train_dataset, val_dataset = prepare_ts2vec_data(train_df, val_df, config)
    
    # 创建模型
    model = create_model(config, device)
    
    # 训练模型
    start_time = datetime.now()
    history = train_model(model, train_dataset, val_dataset, config, device)
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    logger.info(f"训练耗时: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 注意：最终评估应该使用独立的评估脚本和测试集
    # 这里不再评估测试集，避免数据泄露
    logger.info("训练完成！使用 training/evaluate_ts2vec.py 进行最终评估")
    
    # 保存训练报告
    report = {
        'training_time_seconds': training_time,
        'device': device,
        'config': config['ts2vec'],
        'data_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'train_time_range': f"{train_df.index[0]} to {train_df.index[-1]}",
            'val_time_range': f"{val_df.index[0]} to {val_df.index[-1]}"
        },
        'final_metrics': {
            'best_val_loss': min(history['val_loss'])
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'learning_rate': [float(x) for x in history['learning_rate']]
        }
    }
    
    # 保存为JSON
    import json
    report_path = 'training/output/ts2vec_training_report.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"训练报告已保存: {report_path}")
    
    # 保存为文本
    report_txt_path = 'training/output/ts2vec_training_report.txt'
    with open(report_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TS2Vec模型训练报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)\n")
        f.write(f"设备: {device}\n\n")
        f.write("数据信息:\n")
        f.write(f"  训练样本: {len(train_dataset)}\n")
        f.write(f"  验证样本: {len(val_dataset)}\n")
        f.write(f"  训练时间范围: {train_df.index[0]} 到 {train_df.index[-1]}\n")
        f.write(f"  验证时间范围: {val_df.index[0]} 到 {val_df.index[-1]}\n\n")
        f.write("最终指标:\n")
        f.write(f"  最佳验证损失: {min(history['val_loss']):.4f}\n\n")
        f.write("模型配置:\n")
        for key, value in config['ts2vec'].items():
            f.write(f"  {key}: {value}\n")
    logger.info(f"训练报告已保存: {report_txt_path}")
    
    logger.info("=" * 80)
    logger.info("训练完成!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()