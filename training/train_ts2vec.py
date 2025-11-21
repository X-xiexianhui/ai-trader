"""
TS2Vec模型训练脚本

功能：
1. 从本地文件读取处理后的MES数据
2. 准备训练/验证数据集
3. 训练TS2Vec模型
4. 保存训练好的模型和训练历史

使用方法:
    python training/train_ts2vec.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.ts2vec.model import TS2VecModel
from src.models.ts2vec.data_preparation import TS2VecDataset
from src.models.ts2vec.training import TS2VecTrainer
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger('train_ts2vec', log_dir='training/output', log_file='train_ts2vec.log')


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_processed_data(data_path: str) -> pd.DataFrame:
    """
    加载处理后的数据
    
    Args:
        data_path: 数据文件路径（CSV或Parquet）
        
    Returns:
        DataFrame with datetime index
    """
    logger.info(f"加载处理后的数据: {data_path}")
    
    # 根据文件扩展名选择加载方法
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    
    logger.info(f"数据加载完成: {len(df)}行 × {len(df.columns)}列")
    logger.info(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    
    return df


def prepare_ts2vec_data(
    df: pd.DataFrame,
    config: dict,
    train_ratio: float = 0.8
) -> tuple:
    """
    准备TS2Vec训练数据
    
    Args:
        df: 处理后的DataFrame
        config: 配置字典
        train_ratio: 训练集比例
        
    Returns:
        (train_loader, val_loader, dataset_info)
    """
    logger.info("=" * 80)
    logger.info("准备TS2Vec训练数据...")
    logger.info("=" * 80)
    
    # 提取OHLC数据（TS2Vec只使用价格数据）
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    ohlc_data = df[ohlc_columns].values
    
    logger.info(f"OHLC数据形状: {ohlc_data.shape}")
    
    # 获取TS2Vec配置
    ts2vec_config = config['ts2vec']
    window_length = ts2vec_config['window_length']
    batch_size = ts2vec_config['batch_size']
    
    # 数据增强参数
    aug_params = {
        'aug_types': ['masking', 'warping', 'scaling'],
        'masking_ratio': ts2vec_config['masking_ratio'],
        'warp_ratio': ts2vec_config['time_warp_ratio'],
        'scale_range': ts2vec_config['magnitude_scale_range']
    }
    
    # 创建完整数据集
    logger.info(f"创建TS2Vec数据集 (窗口长度={window_length})...")
    full_dataset = TS2VecDataset(
        data=ohlc_data,
        window_length=window_length,
        stride=1,
        augmentation_params=aug_params
    )
    
    # 划分训练集和验证集
    total_size = len(full_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    logger.info(f"数据集划分: 训练集={train_size}, 验证集={val_size}")
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    dataset_info = {
        'total_windows': total_size,
        'train_windows': train_size,
        'val_windows': val_size,
        'window_length': window_length,
        'input_dim': ohlc_data.shape[1],
        'batch_size': batch_size
    }
    
    logger.info(f"DataLoader创建完成:")
    logger.info(f"  训练批次数: {len(train_loader)}")
    logger.info(f"  验证批次数: {len(val_loader)}")
    
    return train_loader, val_loader, dataset_info


def create_ts2vec_model(config: dict, device: str) -> TS2VecModel:
    """
    创建TS2Vec模型
    
    Args:
        config: 配置字典
        device: 计算设备
        
    Returns:
        TS2VecModel实例
    """
    logger.info("=" * 80)
    logger.info("创建TS2Vec模型...")
    logger.info("=" * 80)
    
    ts2vec_config = config['ts2vec']
    
    model = TS2VecModel(
        input_dim=ts2vec_config['input_dim'],
        hidden_dim=ts2vec_config['hidden_dim'],
        output_dim=ts2vec_config['hidden_dim'] // 2,  # 投影维度为隐藏维度的一半
        num_layers=ts2vec_config['num_layers'],
        kernel_size=ts2vec_config['kernel_size'],
        dilation_rates=ts2vec_config['dilation_rates'],
        temperature=ts2vec_config['temperature']
    )
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"模型创建完成:")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    logger.info(f"  计算设备: {device}")
    
    return model


def train_model(
    model: TS2VecModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str
) -> dict:
    """
    训练TS2Vec模型
    
    Args:
        model: TS2Vec模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        device: 计算设备
        
    Returns:
        训练历史
    """
    logger.info("=" * 80)
    logger.info("开始训练TS2Vec模型...")
    logger.info("=" * 80)
    
    ts2vec_config = config['ts2vec']
    
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
        device=device
    )
    
    # 训练
    history = trainer.train()
    
    # 加载最佳模型
    trainer.load_best_model()
    
    logger.info("训练完成!")
    
    return history


def plot_training_history(history: dict, save_path: str) -> None:
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    logger.info("绘制训练历史曲线...")
    
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
    
    logger.info(f"训练历史曲线已保存: {save_path}")


def save_training_summary(
    dataset_info: dict,
    history: dict,
    config: dict,
    save_path: str
) -> None:
    """
    保存训练摘要报告
    
    Args:
        dataset_info: 数据集信息
        history: 训练历史
        config: 配置字典
        save_path: 保存路径
    """
    logger.info("生成训练摘要报告...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TS2Vec模型训练摘要报告")
    report_lines.append("=" * 80)
    report_lines.append(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. 数据集信息
    report_lines.append("1. 数据集信息")
    report_lines.append("-" * 80)
    report_lines.append(f"总窗口数: {dataset_info['total_windows']}")
    report_lines.append(f"训练窗口: {dataset_info['train_windows']}")
    report_lines.append(f"验证窗口: {dataset_info['val_windows']}")
    report_lines.append(f"窗口长度: {dataset_info['window_length']}")
    report_lines.append(f"输入维度: {dataset_info['input_dim']} (OHLC)")
    report_lines.append(f"批次大小: {dataset_info['batch_size']}")
    report_lines.append("")
    
    # 2. 模型配置
    report_lines.append("2. 模型配置")
    report_lines.append("-" * 80)
    ts2vec_config = config['ts2vec']
    report_lines.append(f"隐藏维度: {ts2vec_config['hidden_dim']}")
    report_lines.append(f"编码器层数: {ts2vec_config['num_layers']}")
    report_lines.append(f"卷积核大小: {ts2vec_config['kernel_size']}")
    report_lines.append(f"膨胀率: {ts2vec_config['dilation_rates']}")
    report_lines.append(f"温度参数: {ts2vec_config['temperature']}")
    report_lines.append("")
    
    # 3. 训练配置
    report_lines.append("3. 训练配置")
    report_lines.append("-" * 80)
    report_lines.append(f"训练轮数: {ts2vec_config['num_epochs']}")
    report_lines.append(f"学习率: {ts2vec_config['learning_rate']}")
    report_lines.append(f"早停patience: {ts2vec_config['patience']}")
    report_lines.append(f"数据增强:")
    report_lines.append(f"  - 时间遮蔽比例: {ts2vec_config['masking_ratio']}")
    report_lines.append(f"  - 时间扭曲比例: {ts2vec_config['time_warp_ratio']}")
    report_lines.append(f"  - 幅度缩放范围: {ts2vec_config['magnitude_scale_range']}")
    report_lines.append("")
    
    # 4. 训练结果
    report_lines.append("4. 训练结果")
    report_lines.append("-" * 80)
    report_lines.append(f"实际训练轮数: {len(history['train_loss'])}")
    report_lines.append(f"最佳验证损失: {min(history['val_loss']):.6f}")
    report_lines.append(f"最终训练损失: {history['train_loss'][-1]:.6f}")
    report_lines.append(f"最终验证损失: {history['val_loss'][-1]:.6f}")
    
    # 找到最佳epoch
    best_epoch = np.argmin(history['val_loss']) + 1
    report_lines.append(f"最佳epoch: {best_epoch}")
    report_lines.append("")
    
    # 5. 损失变化趋势
    report_lines.append("5. 损失变化趋势")
    report_lines.append("-" * 80)
    epochs_to_show = [1, 10, 20, 30, 40, 50, len(history['train_loss'])]
    epochs_to_show = [e for e in epochs_to_show if e <= len(history['train_loss'])]
    
    report_lines.append(f"{'Epoch':<10} {'Train Loss':<15} {'Val Loss':<15}")
    report_lines.append("-" * 40)
    for epoch in epochs_to_show:
        idx = epoch - 1
        report_lines.append(
            f"{epoch:<10} {history['train_loss'][idx]:<15.6f} {history['val_loss'][idx]:<15.6f}"
        )
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("训练完成！")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"训练摘要报告已保存: {save_path}")
    
    # 同时打印到控制台
    print("\n" + report_text)


def main():
    """主函数"""
    try:
        # 检测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用计算设备: {device}")
        
        # 1. 加载配置
        config = load_config('configs/config.yaml')
        
        # 2. 加载处理后的数据
        # 优先使用Parquet格式（更快）
        data_path = 'training/output/mes_features_normalized.parquet'
        if not Path(data_path).exists():
            data_path = 'training/output/mes_features_normalized.csv'
        
        df = load_processed_data(data_path)
        
        # 3. 准备训练数据
        train_loader, val_loader, dataset_info = prepare_ts2vec_data(
            df, config, train_ratio=0.8
        )
        
        # 4. 创建模型
        model = create_ts2vec_model(config, device)
        
        # 5. 训练模型
        history = train_model(model, train_loader, val_loader, config, device)
        
        # 6. 绘制训练历史
        plot_path = 'training/output/ts2vec_training_history.png'
        plot_training_history(history, plot_path)
        
        # 7. 保存训练摘要
        summary_path = 'training/output/ts2vec_training_summary.txt'
        save_training_summary(dataset_info, history, config, summary_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ TS2Vec模型训练完成！")
        logger.info("=" * 80)
        logger.info(f"最佳模型保存在: models/checkpoints/ts2vec/best_model.pt")
        logger.info(f"训练历史图: {plot_path}")
        logger.info(f"训练摘要: {summary_path}")
        
    except Exception as e:
        logger.error(f"\n✗ 训练失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()