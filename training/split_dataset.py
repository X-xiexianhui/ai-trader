"""
数据集划分脚本

将清洗后的数据划分为训练集、验证集和测试集，并保存为单独的文件
这样训练和评估时可以直接使用，确保数据划分的一致性
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
import logging
from src.utils.logger import setup_logger

# 设置日志
logger = setup_logger(
    name='dataset_split',
    log_level='INFO',
    log_dir='logs',
    log_file='dataset_split.log'
)


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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
    logger.info(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
    
    return df


def split_dataset(df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15) -> tuple:
    """
    划分数据集
    
    Args:
        df: 原始数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        (train_df, val_df, test_df)
    """
    logger.info("划分数据集...")
    
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    total_size = len(df)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    # 按时间顺序划分（不打乱）
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    logger.info(f"总样本数: {total_size}")
    logger.info(f"训练集: {len(train_df)} 样本 ({len(train_df)/total_size*100:.1f}%)")
    logger.info(f"  时间范围: {train_df.index[0]} 到 {train_df.index[-1]}")
    logger.info(f"验证集: {len(val_df)} 样本 ({len(val_df)/total_size*100:.1f}%)")
    logger.info(f"  时间范围: {val_df.index[0]} 到 {val_df.index[-1]}")
    logger.info(f"测试集: {len(test_df)} 样本 ({len(test_df)/total_size*100:.1f}%)")
    logger.info(f"  时间范围: {test_df.index[0]} 到 {test_df.index[-1]}")
    
    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: str) -> None:
    """
    保存划分后的数据集
    
    Args:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        output_dir: 输出目录
    """
    logger.info(f"保存划分后的数据集到: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为CSV（重置索引以保存时间戳）
    train_path = os.path.join(output_dir, 'MES_train.csv')
    val_path = os.path.join(output_dir, 'MES_val.csv')
    test_path = os.path.join(output_dir, 'MES_test.csv')
    
    train_df.reset_index().to_csv(train_path, index=False)
    val_df.reset_index().to_csv(val_path, index=False)
    test_df.reset_index().to_csv(test_path, index=False)
    
    logger.info(f"✓ 训练集已保存: {train_path}")
    logger.info(f"✓ 验证集已保存: {val_path}")
    logger.info(f"✓ 测试集已保存: {test_path}")
    
    # 保存划分信息
    split_info = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
        'val_ratio': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
        'test_ratio': len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
        'train_time_range': f"{train_df.index[0]} to {train_df.index[-1]}",
        'val_time_range': f"{val_df.index[0]} to {val_df.index[-1]}",
        'test_time_range': f"{test_df.index[0]} to {test_df.index[-1]}"
    }
    
    import json
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"✓ 划分信息已保存: {info_path}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("数据集划分")
    logger.info("=" * 80)
    
    # 加载配置
    config = load_config()
    
    # 加载数据
    data_path = os.path.join(
        config['data']['processed_data_dir'],
        'MES_cleaned_5m.csv'
    )
    df = load_data(data_path)
    
    # 划分数据集
    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 保存划分后的数据集
    output_dir = config['data']['processed_data_dir']
    save_splits(train_df, val_df, test_df, output_dir)
    
    logger.info("=" * 80)
    logger.info("数据集划分完成!")
    logger.info("=" * 80)
    logger.info("\n使用方法:")
    logger.info("  训练: 使用 MES_train.csv 和 MES_val.csv")
    logger.info("  评估: 使用 MES_test.csv")


if __name__ == '__main__':
    main()