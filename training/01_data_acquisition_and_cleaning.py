"""
数据获取和清洗脚本

功能：
1. 从yfinance下载OHLCV数据
2. 执行数据清洗流程（缺失值、异常值、时间对齐）
3. 计算27维手工特征
4. 保存清洗后的数据和特征

使用方法：
    python training/01_data_acquisition_and_cleaning.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime

from src.data.downloader import DataDownloader
from src.data.storage import DataStorage
from src.features.data_cleaner import DataCleaner
from src.features.feature_calculator import FeatureCalculator
from src.utils.logger import setup_logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="data_acquisition",
        log_file="01_data_acquisition.log",
        log_level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("开始数据获取和清洗流程")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("\n步骤1: 加载配置文件")
    config = load_config()
    
    data_config = config['data']
    symbols = data_config['symbols']
    interval = data_config['interval']
    start_date = data_config['start_date']
    end_date = data_config['end_date']
    
    logger.info(f"品种: {symbols}")
    logger.info(f"周期: {interval}")
    logger.info(f"时间范围: {start_date} 到 {end_date}")
    
    # 2. 下载数据
    logger.info("\n步骤2: 下载原始数据")
    downloader = DataDownloader(max_retries=3, retry_delay=5)
    
    raw_data_dict = {}
    for symbol in symbols:
        logger.info(f"\n下载 {symbol}...")
        df = downloader.download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if df is not None and not df.empty:
            # 标准化列名为大写（与特征计算器匹配）
            df.columns = [col.capitalize() for col in df.columns]
            raw_data_dict[symbol] = df
            logger.info(f"✓ {symbol} 下载成功: {len(df)} 条记录")
        else:
            logger.warning(f"✗ {symbol} 下载失败")
    
    if not raw_data_dict:
        logger.error("没有成功下载任何数据，退出")
        return
    
    # 保存原始数据
    logger.info("\n保存原始数据...")
    raw_storage = DataStorage(base_path='data/raw')
    raw_storage.save_multiple_parquet(raw_data_dict, compression='snappy')
    
    # 3. 数据清洗
    logger.info("\n步骤3: 数据清洗")
    cleaner = DataCleaner(
        max_consecutive_missing=data_config['max_consecutive_missing'],
        sigma_threshold=data_config['outlier_sigma']
    )
    
    cleaned_data_dict = {}
    cleaning_reports = {}
    
    for symbol, raw_df in raw_data_dict.items():
        logger.info(f"\n清洗 {symbol}...")
        
        try:
            # 执行完整清洗流程
            cleaned_df, report = cleaner.clean_pipeline(
                df=raw_df,
                target_timezone=data_config['target_timezone'],
                trading_hours=None  # 24小时交易
            )
            
            cleaned_data_dict[symbol] = cleaned_df
            cleaning_reports[symbol] = report
            
            logger.info(f"✓ {symbol} 清洗完成: {len(cleaned_df)} 条记录")
            
        except Exception as e:
            logger.error(f"✗ {symbol} 清洗失败: {e}")
            continue
    
    if not cleaned_data_dict:
        logger.error("没有成功清洗任何数据，退出")
        return
    
    # 4. 计算手工特征
    logger.info("\n步骤4: 计算27维手工特征")
    feature_calculator = FeatureCalculator()
    
    feature_data_dict = {}
    
    for symbol, cleaned_df in cleaned_data_dict.items():
        logger.info(f"\n计算 {symbol} 的特征...")
        
        try:
            # 计算所有特征
            df_with_features = feature_calculator.calculate_all_features(cleaned_df)
            
            feature_data_dict[symbol] = df_with_features
            
            logger.info(f"✓ {symbol} 特征计算完成")
            logger.info(f"  - 数据行数: {len(df_with_features)}")
            logger.info(f"  - 特征数量: {len(feature_calculator.get_feature_names())}")
            
        except Exception as e:
            logger.error(f"✗ {symbol} 特征计算失败: {e}")
            continue
    
    if not feature_data_dict:
        logger.error("没有成功计算任何特征，退出")
        return
    
    # 5. 保存处理后的数据
    logger.info("\n步骤5: 保存处理后的数据")
    processed_storage = DataStorage(base_path='data/processed')
    
    for symbol, df_features in feature_data_dict.items():
        # 保存为parquet格式
        success = processed_storage.save_parquet(
            data=df_features,
            symbol=f"{symbol}_with_features",
            compression='snappy'
        )
        
        if success:
            logger.info(f"✓ {symbol} 数据已保存")
        else:
            logger.warning(f"✗ {symbol} 数据保存失败")
    
    # 6. 生成数据摘要报告
    logger.info("\n步骤6: 生成数据摘要报告")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("数据获取和清洗报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for symbol in feature_data_dict.keys():
        report_lines.append(f"\n品种: {symbol}")
        report_lines.append("-" * 80)
        
        # 原始数据信息
        if symbol in raw_data_dict:
            raw_df = raw_data_dict[symbol]
            report_lines.append(f"原始数据: {len(raw_df)} 条记录")
            report_lines.append(f"  时间范围: {raw_df.index[0]} 到 {raw_df.index[-1]}")
        
        # 清洗后数据信息
        if symbol in cleaned_data_dict:
            cleaned_df = cleaned_data_dict[symbol]
            report_lines.append(f"清洗后数据: {len(cleaned_df)} 条记录")
        
        # 特征数据信息
        if symbol in feature_data_dict:
            feature_df = feature_data_dict[symbol]
            report_lines.append(f"特征数据: {len(feature_df)} 条记录")
            report_lines.append(f"  OHLCV列: {['Open', 'High', 'Low', 'Close', 'Volume']}")
            report_lines.append(f"  手工特征: {len(feature_calculator.get_feature_names())} 个")
            
            # 特征分组
            feature_groups = feature_calculator.get_feature_groups()
            for group_name, features in feature_groups.items():
                report_lines.append(f"    - {group_name}: {len(features)} 个")
        
        # 清洗报告摘要
        if symbol in cleaning_reports:
            report = cleaning_reports[symbol]
            report_lines.append(f"\n清洗统计:")
            
            if 'missing_values' in report:
                mv = report['missing_values']
                report_lines.append(f"  - 删除数据段: {len(mv['removed_segments'])} 个")
                report_lines.append(f"  - 插值点数: {sum(mv['interpolated_points'].values())} 个")
            
            if 'outliers' in report:
                ol = report['outliers']
                report_lines.append(f"  - 检测异常: {ol['total_outliers']} 个")
                report_lines.append(f"  - 修正尖峰: {ol['spike_outliers']} 个")
            
            if 'validation' in report:
                vd = report['validation']
                report_lines.append(f"  - 质量验证: {'通过✓' if vd['is_valid'] else '失败✗'}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append(f"总计: 成功处理 {len(feature_data_dict)} 个品种")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = "training/output/01_data_acquisition_report.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n报告已保存: {report_path}")
    
    # 7. 输出特征名称列表
    feature_names = feature_calculator.get_feature_names()
    feature_names_path = "training/output/feature_names.txt"
    
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        f.write("27维手工特征列表:\n")
        f.write("=" * 80 + "\n\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"{i:2d}. {name}\n")
    
    logger.info(f"特征名称列表已保存: {feature_names_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("数据获取和清洗流程完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()