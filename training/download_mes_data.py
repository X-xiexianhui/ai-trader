"""
下载MES（Micro E-mini S&P 500期货）最近60天的5分钟K线数据

使用方法:
    python download_mes_data.py
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timedelta
from src.data.downloader import DataDownloader
from src.utils.logger import setup_logger
import logging

# 设置日志
logger = setup_logger('download_mes', log_dir='training/output', log_file='download_mes.log')


def download_mes_data(days: int = 60, save_path: str = None):
    """
    下载MES最近N天的5分钟K线数据
    
    Args:
        days: 下载天数，默认60天（注意：雅虎金融5分钟数据最多只能获取最近60天）
        save_path: 保存路径，默认为training/output/mes_5m_data.csv
    """
    # MES期货代码 (Micro E-mini S&P 500)
    # 注意：雅虎金融可能使用不同的代码格式
    symbol = 'MES=F'
    
    # 如果MES=F不可用，尝试其他可能的代码
    alternative_symbols = ['ES=F', '^GSPC']  # ES期货或S&P 500指数
    
    # 设置默认保存路径
    if save_path is None:
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / 'mes_5m_data.csv'
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始下载 {symbol} 最近 {days} 天的5分钟K线数据")
    logger.info(f"保存路径: {save_path}")
    
    # 创建下载器
    downloader = DataDownloader(
        max_retries=3,
        retry_delay=5,
        timeout=30
    )
    
    # 计算日期范围（确保不超过60天，这是雅虎金融5分钟数据的限制）
    end_date = datetime.now()
    # 雅虎金融5分钟数据限制在最近60天内
    max_days = min(days, 59)  # 使用59天以确保在限制内
    start_date = end_date - timedelta(days=max_days)
    
    logger.info(f"日期范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"注意：雅虎金融5分钟数据最多只能获取最近60天")
    
    # 尝试下载数据
    df = None
    symbols_to_try = [symbol] + alternative_symbols
    
    for try_symbol in symbols_to_try:
        logger.info(f"尝试下载 {try_symbol}")
        df = downloader.download(
            symbol=try_symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            interval='5m'
        )
        
        if df is not None and not df.empty:
            logger.info(f"成功使用代码 {try_symbol} 下载数据")
            symbol = try_symbol  # 更新使用的代码
            break
        else:
            logger.warning(f"{try_symbol} 下载失败，尝试下一个代码")
    
    if df is None or df.empty:
        logger.error("所有代码都下载失败")
        logger.info("提示：MES期货可能需要特殊的数据源，或者使用ES=F（标准E-mini S&P 500）代替")
        return None
    
    # 添加品种列
    df['symbol'] = symbol
    
    # 重置索引，将时间戳作为列
    df = df.reset_index()
    df = df.rename(columns={'index': 'datetime', 'Datetime': 'datetime'})
    
    # 调整列顺序
    columns_order = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    df = df[columns_order]
    
    # 保存到CSV
    df.to_csv(save_path, index=False)
    logger.info(f"数据已保存到: {save_path}")
    logger.info(f"共 {len(df)} 条记录")
    
    # 显示数据统计信息
    logger.info("\n数据统计:")
    logger.info(f"  时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}")
    logger.info(f"  价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    logger.info(f"  平均成交量: {df['volume'].mean():.0f}")
    logger.info(f"  数据完整性: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    # 显示前几行数据
    logger.info("\n前5行数据:")
    logger.info(f"\n{df.head()}")
    
    # 同时保存为Parquet格式（更高效）
    parquet_path = save_path.with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    logger.info(f"\n数据也已保存为Parquet格式: {parquet_path}")
    
    return df


def main():
    """主函数"""
    try:
        # 下载60天的数据
        df = download_mes_data(days=60)
        
        if df is not None:
            logger.info("\n✓ 数据下载成功！")
        else:
            logger.error("\n✗ 数据下载失败")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"\n✗ 发生错误: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()