#!/usr/bin/env python3
"""
数据下载脚本

独立的数据下载工具，用于获取OHLCV数据
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.downloader import YahooFinanceDownloader
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def download_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "5m",
    output_dir: str = "data/raw"
):
    """
    下载市场数据
    
    Args:
        symbol: 交易品种
        start_date: 开始日期
        end_date: 结束日期
        interval: 数据频率
        output_dir: 输出目录
    """
    # 设置日志
    setup_logger(
        log_dir=Path(__file__).parent / "logs" / "data",
        log_level=logging.INFO
    )
    
    logger.info("=" * 80)
    logger.info("Data Download Tool")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Interval: {interval}")
    logger.info("=" * 80)
    
    try:
        # 初始化下载器
        downloader = YahooFinanceDownloader()
        
        # 下载数据
        logger.info("Downloading data...")
        data = downloader.download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        logger.info(f"Downloaded {len(data)} bars")
        
        # 显示数据信息
        logger.info("\nData Summary:")
        logger.info(f"  Columns: {list(data.columns)}")
        logger.info(f"  Date Range: {data.index[0]} to {data.index[-1]}")
        logger.info(f"  Shape: {data.shape}")
        logger.info(f"\nFirst few rows:")
        logger.info(f"\n{data.head()}")
        logger.info(f"\nLast few rows:")
        logger.info(f"\n{data.tail()}")
        
        # 保存数据
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        safe_symbol = symbol.replace('=', '_').replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{safe_symbol}_{start_date}_{end_date}_{timestamp}.parquet"
        
        output_file = output_path / filename
        data.to_parquet(output_file)
        
        logger.info("=" * 80)
        logger.info(f"✓ Data saved to: {output_file}")
        logger.info("=" * 80)
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Download OHLCV market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 下载ES期货数据（最近30天）
  python download_data.py --symbol ES=F --days 30
  
  # 下载指定日期范围的数据
  python download_data.py --symbol ES=F --start 2023-01-01 --end 2023-12-31
  
  # 下载1分钟数据
  python download_data.py --symbol ES=F --days 7 --interval 1m
  
  # 下载多个品种
  python download_data.py --symbol AAPL MSFT GOOGL --days 90
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        nargs="+",
        required=True,
        help="Trading symbol(s) to download"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to download (from today backwards)"
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # 确定日期范围
    if args.days:
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
    elif args.start and args.end:
        start_str = args.start
        end_str = args.end
    else:
        # 默认下载最近30天
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
    
    # 下载所有品种
    symbols = args.symbol if isinstance(args.symbol, list) else [args.symbol]
    
    success_count = 0
    for symbol in symbols:
        logger.info(f"\nProcessing {symbol}...")
        result = download_data(
            symbol=symbol,
            start_date=start_str,
            end_date=end_str,
            interval=args.interval,
            output_dir=args.output
        )
        
        if result:
            success_count += 1
    
    # 显示结果
    print("\n" + "=" * 80)
    print(f"Download Summary:")
    print(f"  Total symbols: {len(symbols)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(symbols) - success_count}")
    print(f"  Output directory: {args.output}")
    print("=" * 80)
    
    if success_count == len(symbols):
        print("\n✓ All downloads completed successfully!")
        sys.exit(0)
    else:
        print("\n⚠ Some downloads failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()