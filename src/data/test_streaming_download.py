"""
测试脚本：流式下载长期历史数据
演示如何使用流式下载功能边下载边保存，适合下载长期历史数据
"""

import logging
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data import (
    IBConnector,
    IBHistoricalDataDownloader,
    create_futures_contract,
    DataStorage
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_streaming_download():
    """测试流式下载功能"""
    
    # 配置参数
    symbol = 'ES'  # E-mini S&P 500
    exchange = 'CME'
    bar_size = '5m'
    
    # 下载最近6个月的数据（演示长期数据下载）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    # 保存路径
    save_path = f'data/raw/{symbol}_{bar_size}_streaming.csv'
    
    logger.info("=" * 70)
    logger.info("流式下载历史数据测试")
    logger.info("=" * 70)
    logger.info(f"合约: {symbol}")
    logger.info(f"交易所: {exchange}")
    logger.info(f"K线周期: {bar_size}")
    logger.info(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"结束日期: {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"保存路径: {save_path}")
    logger.info("=" * 70)
    
    try:
        # 连接到IB Gateway
        logger.info("\n步骤1: 连接到IB Gateway...")
        with IBConnector(host='127.0.0.1', port=4002, client_id=1) as connector:
            logger.info("✓ 成功连接到IB Gateway")
            
            # 创建期货合约
            logger.info(f"\n步骤2: 创建{symbol}期货合约...")
            contract = create_futures_contract(
                symbol=symbol,
                exchange=exchange
            )
            
            # 验证合约
            logger.info("\n步骤3: 验证合约...")
            qualified = connector.qualify_contract(contract)
            
            if not qualified:
                logger.error("✗ 合约验证失败！")
                return None
            
            logger.info(f"✓ 合约验证成功: {qualified}")
            
            # 创建下载器
            downloader = IBHistoricalDataDownloader(connector)
            
            # 使用流式下载
            logger.info(f"\n步骤4: 开始流式下载...")
            logger.info("提示：数据将边下载边保存到文件，不会占用大量内存")
            
            df = downloader.download_historical_data_streaming(
                contract=qualified,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                what_to_show='TRADES',
                use_rth=False,
                save_path=save_path,
                chunk_days=30  # 每次下载30天
            )
            
            if df is None or df.empty:
                logger.error("✗ 未获取到数据！")
                return None
            
            logger.info(f"\n✓ 流式下载完成！")
            
            # 数据统计
            logger.info("\n步骤5: 数据统计...")
            logger.info(f"数据起始时间: {df['date'].min()}")
            logger.info(f"数据结束时间: {df['date'].max()}")
            logger.info(f"数据条数: {len(df)}")
            logger.info(f"数据跨度: {(df['date'].max() - df['date'].min()).days} 天")
            
            # 显示数据样本
            logger.info("\n前5条数据:")
            print(df.head().to_string())
            
            logger.info("\n后5条数据:")
            print(df.tail().to_string())
            
            # 价格统计
            logger.info("\n价格统计:")
            logger.info(f"最高价: {df['high'].max():.2f}")
            logger.info(f"最低价: {df['low'].min():.2f}")
            logger.info(f"平均收盘价: {df['close'].mean():.2f}")
            logger.info(f"总成交量: {df['volume'].sum():,.0f}")
            
            # 文件大小
            import os
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                logger.info(f"\n文件大小: {file_size:.2f} MB")
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ 测试完成！")
            logger.info("=" * 70)
            
            return df
            
    except Exception as e:
        logger.error(f"\n✗ 发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def compare_methods():
    """比较标准下载和流式下载的区别"""
    
    logger.info("\n" + "=" * 70)
    logger.info("下载方法对比")
    logger.info("=" * 70)
    
    print("\n1. 标准下载 (download_historical_data_range):")
    print("   优点：")
    print("   - 简单易用")
    print("   - 返回完整DataFrame")
    print("   缺点：")
    print("   - 所有数据保存在内存中")
    print("   - 不适合下载长期数据（可能内存溢出）")
    print("   - 下载失败需要重新开始")
    
    print("\n2. 流式下载 (download_historical_data_streaming):")
    print("   优点：")
    print("   - 边下载边保存，内存占用小")
    print("   - 适合下载长期历史数据")
    print("   - 支持断点续传（CSV格式）")
    print("   - 可以实时查看下载进度")
    print("   缺点：")
    print("   - 需要指定保存路径")
    print("   - Parquet格式不支持真正的追加")
    
    print("\n推荐使用场景：")
    print("- 下载 < 1个月数据：使用标准下载")
    print("- 下载 > 1个月数据：使用流式下载")
    print("- 下载 > 6个月数据：强烈推荐流式下载")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("流式下载历史数据测试")
    print("=" * 70)
    print("\n请确保：")
    print("1. IB Gateway或TWS已启动")
    print("2. API连接已启用")
    print("3. 端口设置为4002（模拟盘）或4001（实盘）")
    print("4. 已订阅ES市场数据")
    
    # 显示方法对比
    compare_methods()
    
    print("\n按Enter键开始测试...")
    input()
    
    # 运行测试
    df = test_streaming_download()
    
    if df is not None:
        print("\n" + "=" * 70)
        print("测试成功完成！")
        print("=" * 70)
        print(f"\n数据已保存，共 {len(df)} 条记录")
        print("\n使用流式下载的优势：")
        print("✓ 内存占用小（边下载边保存）")
        print("✓ 支持长期数据下载")
        print("✓ 可以实时查看进度")
        print("✓ 下载过程更稳定")
    else:
        print("\n" + "=" * 70)
        print("测试失败！请检查错误信息")
        print("=" * 70)