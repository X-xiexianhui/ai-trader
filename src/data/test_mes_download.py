"""
测试脚本：下载MES连续合约最近3个月的5分钟历史数据
MES = Micro E-mini S&P 500 期货
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
    create_continuous_futures_contract,
    DataStorage
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def download_mes_continuous_data():
    """下载MES连续合约历史数据"""
    
    # 配置参数
    symbol = 'MES'  # Micro E-mini S&P 500
    exchange = 'CME'
    bar_size = '5m'
    
    # 计算日期范围（最近3个月）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    logger.info("=" * 60)
    logger.info("MES连续合约历史数据下载测试")
    logger.info("=" * 60)
    logger.info(f"合约: {symbol} (Micro E-mini S&P 500)")
    logger.info(f"交易所: {exchange}")
    logger.info(f"K线周期: {bar_size}")
    logger.info(f"开始日期: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"结束日期: {end_date.strftime('%Y-%m-%d')}")
    logger.info("=" * 60)
    
    try:
        # 连接到IB Gateway
        logger.info("\n步骤1: 连接到IB Gateway...")
        with IBConnector(host='127.0.0.1', port=4001, client_id=1) as connector:
            logger.info("✓ 成功连接到IB Gateway")
            
            # 获取服务器时间
            server_time = connector.get_current_time()
            logger.info(f"IB服务器时间: {server_time}")
            
            # 创建连续期货合约
            logger.info(f"\n步骤2: 创建{symbol}连续期货合约...")
            contract = create_continuous_futures_contract(
                symbol=symbol,
                exchange=exchange
            )
            logger.info(f"合约创建完成: {contract.symbol} {contract.secType} {contract.exchange}")
            
            # 验证合约
            logger.info("\n步骤3: 验证合约...")
            qualified = connector.qualify_contract(contract)
            
            if not qualified:
                logger.error("✗ 合约验证失败！")
                logger.error("可能的原因：")
                logger.error("  1. 没有订阅MES市场数据")
                logger.error("  2. 合约代码或交易所不正确")
                logger.error("  3. IB Gateway配置问题")
                return None
            
            logger.info(f"✓ 合约验证成功: {qualified}")
            
            # 获取合约详情
            logger.info("\n步骤4: 获取合约详情...")
            details = connector.get_contract_details(qualified)
            if details:
                detail = details[0]
                logger.info(f"合约全名: {detail.longName}")
                logger.info(f"合约月份: {detail.contractMonth}")
                logger.info(f"交易时间: {detail.tradingHours}")
                logger.info(f"最小价格变动: {detail.minTick}")
            
            # 下载历史数据
            logger.info(f"\n步骤5: 下载历史数据...")
            logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
            downloader = IBHistoricalDataDownloader(connector)
            
            df = downloader.download_historical_data_range(
                contract=qualified,
                start_date=start_date,
                bar_size=bar_size,
                end_date=end_date,
                what_to_show='TRADES',
                use_rth=True  # 仅包含常规交易时间数据
            )
            
            if df is None or df.empty:
                logger.error("✗ 未获取到数据！")
                return None
            
            logger.info(f"✓ 成功下载 {len(df)} 条数据")
            
            # 数据统计
            logger.info("\n步骤6: 数据统计...")
            logger.info(f"数据起始时间: {df['date'].min()}")
            logger.info(f"数据结束时间: {df['date'].max()}")
            logger.info(f"数据条数: {len(df)}")
            logger.info(f"数据列: {list(df.columns)}")
            
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
            
            # 保存数据
            logger.info("\n步骤7: 保存数据...")
            
            # 使用DataStorage保存
            storage = DataStorage(base_dir='data')
            
            metadata = {
                'source': 'IB Gateway',
                'symbol': symbol,
                'contract_type': 'CONTFUT',
                'exchange': exchange,
                'interval': bar_size,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'records': len(df),
                'data_range': {
                    'start': df['date'].min().isoformat(),
                    'end': df['date'].max().isoformat()
                },
                'price_stats': {
                    'high': float(df['high'].max()),
                    'low': float(df['low'].min()),
                    'avg_close': float(df['close'].mean())
                }
            }
            
            # 保存为Parquet格式（推荐）
            filepath_parquet = storage.save_raw_data(
                df=df,
                symbol=symbol,
                interval=bar_size,
                format='parquet',
                metadata=metadata
            )
            logger.info(f"✓ 数据已保存为Parquet: {filepath_parquet}")
            
            # 同时保存为CSV格式（便于查看）
            filepath_csv = storage.save_raw_data(
                df=df,
                symbol=symbol,
                interval=bar_size,
                format='csv',
                metadata=metadata
            )
            logger.info(f"✓ 数据已保存为CSV: {filepath_csv}")
            
            # 显示存储信息
            logger.info("\n步骤8: 存储信息...")
            storage_info = storage.get_storage_info()
            logger.info(f"原始数据文件数: {storage_info['raw_data']['count']}")
            logger.info(f"原始数据大小: {storage_info['raw_data']['size_mb']:.2f} MB")
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ 测试完成！")
            logger.info("=" * 60)
            
            return df
            
    except Exception as e:
        logger.error(f"\n✗ 发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def verify_saved_data():
    """验证保存的数据"""
    logger.info("\n验证保存的数据...")
    
    storage = DataStorage(base_dir='data')
    
    # 加载最新的MES数据
    df = storage.load_latest_raw_data(symbol='MES', interval='5m')
    
    if df is not None:
        logger.info(f"✓ 成功加载数据: {len(df)} 条记录")
        logger.info(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
        return True
    else:
        logger.error("✗ 无法加载数据")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MES连续合约历史数据下载测试")
    print("=" * 60)
    print("\n请确保：")
    print("1. IB Gateway或TWS已启动")
    print("2. API连接已启用（配置 -> API -> 启用ActiveX和Socket客户端）")
    print("3. 端口设置为4001（模拟盘）或4001（实盘）")
    print("4. 已订阅MES市场数据")
    print("\n按Enter键开始测试...")
    input()
    
    # 下载数据
    df = download_mes_continuous_data()
    
    if df is not None:
        # 验证数据
        verify_saved_data()
        
        print("\n" + "=" * 60)
        print("测试成功完成！")
        print("=" * 60)
        print(f"\n数据已保存到 data/raw/ 目录")
        print(f"共下载 {len(df)} 条5分钟K线数据")
        print("\n可以使用以下代码加载数据：")
        print("```python")
        print("from src.data import DataStorage")
        print("storage = DataStorage()")
        print("df = storage.load_latest_raw_data('MES', '5m')")
        print("```")
    else:
        print("\n" + "=" * 60)
        print("测试失败！请检查错误信息")
        print("=" * 60)