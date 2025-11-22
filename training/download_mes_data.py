"""
下载MES期货主连合约历史数据
时间范围: 2023年12月 到 2025年12月
只下载交易时间段数据
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import pytz

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ib_connector import IBConnector
from src.data.ib_historical_data import IBHistoricalDataDownloader
from src.data.futures_contract_downloader import FuturesContractDownloader

# 配置日志 - 保存到logs文件夹
log_dir = project_root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'mes_download.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """主函数：下载MES期货主连合约数据"""
    
    # 查询参数：不做时区转换，直接使用naive datetime
    # 但返回的数据时间戳会转换为美东时区（US/Eastern）
    start_date = datetime(2023, 12, 1, 0, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59, 59)
    
    logger.info("=" * 80)
    logger.info("MES期货主连合约数据下载")
    logger.info("=" * 80)
    logger.info(f"合约代码: MES (Micro E-mini S&P 500)")
    logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"返回数据时区: 美东时区 (US/Eastern)")
    logger.info(f"K线周期: 5分钟")
    logger.info(f"交易时段: 仅常规交易时间 (RTH)")
    logger.info(f"拼接方法: 比例调整法 (ratio)")
    logger.info("=" * 80)
    
    # IB Gateway连接参数
    # 4001 = TWS Paper Trading
    # 4002 = TWS Live Trading
    # 4003 = IB Gateway Paper Trading
    # 4004 = IB Gateway Live Trading
    host = '127.0.0.1'
    port = 4001  # 默认使用Paper Trading端口
    
    try:
        # 创建输出目录 - 保存到data/raw文件夹
        output_dir = project_root / 'data' / 'raw' / 'mes_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 连接到IB Gateway
        logger.info(f"\n正在连接到IB Gateway: {host}:{port}")
        connector = IBConnector(host=host, port=port, client_id=1)
        
        if not connector.connect():
            logger.error("无法连接到IB Gateway，请确保:")
            logger.error("1. IB Gateway或TWS已启动")
            logger.error("2. API连接已启用 (配置 -> API -> 启用ActiveX和Socket客户端)")
            logger.error("3. 端口号正确 (默认4001为Paper Trading)")
            return
        
        # 获取服务器时间（验证连接）
        server_time = connector.get_current_time()
        if server_time:
            logger.info(f"IB服务器时间: {server_time}")
        
        # 创建下载器
        downloader = IBHistoricalDataDownloader(connector)
        contract_downloader = FuturesContractDownloader(connector, downloader)
        
        # 下载MES期货数据
        # MES = Micro E-mini S&P 500 期货
        # 交易所: CME (Chicago Mercantile Exchange)
        # 合约乘数: $5 per point (标准ES是$50)
        logger.info("\n开始下载MES期货数据...")
        
        df = contract_downloader.download_contracts_range(
            symbol='MES',
            start_date=start_date,
            end_date=end_date,
            exchange='CME',
            bar_size='5m',
            stitch_method='ratio',  # 使用比例调整法拼接合约
            save_dir=str(output_dir),
            max_future_months=3,  # 最多下载未来3个月的合约
            use_rth=True,  # 只下载常规交易时间数据
            target_tz='US/Eastern'  # 将返回数据的时间戳转换为美东时区
        )
        
        if df is not None and not df.empty:
            logger.info("\n" + "=" * 80)
            logger.info("下载成功！")
            logger.info("=" * 80)
            logger.info(f"总记录数: {len(df):,}")
            logger.info(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"数据列: {list(df.columns)}")
            logger.info("\n数据预览:")
            logger.info(df.head(10).to_string())
            logger.info("\n数据统计:")
            logger.info(df.describe().to_string())
            
            # 保存最终拼接数据
            final_output = output_dir / 'MES_stitched_5m_ratio.csv'
            logger.info(f"\n最终数据已保存到: {final_output}")
            
            # 数据质量检查
            logger.info("\n" + "=" * 80)
            logger.info("数据质量检查")
            logger.info("=" * 80)
            
            # 检查缺失值
            missing = df.isnull().sum()
            if missing.any():
                logger.warning("发现缺失值:")
                logger.warning(missing[missing > 0].to_string())
            else:
                logger.info("✓ 无缺失值")
            
            # 检查重复日期
            duplicates = df['date'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"发现 {duplicates} 个重复日期")
            else:
                logger.info("✓ 无重复日期")
            
            # 检查价格异常
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    if (df[col] <= 0).any():
                        logger.warning(f"{col} 列存在非正值")
                    else:
                        logger.info(f"✓ {col} 列价格正常")
            
            logger.info("\n" + "=" * 80)
            logger.info("下载完成！")
            logger.info("=" * 80)
            
        else:
            logger.error("下载失败或未获取到数据")
            logger.error("可能的原因:")
            logger.error("1. 合约代码不正确")
            logger.error("2. 日期范围超出可用数据范围")
            logger.error("3. IB账户权限不足")
            logger.error("4. 市场数据订阅未激活")
        
        # 断开连接
        connector.disconnect()
        
    except KeyboardInterrupt:
        logger.info("\n用户中断下载")
    except Exception as e:
        logger.error(f"\n下载过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("\n程序结束")


if __name__ == '__main__':
    main()