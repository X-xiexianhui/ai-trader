"""
回测系统演示脚本
展示如何使用数据下载、存储和回测功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data.downloader import DataDownloader, IncrementalUpdater
from src.data.storage import DataStorage
from src.backtest.engine import BacktestEngine
from src.backtest.strategy import PPOStrategy
from src.backtest.recorder import BacktestRecorder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def demo_data_download():
    """演示数据下载功能"""
    logger.info("=" * 50)
    logger.info("演示1: 数据下载")
    logger.info("=" * 50)
    
    # 创建下载器
    downloader = DataDownloader(max_retries=3, retry_delay=5)
    
    # 下载单个品种
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"下载 {symbol} 最近30天的5分钟数据...")
    data = downloader.download(
        symbol=symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='5m'
    )
    
    if data is not None:
        logger.info(f"下载成功！数据形状: {data.shape}")
        logger.info(f"数据预览:\n{data.head()}")
        return data
    else:
        logger.error("下载失败")
        return None


def demo_data_storage(data):
    """演示数据存储功能"""
    logger.info("\n" + "=" * 50)
    logger.info("演示2: 数据存储")
    logger.info("=" * 50)
    
    # 创建存储管理器
    storage = DataStorage(base_path='data/raw')
    
    # 保存为Parquet格式
    logger.info("保存数据为Parquet格式...")
    success = storage.save_parquet(data, 'AAPL', compression='snappy')
    
    if success:
        logger.info("保存成功！")
        
        # 加载数据
        logger.info("从Parquet加载数据...")
        loaded_data = storage.load_parquet('AAPL')
        
        if loaded_data is not None:
            logger.info(f"加载成功！数据形状: {loaded_data.shape}")
            
            # 验证数据一致性
            if data.equals(loaded_data):
                logger.info("数据一致性验证通过！")
            else:
                logger.warning("数据不一致")
        
        # 获取文件信息
        info = storage.get_file_info('AAPL', format='parquet')
        if info:
            logger.info(f"文件信息: {info}")
    
    return loaded_data


def demo_incremental_update(data):
    """演示增量更新功能"""
    logger.info("\n" + "=" * 50)
    logger.info("演示3: 增量更新")
    logger.info("=" * 50)
    
    # 创建下载器和更新器
    downloader = DataDownloader()
    updater = IncrementalUpdater(downloader)
    
    # 模拟增量更新
    logger.info("执行增量更新...")
    updated_data, new_records = updater.update('AAPL', data, interval='5m')
    
    logger.info(f"更新完成！新增 {new_records} 条记录")
    logger.info(f"更新后数据形状: {updated_data.shape}")
    
    return updated_data


def demo_backtest(data):
    """演示回测功能"""
    logger.info("\n" + "=" * 50)
    logger.info("演示4: 回测系统")
    logger.info("=" * 50)
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_cash=100000.0,
        commission=0.001,
        slippage=0.0005
    )
    
    # 添加数据
    logger.info("添加回测数据...")
    engine.add_data(data, name='AAPL')
    
    # 添加策略
    logger.info("添加交易策略...")
    engine.add_strategy(PPOStrategy, verbose=True)
    
    # 运行回测
    logger.info("开始回测...")
    results = engine.run()
    
    # 获取结果
    if results:
        strategy = results[0]
        backtest_results = engine.get_results(strategy)
        
        logger.info("\n回测结果:")
        logger.info(f"初始资金: ${backtest_results['initial_cash']:,.2f}")
        logger.info(f"最终资金: ${backtest_results['final_value']:,.2f}")
        logger.info(f"总收益率: {backtest_results['total_return']:.2f}%")
        logger.info(f"夏普比率: {backtest_results.get('sharpe_ratio', 'N/A')}")
        logger.info(f"最大回撤: {backtest_results.get('max_drawdown', 'N/A')}")
        logger.info(f"总交易次数: {backtest_results.get('total_trades', 0)}")
        logger.info(f"胜率: {backtest_results.get('win_rate', 0):.2f}%")
        logger.info(f"盈亏比: {backtest_results.get('profit_factor', 0):.2f}")
        
        return strategy, backtest_results
    
    return None, None


def demo_recorder(strategy):
    """演示结果记录功能"""
    logger.info("\n" + "=" * 50)
    logger.info("演示5: 结果记录")
    logger.info("=" * 50)
    
    # 创建记录器
    recorder = BacktestRecorder(output_dir='results/backtest')
    
    # 从策略中提取交易记录
    if hasattr(strategy, 'trades') and strategy.trades:
        for trade in strategy.trades:
            recorder.record_trade(trade)
        
        logger.info(f"记录了 {len(strategy.trades)} 笔交易")
        
        # 生成完整报告
        logger.info("生成完整报告...")
        files = recorder.generate_full_report()
        
        logger.info("\n生成的文件:")
        for file_type, filepath in files.items():
            if filepath:
                logger.info(f"  {file_type}: {filepath}")
    else:
        logger.warning("没有交易记录可供记录")


def main():
    """主函数"""
    logger.info("开始模块5功能演示")
    logger.info("=" * 50)
    
    try:
        # 1. 数据下载
        data = demo_data_download()
        
        if data is None:
            logger.error("数据下载失败，使用模拟数据")
            # 生成模拟数据
            dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
            data = pd.DataFrame({
                'open': np.random.randn(1000).cumsum() + 100,
                'high': np.random.randn(1000).cumsum() + 102,
                'low': np.random.randn(1000).cumsum() + 98,
                'close': np.random.randn(1000).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 1000)
            }, index=dates)
            
            # 确保OHLC一致性
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # 2. 数据存储
        loaded_data = demo_data_storage(data)
        
        # 3. 增量更新（可选）
        # updated_data = demo_incremental_update(loaded_data)
        
        # 4. 回测
        strategy, results = demo_backtest(data)
        
        # 5. 结果记录
        if strategy:
            demo_recorder(strategy)
        
        logger.info("\n" + "=" * 50)
        logger.info("演示完成！")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"演示过程中出错: {str(e)}", exc_info=True)


if __name__ == '__main__':
    main()