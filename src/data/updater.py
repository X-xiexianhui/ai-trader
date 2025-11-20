"""
数据更新器模块
Data Updater Module

实现增量数据更新功能，智能检测和下载新数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.data.downloader import YahooFinanceDownloader
from src.data.cache import DataCache

logger = get_logger(__name__)


class DataUpdater:
    """
    数据增量更新器
    
    功能:
    - 检测本地最新数据时间戳
    - 仅下载新增数据
    - 合并到现有数据集
    - 避免重复数据
    - 自动缓存更新
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化更新器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 初始化下载器和缓存
        self.downloader = YahooFinanceDownloader(config_path)
        self.cache = DataCache(config_path)
        
        # 增量更新配置
        self.incremental_enabled = self.config.get('cache.incremental_update.enabled', True)
        self.check_interval = self.config.get('cache.incremental_update.check_interval', 3600)
        
        logger.info("数据更新器初始化完成")
    
    def update(
        self,
        symbol: str,
        interval: str = "5m",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        force: bool = False
    ) -> pd.DataFrame:
        """
        更新指定品种的数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            start: 开始日期（仅在force=True或无缓存时使用）
            end: 结束日期，默认为当前时间
            force: 是否强制全量下载
        
        Returns:
            pd.DataFrame: 更新后的完整数据
        """
        logger.info(f"开始更新数据: {symbol} ({interval})")
        
        # 如果强制更新或缓存不存在，执行全量下载
        if force or not self.cache.exists(symbol, interval):
            return self._full_update(symbol, interval, start, end)
        
        # 否则执行增量更新
        if self.incremental_enabled:
            return self._incremental_update(symbol, interval, end)
        else:
            # 如果禁用增量更新，直接返回缓存数据
            return self.cache.load(symbol, interval, check_expiry=False)
    
    def update_multiple(
        self,
        symbols: List[str],
        interval: str = "5m",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        force: bool = False,
        delay: float = 1.0
    ) -> Dict[str, pd.DataFrame]:
        """
        批量更新多个品种的数据
        
        Args:
            symbols: 品种代码列表
            interval: 数据频率
            start: 开始日期
            end: 结束日期
            force: 是否强制全量下载
            delay: 品种间延迟时间(秒)
        
        Returns:
            Dict[str, pd.DataFrame]: 品种代码到数据的映射
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"开始批量更新 {total} 个品种")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"更新进度: {i}/{total} - {symbol}")
                
                data = self.update(
                    symbol=symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    force=force
                )
                
                if data is not None and not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"品种 {symbol} 更新后数据为空")
                
                # 品种间延迟
                if i < total and delay > 0:
                    import time
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"更新品种 {symbol} 失败: {str(e)}")
                continue
        
        logger.info(f"批量更新完成，成功: {len(results)}/{total}")
        return results
    
    def _full_update(
        self,
        symbol: str,
        interval: str,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]]
    ) -> pd.DataFrame:
        """
        全量更新数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            start: 开始日期
            end: 结束日期
        
        Returns:
            pd.DataFrame: 下载的数据
        """
        logger.info(f"执行全量更新: {symbol} ({interval})")
        
        # 如果没有指定开始日期，使用默认配置
        if start is None:
            start = self.config.get('time_range.default.start_date', '2020-01-01')
        
        # 下载数据
        data = self.downloader.download(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval
        )
        
        if data is not None and not data.empty:
            # 保存到缓存
            self.cache.save(data, symbol, interval)
            logger.info(f"全量更新完成: {symbol}, {len(data)} 条记录")
        else:
            logger.warning(f"全量更新失败: {symbol}, 数据为空")
        
        return data
    
    def _incremental_update(
        self,
        symbol: str,
        interval: str,
        end: Optional[Union[str, datetime]]
    ) -> pd.DataFrame:
        """
        增量更新数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            end: 结束日期
        
        Returns:
            pd.DataFrame: 更新后的完整数据
        """
        logger.info(f"执行增量更新: {symbol} ({interval})")
        
        # 加载现有数据
        existing_data = self.cache.load(symbol, interval, check_expiry=False)
        
        if existing_data is None or existing_data.empty:
            logger.warning(f"缓存数据为空，转为全量更新: {symbol}")
            return self._full_update(symbol, interval, None, end)
        
        # 获取最新数据时间戳
        if 'datetime' not in existing_data.columns:
            logger.error(f"数据缺少datetime列: {symbol}")
            return existing_data
        
        last_timestamp = existing_data['datetime'].max()
        logger.info(f"本地最新数据时间: {last_timestamp}")
        
        # 计算需要下载的时间范围
        # 从最后一个时间戳的下一个时间点开始
        start_download = last_timestamp + self._get_interval_timedelta(interval)
        
        # 如果没有指定结束时间，使用当前时间
        if end is None:
            end = datetime.now()
        elif isinstance(end, str):
            end = pd.to_datetime(end)
        
        # 检查是否需要下载新数据
        if start_download >= end:
            logger.info(f"数据已是最新，无需更新: {symbol}")
            return existing_data
        
        logger.info(f"下载新数据: {start_download} 到 {end}")
        
        # 下载新数据
        new_data = self.downloader.download(
            symbol=symbol,
            start=start_download,
            end=end,
            interval=interval
        )
        
        if new_data is None or new_data.empty:
            logger.info(f"没有新数据: {symbol}")
            return existing_data
        
        logger.info(f"下载到 {len(new_data)} 条新数据")
        
        # 合并数据
        merged_data = self._merge_data(existing_data, new_data)
        
        # 保存更新后的数据
        self.cache.save(merged_data, symbol, interval)
        
        logger.info(f"增量更新完成: {symbol}, 总计 {len(merged_data)} 条记录")
        return merged_data
    
    def _merge_data(
        self,
        existing_data: pd.DataFrame,
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        合并现有数据和新数据
        
        Args:
            existing_data: 现有数据
            new_data: 新数据
        
        Returns:
            pd.DataFrame: 合并后的数据
        """
        # 确保两个数据集有相同的列
        common_columns = list(set(existing_data.columns) & set(new_data.columns))
        
        existing_data = existing_data[common_columns]
        new_data = new_data[common_columns]
        
        # 合并数据
        merged = pd.concat([existing_data, new_data], ignore_index=True)
        
        # 按时间排序
        if 'datetime' in merged.columns:
            merged = merged.sort_values('datetime')
        
        # 删除重复行（基于datetime）
        if 'datetime' in merged.columns:
            merged = merged.drop_duplicates(subset=['datetime'], keep='last')
        
        # 重置索引
        merged = merged.reset_index(drop=True)
        
        logger.debug(f"数据合并完成: {len(existing_data)} + {len(new_data)} = {len(merged)}")
        
        return merged
    
    def _get_interval_timedelta(self, interval: str) -> timedelta:
        """
        根据频率字符串获取时间增量
        
        Args:
            interval: 频率字符串 (如 "5m", "1h", "1d")
        
        Returns:
            timedelta: 时间增量
        """
        interval_map = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '1d': timedelta(days=1),
        }
        
        return interval_map.get(interval, timedelta(minutes=5))
    
    def get_update_status(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        获取数据更新状态
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            Dict: 更新状态信息
        """
        status = {
            'symbol': symbol,
            'interval': interval,
            'cached': self.cache.exists(symbol, interval),
            'last_update': None,
            'data_range': None,
            'record_count': 0,
            'needs_update': False
        }
        
        # 获取缓存元数据
        metadata = self.cache.get_metadata(symbol, interval)
        if metadata:
            status['last_update'] = metadata.get('created_at')
            status['data_range'] = {
                'start': metadata.get('start_date'),
                'end': metadata.get('end_date')
            }
            status['record_count'] = metadata.get('rows', 0)
        
        # 检查是否需要更新
        if status['cached']:
            data = self.cache.load(symbol, interval, check_expiry=False)
            if data is not None and not data.empty and 'datetime' in data.columns:
                last_timestamp = data['datetime'].max()
                current_time = datetime.now()
                
                # 如果最后数据时间距离现在超过检查间隔，则需要更新
                time_diff = (current_time - last_timestamp).total_seconds()
                status['needs_update'] = time_diff > self.check_interval
                status['time_since_last_data'] = time_diff
        else:
            status['needs_update'] = True
        
        return status
    
    def check_all_updates(self) -> List[Dict[str, Any]]:
        """
        检查所有已缓存品种的更新状态
        
        Returns:
            List[Dict]: 所有品种的更新状态列表
        """
        cached_symbols = self.cache.list_cached_symbols()
        statuses = []
        
        for item in cached_symbols:
            status = self.get_update_status(item['symbol'], item['interval'])
            statuses.append(status)
        
        return statuses


def main():
    """测试函数"""
    # 创建更新器
    updater = DataUpdater()
    
    # 测试全量更新
    print("\n=== 测试全量更新 ===")
    try:
        data = updater.update(
            symbol="AAPL",
            interval="1d",
            start="2024-01-01",
            end="2024-01-31",
            force=True
        )
        print(f"全量更新成功: {len(data)} 条记录")
        print(f"数据范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
    except Exception as e:
        print(f"全量更新失败: {str(e)}")
    
    # 测试增量更新
    print("\n=== 测试增量更新 ===")
    try:
        data = updater.update(
            symbol="AAPL",
            interval="1d",
            end="2024-02-15",
            force=False
        )
        print(f"增量更新成功: {len(data)} 条记录")
        print(f"数据范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
    except Exception as e:
        print(f"增量更新失败: {str(e)}")
    
    # 测试更新状态
    print("\n=== 测试更新状态 ===")
    status = updater.get_update_status("AAPL", "1d")
    print(f"更新状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 测试批量更新
    print("\n=== 测试批量更新 ===")
    symbols = ["AAPL", "MSFT"]
    results = updater.update_multiple(
        symbols=symbols,
        interval="1d",
        start="2024-01-01",
        end="2024-01-31",
        force=True,
        delay=1.0
    )
    print(f"批量更新完成: {len(results)} 个品种")
    for symbol, data in results.items():
        print(f"  {symbol}: {len(data)} 条记录")


if __name__ == "__main__":
    main()