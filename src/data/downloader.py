"""
数据下载器模块
Data Downloader Module

实现基于yfinance的数据下载功能，支持多种数据源和错误处理
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict, Any
import time
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.helpers import get_timestamp

logger = get_logger(__name__)


class YahooFinanceDownloader:
    """
    Yahoo Finance数据下载器
    
    功能:
    - 下载OHLCV数据
    - 支持多种时间频率
    - 错误处理和重试机制
    - 速率限制
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化下载器
        
        Args:
            config_path: 配置文件路径，默认使用data_config.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 获取Yahoo Finance配置
        self.timeout = self.config.get('data_sources.yahoo_finance.api.timeout', 30)
        self.max_retries = self.config.get('data_sources.yahoo_finance.api.max_retries', 3)
        self.retry_delay = self.config.get('data_sources.yahoo_finance.api.retry_delay', 5)
        
        # 速率限制配置
        self.rate_limit_rpm = self.config.get('data_sources.yahoo_finance.rate_limit.requests_per_minute', 60)
        self.rate_limit_rph = self.config.get('data_sources.yahoo_finance.rate_limit.requests_per_hour', 2000)
        
        # 频率映射
        self.freq_mapping = self.config.get('frequency.yahoo_mapping', {})
        
        # 请求计数器
        self._request_count_minute = 0
        self._request_count_hour = 0
        self._last_request_time = None
        self._minute_start_time = None
        self._hour_start_time = None
        
        logger.info("Yahoo Finance下载器初始化完成")
    
    def download(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        interval: str = "5m",
        **kwargs
    ) -> pd.DataFrame:
        """
        下载指定品种的OHLCV数据
        
        Args:
            symbol: 品种代码 (如 "AAPL", "GC=F")
            start: 开始日期 (格式: "YYYY-MM-DD" 或 datetime对象)
            end: 结束日期 (格式: "YYYY-MM-DD" 或 datetime对象)，默认为当前日期
            interval: 数据频率 ("1m", "5m", "15m", "30m", "1h", "1d")
            **kwargs: 其他yfinance参数
        
        Returns:
            pd.DataFrame: OHLCV数据，包含列: Open, High, Low, Close, Volume
        
        Raises:
            ValueError: 参数错误
            RuntimeError: 下载失败
        """
        # 参数验证
        if not symbol:
            raise ValueError("品种代码不能为空")
        
        # 转换日期格式
        if isinstance(start, str):
            start = pd.to_datetime(start)
        if end is None:
            end = datetime.now()
        elif isinstance(end, str):
            end = pd.to_datetime(end)
        
        # 验证日期范围
        if start >= end:
            raise ValueError(f"开始日期({start})必须早于结束日期({end})")
        
        # 映射频率
        yf_interval = self.freq_mapping.get(interval, interval)
        
        logger.info(f"开始下载数据: {symbol}, {start} 到 {end}, 频率: {yf_interval}")
        
        # 带重试的下载
        for attempt in range(self.max_retries):
            try:
                # 检查速率限制
                self._check_rate_limit()
                
                # 下载数据
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start,
                    end=end,
                    interval=yf_interval,
                    timeout=self.timeout,
                    **kwargs
                )
                
                # 更新请求计数
                self._update_request_count()
                
                # 验证数据
                if data.empty:
                    logger.warning(f"下载的数据为空: {symbol}")
                    return pd.DataFrame()
                
                # 数据清理
                data = self._clean_data(data, symbol)
                
                logger.info(f"成功下载 {len(data)} 条数据: {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # 等待后重试
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    error_msg = f"下载失败，已达到最大重试次数: {symbol}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
        
        return pd.DataFrame()
    
    def download_multiple(
        self,
        symbols: List[str],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        interval: str = "5m",
        delay: float = 1.0,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        批量下载多个品种的数据
        
        Args:
            symbols: 品种代码列表
            start: 开始日期
            end: 结束日期
            interval: 数据频率
            delay: 品种间延迟时间(秒)
            **kwargs: 其他yfinance参数
        
        Returns:
            Dict[str, pd.DataFrame]: 品种代码到数据的映射
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"开始批量下载 {total} 个品种的数据")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"下载进度: {i}/{total} - {symbol}")
                
                data = self.download(
                    symbol=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    **kwargs
                )
                
                if not data.empty:
                    results[symbol] = data
                else:
                    logger.warning(f"品种 {symbol} 数据为空，跳过")
                
                # 品种间延迟
                if i < total and delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"下载品种 {symbol} 失败: {str(e)}")
                continue
        
        logger.info(f"批量下载完成，成功: {len(results)}/{total}")
        return results
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取品种信息
        
        Args:
            symbol: 品种代码
        
        Returns:
            Dict: 品种信息字典
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 提取关键信息
            symbol_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'exchange': info.get('exchange', 'Unknown'),
                'currency': info.get('currency', 'USD'),
                'timezone': info.get('timeZoneFullName', 'UTC'),
                'market_cap': info.get('marketCap', None),
                'sector': info.get('sector', None),
                'industry': info.get('industry', None),
            }
            
            logger.info(f"获取品种信息成功: {symbol}")
            return symbol_info
            
        except Exception as e:
            logger.error(f"获取品种信息失败: {symbol}, {str(e)}")
            return {'symbol': symbol, 'error': str(e)}
    
    def _clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        清理下载的数据
        
        Args:
            data: 原始数据
            symbol: 品种代码
        
        Returns:
            pd.DataFrame: 清理后的数据
        """
        # 复制数据
        df = data.copy()
        
        # 重置索引，将时间作为列
        df = df.reset_index()
        
        # 重命名列
        column_mapping = {
            'Date': 'datetime',
            'Datetime': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # 选择需要的列
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        
        # 添加品种代码
        df['symbol'] = symbol
        
        # 确保datetime列是datetime类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 按时间排序
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # 删除重复行
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # 删除全为NaN的行
        df = df.dropna(how='all', subset=['open', 'high', 'low', 'close'])
        
        logger.debug(f"数据清理完成: {symbol}, 剩余 {len(df)} 条记录")
        
        return df
    
    def _check_rate_limit(self):
        """检查速率限制"""
        current_time = time.time()
        
        # 初始化时间戳
        if self._minute_start_time is None:
            self._minute_start_time = current_time
            self._hour_start_time = current_time
        
        # 检查分钟级限制
        if current_time - self._minute_start_time >= 60:
            # 重置分钟计数器
            self._request_count_minute = 0
            self._minute_start_time = current_time
        
        # 检查小时级限制
        if current_time - self._hour_start_time >= 3600:
            # 重置小时计数器
            self._request_count_hour = 0
            self._hour_start_time = current_time
        
        # 检查是否超过限制
        if self._request_count_minute >= self.rate_limit_rpm:
            wait_time = 60 - (current_time - self._minute_start_time)
            if wait_time > 0:
                logger.warning(f"达到分钟级速率限制，等待 {wait_time:.1f} 秒")
                time.sleep(wait_time)
                self._request_count_minute = 0
                self._minute_start_time = time.time()
        
        if self._request_count_hour >= self.rate_limit_rph:
            wait_time = 3600 - (current_time - self._hour_start_time)
            if wait_time > 0:
                logger.warning(f"达到小时级速率限制，等待 {wait_time:.1f} 秒")
                time.sleep(wait_time)
                self._request_count_hour = 0
                self._hour_start_time = time.time()
    
    def _update_request_count(self):
        """更新请求计数"""
        self._request_count_minute += 1
        self._request_count_hour += 1
        self._last_request_time = time.time()


def main():
    """测试函数"""
    # 创建下载器
    downloader = YahooFinanceDownloader()
    
    # 测试单个品种下载
    print("\n=== 测试单个品种下载 ===")
    try:
        data = downloader.download(
            symbol="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            interval="1d"
        )
        print(f"下载成功: {len(data)} 条记录")
        print(data.head())
        print(f"\n数据列: {data.columns.tolist()}")
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"下载失败: {str(e)}")
    
    # 测试品种信息获取
    print("\n=== 测试品种信息获取 ===")
    info = downloader.get_symbol_info("AAPL")
    print(f"品种信息: {info}")
    
    # 测试批量下载
    print("\n=== 测试批量下载 ===")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = downloader.download_multiple(
        symbols=symbols,
        start="2024-01-01",
        end="2024-01-31",
        interval="1d",
        delay=1.0
    )
    print(f"批量下载完成: {len(results)} 个品种")
    for symbol, data in results.items():
        print(f"  {symbol}: {len(data)} 条记录")


if __name__ == "__main__":
    main()