"""
数据下载管理模块
实现基于yfinance的数据下载器，支持多品种、5分钟K线数据下载
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class DataDownloader:
    """
    yfinance数据下载器
    
    功能：
    1. 支持多品种下载
    2. 支持5分钟K线
    3. 支持日期范围指定
    4. 错误处理和重试
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 5,
        timeout: int = 30
    ):
        """
        初始化数据下载器
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            timeout: 请求超时时间（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '5m'
    ) -> Optional[pd.DataFrame]:
        """
        下载单个品种的数据
        
        Args:
            symbol: 品种代码（如'AAPL', 'ES=F'）
            start_date: 开始日期（格式：'YYYY-MM-DD'）
            end_date: 结束日期（格式：'YYYY-MM-DD'）
            interval: K线周期（'5m', '15m', '1h', '1d'等）
            
        Returns:
            包含OHLCV数据的DataFrame，失败返回None
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"下载 {symbol} 数据，尝试 {attempt + 1}/{self.max_retries}")
                
                # 下载数据
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    timeout=self.timeout
                )
                
                if df.empty:
                    logger.warning(f"{symbol} 返回空数据")
                    return None
                
                # 标准化列名
                df = self._standardize_columns(df)
                
                # 验证数据
                if self._validate_data(df):
                    logger.info(f"成功下载 {symbol} 数据，共 {len(df)} 条记录")
                    return df
                else:
                    logger.warning(f"{symbol} 数据验证失败")
                    return None
                    
            except Exception as e:
                logger.error(f"下载 {symbol} 失败（尝试 {attempt + 1}/{self.max_retries}）: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"{symbol} 下载失败，已达最大重试次数")
                    return None
        
        return None
    
    def download_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '5m'
    ) -> Dict[str, pd.DataFrame]:
        """
        下载多个品种的数据
        
        Args:
            symbols: 品种代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: K线周期
            
        Returns:
            字典，键为品种代码，值为DataFrame
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"开始下载 {symbol}")
            df = self.download(symbol, start_date, end_date, interval)
            
            if df is not None:
                results[symbol] = df
            else:
                logger.warning(f"跳过 {symbol}，下载失败")
            
            # 避免请求过快
            time.sleep(1)
        
        logger.info(f"完成下载，成功 {len(results)}/{len(symbols)} 个品种")
        return results
    
    def download_recent(
        self,
        symbol: str,
        days: int = 7,
        interval: str = '5m'
    ) -> Optional[pd.DataFrame]:
        """
        下载最近N天的数据
        
        Args:
            symbol: 品种代码
            days: 天数
            interval: K线周期
            
        Returns:
            DataFrame或None
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.download(
            symbol,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval
        )
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化列名
        
        Args:
            df: 原始DataFrame
            
        Returns:
            标准化后的DataFrame
        """
        # yfinance返回的列名可能是大写或小写
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 只保留需要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df = df[required_columns]
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据质量
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            验证是否通过
        """
        if df.empty:
            return False
        
        # 检查必需列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error("缺少必需列")
            return False
        
        # 检查缺失值比例
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.1:
            logger.warning(f"缺失值比例过高: {missing_ratio:.2%}")
            return False
        
        # 检查OHLC一致性
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        if invalid_ohlc > 0:
            logger.warning(f"发现 {invalid_ohlc} 条OHLC不一致的记录")
            return False
        
        # 检查负值
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error("发现负值或零值价格")
            return False
        
        return True


class IncrementalUpdater:
    """
    数据增量更新器
    
    功能：
    1. 检测最新数据时间
    2. 仅下载新数据
    3. 合并到现有数据
    4. 去重处理
    """
    
    def __init__(self, downloader: DataDownloader):
        """
        初始化增量更新器
        
        Args:
            downloader: 数据下载器实例
        """
        self.downloader = downloader
    
    def update(
        self,
        symbol: str,
        existing_data: pd.DataFrame,
        interval: str = '5m'
    ) -> Tuple[pd.DataFrame, int]:
        """
        增量更新数据
        
        Args:
            symbol: 品种代码
            existing_data: 现有数据
            interval: K线周期
            
        Returns:
            (更新后的数据, 新增记录数)
        """
        if existing_data.empty:
            logger.info(f"{symbol} 无现有数据，执行全量下载")
            # 下载最近30天数据
            new_data = self.downloader.download_recent(symbol, days=30, interval=interval)
            if new_data is not None:
                return new_data, len(new_data)
            else:
                return existing_data, 0
        
        # 获取最新数据时间
        last_time = existing_data.index[-1]
        logger.info(f"{symbol} 最新数据时间: {last_time}")
        
        # 计算需要更新的时间范围
        start_date = (last_time + timedelta(minutes=5)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 下载新数据
        new_data = self.downloader.download(symbol, start_date, end_date, interval)
        
        if new_data is None or new_data.empty:
            logger.info(f"{symbol} 无新数据")
            return existing_data, 0
        
        # 合并数据
        combined_data = pd.concat([existing_data, new_data])
        
        # 去重（保留最后出现的）
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # 排序
        combined_data = combined_data.sort_index()
        
        new_records = len(combined_data) - len(existing_data)
        logger.info(f"{symbol} 新增 {new_records} 条记录")
        
        return combined_data, new_records
    
    def update_multiple(
        self,
        data_dict: Dict[str, pd.DataFrame],
        interval: str = '5m'
    ) -> Dict[str, Tuple[pd.DataFrame, int]]:
        """
        批量增量更新
        
        Args:
            data_dict: 品种代码到DataFrame的字典
            interval: K线周期
            
        Returns:
            字典，键为品种代码，值为(更新后数据, 新增记录数)
        """
        results = {}
        
        for symbol, existing_data in data_dict.items():
            logger.info(f"更新 {symbol}")
            updated_data, new_records = self.update(symbol, existing_data, interval)
            results[symbol] = (updated_data, new_records)
            
            # 避免请求过快
            time.sleep(1)
        
        total_new = sum(count for _, count in results.values())
        logger.info(f"批量更新完成，共新增 {total_new} 条记录")
        
        return results