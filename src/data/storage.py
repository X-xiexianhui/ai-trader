"""
数据存储模块
实现高效的数据存储方案，支持Parquet和HDF5格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import logging
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataStorage:
    """
    数据存储管理器
    
    支持格式：
    1. Parquet：高压缩比，快速读取
    2. HDF5：支持增量写入
    """
    
    def __init__(self, base_path: str = 'data/raw'):
        """
        初始化存储管理器
        
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_parquet(
        self,
        data: pd.DataFrame,
        symbol: str,
        compression: str = 'snappy'
    ) -> bool:
        """
        保存数据为Parquet格式
        
        Args:
            data: 待保存的DataFrame
            symbol: 品种代码
            compression: 压缩算法（'snappy', 'gzip', 'brotli', 'zstd'）
            
        Returns:
            是否保存成功
        """
        try:
            file_path = self.base_path / f"{symbol}.parquet"
            
            # 保存为Parquet
            data.to_parquet(
                file_path,
                engine='pyarrow',
                compression=compression,
                index=True
            )
            
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"保存 {symbol} 到 {file_path}，大小: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"保存 {symbol} 失败: {str(e)}")
            return False
    
    def load_parquet(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从Parquet文件加载数据
        
        Args:
            symbol: 品种代码
            
        Returns:
            DataFrame或None
        """
        try:
            file_path = self.base_path / f"{symbol}.parquet"
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return None
            
            # 读取Parquet
            data = pd.read_parquet(file_path, engine='pyarrow')
            
            logger.info(f"加载 {symbol}，共 {len(data)} 条记录")
            
            return data
            
        except Exception as e:
            logger.error(f"加载 {symbol} 失败: {str(e)}")
            return None
    
    def save_hdf5(
        self,
        data: pd.DataFrame,
        symbol: str,
        key: str = 'data',
        mode: str = 'w',
        complevel: int = 9
    ) -> bool:
        """
        保存数据为HDF5格式
        
        Args:
            data: 待保存的DataFrame
            symbol: 品种代码
            key: HDF5中的键名
            mode: 写入模式（'w'=覆盖, 'a'=追加）
            complevel: 压缩级别（0-9）
            
        Returns:
            是否保存成功
        """
        try:
            file_path = self.base_path / f"{symbol}.h5"
            
            # 保存为HDF5
            data.to_hdf(
                file_path,
                key=key,
                mode=mode,
                complevel=complevel,
                complib='blosc'
            )
            
            file_size = file_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"保存 {symbol} 到 {file_path}，大小: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"保存 {symbol} 失败: {str(e)}")
            return False
    
    def load_hdf5(
        self,
        symbol: str,
        key: str = 'data'
    ) -> Optional[pd.DataFrame]:
        """
        从HDF5文件加载数据
        
        Args:
            symbol: 品种代码
            key: HDF5中的键名
            
        Returns:
            DataFrame或None
        """
        try:
            file_path = self.base_path / f"{symbol}.h5"
            
            if not file_path.exists():
                logger.warning(f"文件不存在: {file_path}")
                return None
            
            # 读取HDF5
            data = pd.read_hdf(file_path, key=key)
            
            logger.info(f"加载 {symbol}，共 {len(data)} 条记录")
            
            return data
            
        except Exception as e:
            logger.error(f"加载 {symbol} 失败: {str(e)}")
            return None
    
    def append_hdf5(
        self,
        data: pd.DataFrame,
        symbol: str,
        key: str = 'data'
    ) -> bool:
        """
        追加数据到HDF5文件
        
        Args:
            data: 待追加的DataFrame
            symbol: 品种代码
            key: HDF5中的键名
            
        Returns:
            是否追加成功
        """
        try:
            file_path = self.base_path / f"{symbol}.h5"
            
            if not file_path.exists():
                # 文件不存在，直接保存
                return self.save_hdf5(data, symbol, key, mode='w')
            
            # 追加数据
            data.to_hdf(
                file_path,
                key=key,
                mode='a',
                append=True,
                complevel=9,
                complib='blosc'
            )
            
            logger.info(f"追加 {len(data)} 条记录到 {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"追加 {symbol} 失败: {str(e)}")
            return False
    
    def save_multiple_parquet(
        self,
        data_dict: Dict[str, pd.DataFrame],
        compression: str = 'snappy'
    ) -> Dict[str, bool]:
        """
        批量保存为Parquet格式
        
        Args:
            data_dict: 品种代码到DataFrame的字典
            compression: 压缩算法
            
        Returns:
            品种代码到保存结果的字典
        """
        results = {}
        
        for symbol, data in data_dict.items():
            results[symbol] = self.save_parquet(data, symbol, compression)
        
        success_count = sum(results.values())
        logger.info(f"批量保存完成，成功 {success_count}/{len(data_dict)} 个品种")
        
        return results
    
    def load_multiple_parquet(
        self,
        symbols: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载Parquet文件
        
        Args:
            symbols: 品种代码列表
            
        Returns:
            品种代码到DataFrame的字典
        """
        results = {}
        
        for symbol in symbols:
            data = self.load_parquet(symbol)
            if data is not None:
                results[symbol] = data
        
        logger.info(f"批量加载完成，成功 {len(results)}/{len(symbols)} 个品种")
        
        return results
    
    def get_file_info(self, symbol: str, format: str = 'parquet') -> Optional[Dict]:
        """
        获取文件信息
        
        Args:
            symbol: 品种代码
            format: 文件格式（'parquet'或'hdf5'）
            
        Returns:
            文件信息字典或None
        """
        try:
            if format == 'parquet':
                file_path = self.base_path / f"{symbol}.parquet"
            else:
                file_path = self.base_path / f"{symbol}.h5"
            
            if not file_path.exists():
                return None
            
            stat = file_path.stat()
            
            info = {
                'path': str(file_path),
                'size_mb': stat.st_size / 1024 / 1024,
                'modified_time': pd.Timestamp(stat.st_mtime, unit='s'),
                'format': format
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取 {symbol} 文件信息失败: {str(e)}")
            return None
    
    def list_files(self, format: str = 'parquet') -> List[str]:
        """
        列出所有数据文件
        
        Args:
            format: 文件格式
            
        Returns:
            品种代码列表
        """
        try:
            if format == 'parquet':
                pattern = '*.parquet'
            else:
                pattern = '*.h5'
            
            files = list(self.base_path.glob(pattern))
            symbols = [f.stem for f in files]
            
            logger.info(f"找到 {len(symbols)} 个 {format} 文件")
            
            return symbols
            
        except Exception as e:
            logger.error(f"列出文件失败: {str(e)}")
            return []
    
    def delete_file(self, symbol: str, format: str = 'parquet') -> bool:
        """
        删除数据文件
        
        Args:
            symbol: 品种代码
            format: 文件格式
            
        Returns:
            是否删除成功
        """
        try:
            if format == 'parquet':
                file_path = self.base_path / f"{symbol}.parquet"
            else:
                file_path = self.base_path / f"{symbol}.h5"
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"删除文件: {file_path}")
                return True
            else:
                logger.warning(f"文件不存在: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"删除 {symbol} 失败: {str(e)}")
            return False


class DataCache:
    """
    数据缓存管理器
    
    用于在内存中缓存常用数据，提高访问速度
    """
    
    def __init__(self, max_size: int = 10):
        """
        初始化缓存管理器
        
        Args:
            max_size: 最大缓存数量
        """
        self.max_size = max_size
        self.cache: Dict[str, pd.DataFrame] = {}
        self.access_count: Dict[str, int] = {}
    
    def get(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            symbol: 品种代码
            
        Returns:
            DataFrame或None
        """
        if symbol in self.cache:
            self.access_count[symbol] += 1
            logger.debug(f"缓存命中: {symbol}")
            return self.cache[symbol].copy()
        
        logger.debug(f"缓存未命中: {symbol}")
        return None
    
    def put(self, symbol: str, data: pd.DataFrame) -> None:
        """
        将数据放入缓存
        
        Args:
            symbol: 品种代码
            data: DataFrame
        """
        # 如果缓存已满，移除访问次数最少的
        if len(self.cache) >= self.max_size and symbol not in self.cache:
            least_used = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used]
            del self.access_count[least_used]
            logger.debug(f"缓存已满，移除: {least_used}")
        
        self.cache[symbol] = data.copy()
        self.access_count[symbol] = 0
        logger.debug(f"加入缓存: {symbol}")
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        logger.info("缓存已清空")
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)