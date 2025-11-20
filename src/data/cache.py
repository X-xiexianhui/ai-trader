"""
数据缓存模块
Data Cache Module

实现本地数据缓存功能，支持Parquet格式存储、版本管理和增量更新
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from datetime import datetime, timedelta
import json
import hashlib
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.helpers import get_timestamp, save_json, load_json

logger = get_logger(__name__)


class DataCache:
    """
    数据缓存管理器
    
    功能:
    - Parquet格式存储
    - 数据版本管理
    - 增量更新支持
    - 缓存过期检查
    - 元数据管理
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化缓存管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 缓存目录
        cache_dir = self.config.get('cache.directory', 'data/cache')
        self.cache_dir = project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存格式
        self.cache_format = self.config.get('cache.format', 'parquet')
        self.compression = self.config.get('cache.strategy.compression', 'snappy')
        
        # 缓存过期时间
        self.expiry_intraday = self.config.get('cache.expiry.intraday', 3600)
        self.expiry_daily = self.config.get('cache.expiry.daily', 86400)
        
        # 版本控制
        self.versioning_enabled = self.config.get('versioning.enabled', True)
        self.metadata_file = project_root / self.config.get(
            'versioning.metadata_file', 
            'data/cache/versions.json'
        )
        self.keep_versions = self.config.get('versioning.keep_versions', 5)
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        logger.info(f"数据缓存初始化完成: {self.cache_dir}")
    
    def save(
        self,
        data: pd.DataFrame,
        symbol: str,
        interval: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        保存数据到缓存
        
        Args:
            data: 要保存的数据
            symbol: 品种代码
            interval: 数据频率
            metadata: 额外的元数据
        
        Returns:
            bool: 是否保存成功
        """
        try:
            if data.empty:
                logger.warning(f"数据为空，跳过保存: {symbol}")
                return False
            
            # 生成缓存文件路径
            cache_file = self._get_cache_path(symbol, interval)
            
            # 确保目录存在
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存数据
            if self.cache_format == 'parquet':
                data.to_parquet(
                    cache_file,
                    compression=self.compression,
                    index=False
                )
            elif self.cache_format == 'csv':
                data.to_csv(cache_file, index=False)
            elif self.cache_format == 'pickle':
                data.to_pickle(cache_file)
            else:
                raise ValueError(f"不支持的缓存格式: {self.cache_format}")
            
            # 更新元数据
            cache_metadata = {
                'symbol': symbol,
                'interval': interval,
                'rows': len(data),
                'columns': data.columns.tolist(),
                'start_date': str(data['datetime'].min()) if 'datetime' in data.columns else None,
                'end_date': str(data['datetime'].max()) if 'datetime' in data.columns else None,
                'file_path': str(cache_file),
                'file_size': cache_file.stat().st_size,
                'format': self.cache_format,
                'compression': self.compression,
                'created_at': get_timestamp(),
                'checksum': self._calculate_checksum(data)
            }
            
            # 添加用户提供的元数据
            if metadata:
                cache_metadata.update(metadata)
            
            # 保存元数据
            self._save_metadata(symbol, interval, cache_metadata)
            
            logger.info(f"数据保存成功: {symbol} ({interval}), {len(data)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"保存数据失败: {symbol} ({interval}), {str(e)}")
            return False
    
    def load(
        self,
        symbol: str,
        interval: str,
        check_expiry: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            check_expiry: 是否检查缓存过期
        
        Returns:
            pd.DataFrame: 加载的数据，如果不存在或过期则返回None
        """
        try:
            # 获取缓存文件路径
            cache_file = self._get_cache_path(symbol, interval)
            
            # 检查文件是否存在
            if not cache_file.exists():
                logger.debug(f"缓存文件不存在: {symbol} ({interval})")
                return None
            
            # 检查缓存是否过期
            if check_expiry and self._is_expired(symbol, interval):
                logger.info(f"缓存已过期: {symbol} ({interval})")
                return None
            
            # 加载数据
            if self.cache_format == 'parquet':
                data = pd.read_parquet(cache_file)
            elif self.cache_format == 'csv':
                data = pd.read_csv(cache_file)
            elif self.cache_format == 'pickle':
                data = pd.read_pickle(cache_file)
            else:
                raise ValueError(f"不支持的缓存格式: {self.cache_format}")
            
            # 验证数据完整性
            if not self._verify_checksum(data, symbol, interval):
                logger.warning(f"数据校验失败: {symbol} ({interval})")
                return None
            
            logger.info(f"数据加载成功: {symbol} ({interval}), {len(data)} 条记录")
            return data
            
        except Exception as e:
            logger.error(f"加载数据失败: {symbol} ({interval}), {str(e)}")
            return None
    
    def exists(self, symbol: str, interval: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            bool: 缓存是否存在
        """
        cache_file = self._get_cache_path(symbol, interval)
        return cache_file.exists()
    
    def delete(self, symbol: str, interval: str) -> bool:
        """
        删除缓存
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            bool: 是否删除成功
        """
        try:
            cache_file = self._get_cache_path(symbol, interval)
            
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"缓存删除成功: {symbol} ({interval})")
            
            # 删除元数据
            cache_key = self._get_cache_key(symbol, interval)
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata_file()
            
            return True
            
        except Exception as e:
            logger.error(f"删除缓存失败: {symbol} ({interval}), {str(e)}")
            return False
    
    def clear_all(self) -> int:
        """
        清空所有缓存
        
        Returns:
            int: 删除的文件数量
        """
        count = 0
        try:
            for cache_file in self.cache_dir.rglob(f"*.{self._get_file_extension()}"):
                cache_file.unlink()
                count += 1
            
            # 清空元数据
            self.metadata = {}
            self._save_metadata_file()
            
            logger.info(f"清空所有缓存: {count} 个文件")
            return count
            
        except Exception as e:
            logger.error(f"清空缓存失败: {str(e)}")
            return count
    
    def get_metadata(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存元数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            Dict: 元数据字典
        """
        cache_key = self._get_cache_key(symbol, interval)
        return self.metadata.get(cache_key)
    
    def list_cached_symbols(self) -> List[Dict[str, str]]:
        """
        列出所有已缓存的品种
        
        Returns:
            List[Dict]: 品种列表，每个元素包含symbol和interval
        """
        cached = []
        for cache_key in self.metadata.keys():
            parts = cache_key.split('_')
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1])
                interval = parts[-1]
                cached.append({'symbol': symbol, 'interval': interval})
        
        return cached
    
    def get_cache_size(self) -> Dict[str, Any]:
        """
        获取缓存大小统计
        
        Returns:
            Dict: 缓存大小信息
        """
        total_size = 0
        file_count = 0
        
        for cache_file in self.cache_dir.rglob(f"*.{self._get_file_extension()}"):
            total_size += cache_file.stat().st_size
            file_count += 1
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'file_count': file_count,
            'cache_dir': str(self.cache_dir)
        }
    
    def _get_cache_path(self, symbol: str, interval: str) -> Path:
        """生成缓存文件路径"""
        # 清理符号中的特殊字符
        safe_symbol = symbol.replace('=', '_').replace('/', '_')
        filename = f"{safe_symbol}_{interval}.{self._get_file_extension()}"
        return self.cache_dir / filename
    
    def _get_file_extension(self) -> str:
        """获取文件扩展名"""
        extensions = {
            'parquet': 'parquet',
            'csv': 'csv',
            'pickle': 'pkl'
        }
        return extensions.get(self.cache_format, 'parquet')
    
    def _get_cache_key(self, symbol: str, interval: str) -> str:
        """生成缓存键"""
        return f"{symbol}_{interval}"
    
    def _is_expired(self, symbol: str, interval: str) -> bool:
        """检查缓存是否过期"""
        cache_file = self._get_cache_path(symbol, interval)
        
        if not cache_file.exists():
            return True
        
        # 获取文件修改时间
        mtime = cache_file.stat().st_mtime
        current_time = datetime.now().timestamp()
        age = current_time - mtime
        
        # 根据数据频率确定过期时间
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            expiry_time = self.expiry_intraday
        else:
            expiry_time = self.expiry_daily
        
        return age > expiry_time
    
    def _calculate_checksum(self, data: pd.DataFrame) -> str:
        """计算数据校验和"""
        try:
            # 使用数据的字符串表示计算MD5
            data_str = data.to_json()
            return hashlib.md5(data_str.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"计算校验和失败: {str(e)}")
            return ""
    
    def _verify_checksum(self, data: pd.DataFrame, symbol: str, interval: str) -> bool:
        """验证数据校验和"""
        if not self.versioning_enabled:
            return True
        
        metadata = self.get_metadata(symbol, interval)
        if not metadata or 'checksum' not in metadata:
            return True
        
        stored_checksum = metadata['checksum']
        if not stored_checksum:
            return True
        
        current_checksum = self._calculate_checksum(data)
        return current_checksum == stored_checksum
    
    def _load_metadata(self) -> Dict[str, Any]:
        """加载元数据"""
        if not self.versioning_enabled:
            return {}
        
        if self.metadata_file.exists():
            try:
                return load_json(str(self.metadata_file))
            except Exception as e:
                logger.warning(f"加载元数据失败: {str(e)}")
                return {}
        
        return {}
    
    def _save_metadata(self, symbol: str, interval: str, metadata: Dict[str, Any]):
        """保存单个缓存的元数据"""
        if not self.versioning_enabled:
            return
        
        cache_key = self._get_cache_key(symbol, interval)
        
        # 如果启用版本控制，保存历史版本
        if cache_key in self.metadata:
            old_metadata = self.metadata[cache_key]
            if 'versions' not in old_metadata:
                old_metadata['versions'] = []
            
            # 添加当前版本到历史
            old_metadata['versions'].append({
                'created_at': old_metadata.get('created_at'),
                'checksum': old_metadata.get('checksum'),
                'rows': old_metadata.get('rows')
            })
            
            # 限制保留的版本数
            if len(old_metadata['versions']) > self.keep_versions:
                old_metadata['versions'] = old_metadata['versions'][-self.keep_versions:]
            
            metadata['versions'] = old_metadata['versions']
        
        self.metadata[cache_key] = metadata
        self._save_metadata_file()
    
    def _save_metadata_file(self):
        """保存元数据文件"""
        if not self.versioning_enabled:
            return
        
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.metadata, str(self.metadata_file))
        except Exception as e:
            logger.error(f"保存元数据文件失败: {str(e)}")


def main():
    """测试函数"""
    # 创建缓存管理器
    cache = DataCache()
    
    # 创建测试数据
    print("\n=== 创建测试数据 ===")
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': np.random.randn(100) + 100,
        'high': np.random.randn(100) + 101,
        'low': np.random.randn(100) + 99,
        'close': np.random.randn(100) + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })
    print(f"测试数据: {len(test_data)} 条记录")
    
    # 测试保存
    print("\n=== 测试保存 ===")
    success = cache.save(test_data, 'AAPL', '5m')
    print(f"保存结果: {'成功' if success else '失败'}")
    
    # 测试加载
    print("\n=== 测试加载 ===")
    loaded_data = cache.load('AAPL', '5m', check_expiry=False)
    if loaded_data is not None:
        print(f"加载成功: {len(loaded_data)} 条记录")
        print(f"数据一致性: {test_data.equals(loaded_data)}")
    else:
        print("加载失败")
    
    # 测试元数据
    print("\n=== 测试元数据 ===")
    metadata = cache.get_metadata('AAPL', '5m')
    if metadata:
        print(f"元数据: {json.dumps(metadata, indent=2, default=str)}")
    
    # 测试缓存统计
    print("\n=== 缓存统计 ===")
    stats = cache.get_cache_size()
    print(f"缓存大小: {stats['total_size_mb']} MB")
    print(f"文件数量: {stats['file_count']}")
    
    # 测试列出缓存
    print("\n=== 列出缓存 ===")
    cached = cache.list_cached_symbols()
    print(f"已缓存品种: {cached}")
    
    # 测试删除
    print("\n=== 测试删除 ===")
    success = cache.delete('AAPL', '5m')
    print(f"删除结果: {'成功' if success else '失败'}")
    print(f"缓存存在: {cache.exists('AAPL', '5m')}")


if __name__ == "__main__":
    main()