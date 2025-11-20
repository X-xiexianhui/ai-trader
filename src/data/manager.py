"""
数据管理器模块
Data Manager Module

实现多品种数据管理和版本控制功能
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from pathlib import Path
import json
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.helpers import get_timestamp, save_json, load_json
from src.data.downloader import YahooFinanceDownloader
from src.data.cache import DataCache
from src.data.updater import DataUpdater
from src.data.cleaner import DataCleaningPipeline
from src.data.validator import DataValidator

logger = get_logger(__name__)


class DataVersionControl:
    """
    数据版本控制器
    
    功能:
    - 记录数据版本信息
    - 支持版本回退
    - 记录数据变更历史
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化版本控制器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 版本控制配置
        self.enabled = self.config.get('versioning.enabled', True)
        self.metadata_file = project_root / self.config.get(
            'versioning.metadata_file',
            'data/cache/versions.json'
        )
        self.keep_versions = self.config.get('versioning.keep_versions', 5)
        
        # 加载版本历史
        self.versions = self._load_versions()
        
        logger.info("数据版本控制初始化完成")
    
    def create_version(
        self,
        symbol: str,
        interval: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建新版本
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            data: 数据
            metadata: 额外的元数据
        
        Returns:
            str: 版本ID
        """
        if not self.enabled:
            return ""
        
        version_id = get_timestamp()
        key = f"{symbol}_{interval}"
        
        version_info = {
            'version_id': version_id,
            'symbol': symbol,
            'interval': interval,
            'created_at': version_id,
            'records': len(data),
            'start_date': str(data['datetime'].min()) if 'datetime' in data.columns else None,
            'end_date': str(data['datetime'].max()) if 'datetime' in data.columns else None,
            'columns': data.columns.tolist(),
        }
        
        if metadata:
            version_info.update(metadata)
        
        # 添加到版本历史
        if key not in self.versions:
            self.versions[key] = []
        
        self.versions[key].append(version_info)
        
        # 限制保留的版本数
        if len(self.versions[key]) > self.keep_versions:
            self.versions[key] = self.versions[key][-self.keep_versions:]
        
        # 保存版本历史
        self._save_versions()
        
        logger.info(f"创建版本: {symbol} ({interval}), 版本ID: {version_id}")
        return version_id
    
    def get_versions(self, symbol: str, interval: str) -> List[Dict[str, Any]]:
        """
        获取版本列表
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            List[Dict]: 版本列表
        """
        key = f"{symbol}_{interval}"
        return self.versions.get(key, [])
    
    def get_version_info(
        self,
        symbol: str,
        interval: str,
        version_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取指定版本信息
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            version_id: 版本ID
        
        Returns:
            Dict: 版本信息
        """
        versions = self.get_versions(symbol, interval)
        for version in versions:
            if version['version_id'] == version_id:
                return version
        return None
    
    def get_latest_version(
        self,
        symbol: str,
        interval: str
    ) -> Optional[Dict[str, Any]]:
        """
        获取最新版本信息
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            Dict: 版本信息
        """
        versions = self.get_versions(symbol, interval)
        return versions[-1] if versions else None
    
    def _load_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载版本历史"""
        if not self.enabled or not self.metadata_file.exists():
            return {}
        
        try:
            return load_json(str(self.metadata_file))
        except Exception as e:
            logger.warning(f"加载版本历史失败: {str(e)}")
            return {}
    
    def _save_versions(self):
        """保存版本历史"""
        if not self.enabled:
            return
        
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.versions, str(self.metadata_file))
        except Exception as e:
            logger.error(f"保存版本历史失败: {str(e)}")


class MultiSymbolDataManager:
    """
    多品种数据管理器
    
    功能:
    - 统一管理多个品种
    - 批量操作支持
    - 数据索引和查询
    - 版本控制集成
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化数据管理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is None:
            config_path = str(project_root / "configs" / "data_config.yaml")
        
        self.config = ConfigLoader(config_path)
        
        # 初始化各个组件
        self.downloader = YahooFinanceDownloader(config_path)
        self.cache = DataCache(config_path)
        self.updater = DataUpdater(config_path)
        self.cleaner = DataCleaningPipeline(config_path)
        self.validator = DataValidator(config_path)
        self.version_control = DataVersionControl(config_path)
        
        # 数据索引
        self.data_index = {}
        self._build_index()
        
        logger.info("多品种数据管理器初始化完成")
    
    def add_symbol(
        self,
        symbol: str,
        interval: str = "5m",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        force_download: bool = False
    ) -> bool:
        """
        添加品种数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            start: 开始日期
            end: 结束日期
            force_download: 是否强制下载
        
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"添加品种: {symbol} ({interval})")
            
            # 更新数据
            data = self.updater.update(
                symbol=symbol,
                interval=interval,
                start=start,
                end=end,
                force=force_download
            )
            
            if data is None or data.empty:
                logger.warning(f"品种数据为空: {symbol}")
                return False
            
            # 清洗数据
            cleaned_data, report = self.cleaner.fit_transform(
                data,
                symbol=symbol,
                convert_timezone=True,
                filter_trading_hours=False,
                validate=True,
                fix_issues=True
            )
            
            # 保存到缓存
            self.cache.save(cleaned_data, symbol, interval)
            
            # 创建版本
            self.version_control.create_version(
                symbol=symbol,
                interval=interval,
                data=cleaned_data,
                metadata={'quality_score': report.get('validation_report', {}).get('quality_score', 0)}
            )
            
            # 更新索引
            self._add_to_index(symbol, interval)
            
            logger.info(f"品种添加成功: {symbol}, {len(cleaned_data)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"添加品种失败: {symbol}, {str(e)}")
            return False
    
    def get_symbol_data(
        self,
        symbol: str,
        interval: str = "5m",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> Optional[pd.DataFrame]:
        """
        获取品种数据
        
        Args:
            symbol: 品种代码
            interval: 数据频率
            start: 开始日期（可选，用于过滤）
            end: 结束日期（可选，用于过滤）
        
        Returns:
            pd.DataFrame: 品种数据
        """
        # 从缓存加载
        data = self.cache.load(symbol, interval, check_expiry=False)
        
        if data is None:
            logger.warning(f"品种数据不存在: {symbol} ({interval})")
            return None
        
        # 时间范围过滤
        if start is not None or end is not None:
            if 'datetime' in data.columns:
                if start is not None:
                    start = pd.to_datetime(start)
                    data = data[data['datetime'] >= start]
                if end is not None:
                    end = pd.to_datetime(end)
                    data = data[data['datetime'] <= end]
        
        return data
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "5m",
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个品种数据
        
        Args:
            symbols: 品种代码列表
            interval: 数据频率
            start: 开始日期
            end: 结束日期
        
        Returns:
            Dict[str, pd.DataFrame]: 品种代码到数据的映射
        """
        results = {}
        
        for symbol in symbols:
            data = self.get_symbol_data(symbol, interval, start, end)
            if data is not None:
                results[symbol] = data
        
        return results
    
    def update_all_symbols(
        self,
        interval: str = "5m",
        delay: float = 1.0
    ) -> Dict[str, bool]:
        """
        更新所有已添加的品种
        
        Args:
            interval: 数据频率
            delay: 品种间延迟
        
        Returns:
            Dict[str, bool]: 品种代码到更新结果的映射
        """
        symbols = self.list_symbols(interval)
        results = {}
        
        logger.info(f"开始更新所有品种: {len(symbols)} 个")
        
        for symbol in symbols:
            try:
                data = self.updater.update(symbol, interval)
                if data is not None and not data.empty:
                    # 清洗和保存
                    cleaned_data = self.cleaner.transform(data, symbol)
                    self.cache.save(cleaned_data, symbol, interval)
                    results[symbol] = True
                else:
                    results[symbol] = False
                
                if delay > 0:
                    import time
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"更新品种失败: {symbol}, {str(e)}")
                results[symbol] = False
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"更新完成: {success_count}/{len(symbols)} 成功")
        
        return results
    
    def remove_symbol(self, symbol: str, interval: str = "5m") -> bool:
        """
        移除品种
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            bool: 是否成功
        """
        try:
            # 删除缓存
            self.cache.delete(symbol, interval)
            
            # 从索引中移除
            self._remove_from_index(symbol, interval)
            
            logger.info(f"品种移除成功: {symbol} ({interval})")
            return True
            
        except Exception as e:
            logger.error(f"移除品种失败: {symbol}, {str(e)}")
            return False
    
    def list_symbols(self, interval: Optional[str] = None) -> List[str]:
        """
        列出所有品种
        
        Args:
            interval: 数据频率（可选，用于过滤）
        
        Returns:
            List[str]: 品种代码列表
        """
        if interval:
            return [
                item['symbol']
                for item in self.cache.list_cached_symbols()
                if item['interval'] == interval
            ]
        else:
            return list(set(
                item['symbol']
                for item in self.cache.list_cached_symbols()
            ))
    
    def get_symbol_info(self, symbol: str, interval: str = "5m") -> Dict[str, Any]:
        """
        获取品种信息
        
        Args:
            symbol: 品种代码
            interval: 数据频率
        
        Returns:
            Dict: 品种信息
        """
        info = {
            'symbol': symbol,
            'interval': interval,
            'cached': self.cache.exists(symbol, interval),
            'metadata': None,
            'versions': [],
            'latest_version': None
        }
        
        # 获取缓存元数据
        metadata = self.cache.get_metadata(symbol, interval)
        if metadata:
            info['metadata'] = metadata
        
        # 获取版本信息
        versions = self.version_control.get_versions(symbol, interval)
        if versions:
            info['versions'] = versions
            info['latest_version'] = versions[-1]
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据管理统计信息
        
        Returns:
            Dict: 统计信息
        """
        cached_symbols = self.cache.list_cached_symbols()
        
        stats = {
            'total_symbols': len(set(item['symbol'] for item in cached_symbols)),
            'total_datasets': len(cached_symbols),
            'cache_size': self.cache.get_cache_size(),
            'symbols_by_interval': {},
            'total_records': 0
        }
        
        # 按频率统计
        for item in cached_symbols:
            interval = item['interval']
            if interval not in stats['symbols_by_interval']:
                stats['symbols_by_interval'][interval] = []
            stats['symbols_by_interval'][interval].append(item['symbol'])
            
            # 统计总记录数
            metadata = self.cache.get_metadata(item['symbol'], interval)
            if metadata:
                stats['total_records'] += metadata.get('rows', 0)
        
        return stats
    
    def _build_index(self):
        """构建数据索引"""
        cached_symbols = self.cache.list_cached_symbols()
        for item in cached_symbols:
            key = f"{item['symbol']}_{item['interval']}"
            self.data_index[key] = {
                'symbol': item['symbol'],
                'interval': item['interval'],
                'indexed_at': get_timestamp()
            }
    
    def _add_to_index(self, symbol: str, interval: str):
        """添加到索引"""
        key = f"{symbol}_{interval}"
        self.data_index[key] = {
            'symbol': symbol,
            'interval': interval,
            'indexed_at': get_timestamp()
        }
    
    def _remove_from_index(self, symbol: str, interval: str):
        """从索引中移除"""
        key = f"{symbol}_{interval}"
        if key in self.data_index:
            del self.data_index[key]


def main():
    """测试函数"""
    # 创建数据管理器
    manager = MultiSymbolDataManager()
    
    # 测试添加品种
    print("\n=== 测试添加品种 ===")
    success = manager.add_symbol(
        symbol="AAPL",
        interval="1d",
        start="2024-01-01",
        end="2024-01-31",
        force_download=True
    )
    print(f"添加结果: {'成功' if success else '失败'}")
    
    # 测试获取品种数据
    print("\n=== 测试获取品种数据 ===")
    data = manager.get_symbol_data("AAPL", "1d")
    if data is not None:
        print(f"数据: {len(data)} 条记录")
        print(data.head())
    
    # 测试品种信息
    print("\n=== 测试品种信息 ===")
    info = manager.get_symbol_info("AAPL", "1d")
    print(f"品种信息:")
    print(f"  缓存: {info['cached']}")
    print(f"  版本数: {len(info['versions'])}")
    if info['latest_version']:
        print(f"  最新版本: {info['latest_version']['version_id']}")
    
    # 测试统计信息
    print("\n=== 测试统计信息 ===")
    stats = manager.get_statistics()
    print(f"统计信息:")
    print(f"  总品种数: {stats['total_symbols']}")
    print(f"  总数据集: {stats['total_datasets']}")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  缓存大小: {stats['cache_size']['total_size_mb']} MB")


if __name__ == "__main__":
    main()