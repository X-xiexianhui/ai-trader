"""
数据存储模块
提供统一的数据存储和加载接口，支持多种格式
"""

import pandas as pd
import os
from typing import Optional, List, Dict, Union
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataStorage:
    """数据存储管理器"""
    
    def __init__(self, base_dir: str = 'data'):
        """
        初始化数据存储管理器
        
        Args:
            base_dir: 数据存储基础目录
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # 创建目录
        self._create_directories()
        
        logger.info(f"数据存储管理器初始化完成，基础目录: {self.base_dir}")
    
    def _create_directories(self):
        """创建必要的目录结构"""
        for directory in [self.base_dir, self.raw_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"确保目录存在: {directory}")
    
    def save_raw_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        format: str = 'parquet',
        metadata: Optional[Dict] = None
    ) -> str:
        """
        保存原始数据
        
        Args:
            df: 数据DataFrame
            symbol: 交易品种代码
            interval: 时间周期
            format: 保存格式 ('csv', 'parquet', 'hdf5')
            metadata: 元数据（可选）
        
        Returns:
            str: 保存的文件路径
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{interval}_{timestamp}"
            
            # 根据格式保存
            if format == 'csv':
                filepath = self.raw_dir / f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif format == 'parquet':
                filepath = self.raw_dir / f"{filename}.parquet"
                df.to_parquet(filepath, index=False)
            elif format == 'hdf5':
                filepath = self.raw_dir / f"{filename}.h5"
                df.to_hdf(filepath, key='data', mode='w')
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            # 保存元数据
            if metadata:
                self._save_metadata(filepath, metadata)
            
            logger.info(f"原始数据已保存: {filepath} ({len(df)} 条记录)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存原始数据失败: {e}")
            raise
    
    def save_processed_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        processing_type: str,
        format: str = 'parquet',
        metadata: Optional[Dict] = None
    ) -> str:
        """
        保存处理后的数据
        
        Args:
            df: 数据DataFrame
            symbol: 交易品种代码
            interval: 时间周期
            processing_type: 处理类型 (如 'cleaned', 'features', 'normalized')
            format: 保存格式
            metadata: 元数据（可选）
        
        Returns:
            str: 保存的文件路径
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{interval}_{processing_type}_{timestamp}"
            
            # 根据格式保存
            if format == 'csv':
                filepath = self.processed_dir / f"{filename}.csv"
                df.to_csv(filepath, index=False)
            elif format == 'parquet':
                filepath = self.processed_dir / f"{filename}.parquet"
                df.to_parquet(filepath, index=False)
            elif format == 'hdf5':
                filepath = self.processed_dir / f"{filename}.h5"
                df.to_hdf(filepath, key='data', mode='w')
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            # 保存元数据
            if metadata:
                self._save_metadata(filepath, metadata)
            
            logger.info(f"处理后数据已保存: {filepath} ({len(df)} 条记录)")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"保存处理后数据失败: {e}")
            raise
    
    def load_data(
        self,
        filepath: Union[str, Path],
        format: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            filepath: 文件路径
            format: 文件格式（如果为None，从文件扩展名推断）
        
        Returns:
            pd.DataFrame: 加载的数据
        """
        try:
            filepath = Path(filepath)
            
            # 推断格式
            if format is None:
                format = filepath.suffix[1:]  # 去掉点号
            
            # 根据格式加载
            if format == 'csv':
                df = pd.read_csv(filepath)
            elif format == 'parquet':
                df = pd.read_parquet(filepath)
            elif format in ['h5', 'hdf5']:
                df = pd.read_hdf(filepath, key='data')
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"数据已加载: {filepath} ({len(df)} 条记录)")
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def load_latest_raw_data(
        self,
        symbol: str,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """
        加载最新的原始数据
        
        Args:
            symbol: 交易品种代码
            interval: 时间周期
        
        Returns:
            pd.DataFrame: 最新的原始数据，如果不存在返回None
        """
        try:
            # 查找匹配的文件（排除.meta.json文件）
            pattern = f"{symbol}_{interval}_*"
            files = [f for f in self.raw_dir.glob(pattern)
                    if not f.name.endswith('.meta.json')]
            
            if not files:
                logger.warning(f"未找到原始数据: {symbol}_{interval}")
                return None
            
            # 按修改时间排序，获取最新的
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            return self.load_data(latest_file)
            
        except Exception as e:
            logger.error(f"加载最新原始数据失败: {e}")
            return None
    
    def load_latest_processed_data(
        self,
        symbol: str,
        interval: str,
        processing_type: str
    ) -> Optional[pd.DataFrame]:
        """
        加载最新的处理后数据
        
        Args:
            symbol: 交易品种代码
            interval: 时间周期
            processing_type: 处理类型
        
        Returns:
            pd.DataFrame: 最新的处理后数据，如果不存在返回None
        """
        try:
            # 查找匹配的文件（排除.meta.json文件）
            pattern = f"{symbol}_{interval}_{processing_type}_*"
            files = [f for f in self.processed_dir.glob(pattern)
                    if not f.name.endswith('.meta.json')]
            
            if not files:
                logger.warning(f"未找到处理后数据: {symbol}_{interval}_{processing_type}")
                return None
            
            # 按修改时间排序，获取最新的
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            return self.load_data(latest_file)
            
        except Exception as e:
            logger.error(f"加载最新处理后数据失败: {e}")
            return None
    
    def list_raw_data(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        列出原始数据文件
        
        Args:
            symbol: 交易品种代码（可选，用于过滤）
        
        Returns:
            List[Dict]: 文件信息列表
        """
        try:
            pattern = f"{symbol}_*" if symbol else "*"
            # 排除.meta.json文件
            files = [f for f in self.raw_dir.glob(pattern)
                    if not f.name.endswith('.meta.json')]
            
            file_info = []
            for file in files:
                stat = file.stat()
                file_info.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'format': file.suffix[1:]
                })
            
            # 按修改时间排序
            file_info.sort(key=lambda x: x['modified'], reverse=True)
            
            return file_info
            
        except Exception as e:
            logger.error(f"列出原始数据失败: {e}")
            return []
    
    def list_processed_data(
        self,
        symbol: Optional[str] = None,
        processing_type: Optional[str] = None
    ) -> List[Dict]:
        """
        列出处理后数据文件
        
        Args:
            symbol: 交易品种代码（可选）
            processing_type: 处理类型（可选）
        
        Returns:
            List[Dict]: 文件信息列表
        """
        try:
            # 构建搜索模式
            if symbol and processing_type:
                pattern = f"{symbol}_*_{processing_type}_*"
            elif symbol:
                pattern = f"{symbol}_*"
            elif processing_type:
                pattern = f"*_{processing_type}_*"
            else:
                pattern = "*"
            
            # 排除.meta.json文件
            files = [f for f in self.processed_dir.glob(pattern)
                    if not f.name.endswith('.meta.json')]
            
            file_info = []
            for file in files:
                stat = file.stat()
                file_info.append({
                    'filename': file.name,
                    'path': str(file),
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'format': file.suffix[1:]
                })
            
            # 按修改时间排序
            file_info.sort(key=lambda x: x['modified'], reverse=True)
            
            return file_info
            
        except Exception as e:
            logger.error(f"列出处理后数据失败: {e}")
            return []
    
    def _save_metadata(self, data_filepath: Path, metadata: Dict):
        """
        保存元数据到JSON文件
        
        Args:
            data_filepath: 数据文件路径
            metadata: 元数据字典
        """
        try:
            # 生成元数据文件路径
            meta_filepath = data_filepath.with_suffix('.meta.json')
            
            # 添加时间戳
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['data_file'] = str(data_filepath)
            
            # 保存为JSON
            with open(meta_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"元数据已保存: {meta_filepath}")
            
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def load_metadata(self, data_filepath: Union[str, Path]) -> Optional[Dict]:
        """
        加载元数据
        
        Args:
            data_filepath: 数据文件路径
        
        Returns:
            Dict: 元数据字典，如果不存在返回None
        """
        try:
            data_filepath = Path(data_filepath)
            meta_filepath = data_filepath.with_suffix('.meta.json')
            
            if not meta_filepath.exists():
                logger.debug(f"元数据文件不存在: {meta_filepath}")
                return None
            
            with open(meta_filepath, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return metadata
            
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            return None
    
    def delete_data(self, filepath: Union[str, Path]) -> bool:
        """
        删除数据文件及其元数据
        
        Args:
            filepath: 文件路径
        
        Returns:
            bool: 是否成功删除
        """
        try:
            filepath = Path(filepath)
            
            # 删除数据文件
            if filepath.exists():
                filepath.unlink()
                logger.info(f"已删除数据文件: {filepath}")
            
            # 删除元数据文件
            meta_filepath = filepath.with_suffix('.meta.json')
            if meta_filepath.exists():
                meta_filepath.unlink()
                logger.info(f"已删除元数据文件: {meta_filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"删除数据失败: {e}")
            return False
    
    def get_storage_info(self) -> Dict:
        """
        获取存储信息统计
        
        Returns:
            Dict: 存储信息
        """
        try:
            raw_files = list(self.raw_dir.glob('*'))
            processed_files = list(self.processed_dir.glob('*'))
            
            # 计算总大小
            raw_size = sum(f.stat().st_size for f in raw_files if f.is_file())
            processed_size = sum(f.stat().st_size for f in processed_files if f.is_file())
            
            info = {
                'base_dir': str(self.base_dir),
                'raw_data': {
                    'count': len([f for f in raw_files if f.is_file()]),
                    'size_bytes': raw_size,
                    'size_mb': raw_size / (1024 * 1024)
                },
                'processed_data': {
                    'count': len([f for f in processed_files if f.is_file()]),
                    'size_bytes': processed_size,
                    'size_mb': processed_size / (1024 * 1024)
                },
                'total_size_mb': (raw_size + processed_size) / (1024 * 1024)
            }
            
            return info
            
        except Exception as e:
            logger.error(f"获取存储信息失败: {e}")
            return {}


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建存储管理器
    storage = DataStorage(base_dir='data')
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [101 + i * 0.1 for i in range(100)],
        'low': [99 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })
    
    # 保存原始数据
    metadata = {
        'source': 'IB Gateway',
        'description': '测试数据',
        'records': len(test_data)
    }
    
    filepath = storage.save_raw_data(
        df=test_data,
        symbol='ES',
        interval='5m',
        format='parquet',
        metadata=metadata
    )
    
    # 加载数据
    loaded_data = storage.load_data(filepath)
    print(f"\n加载的数据:\n{loaded_data.head()}")
    
    # 加载元数据
    loaded_metadata = storage.load_metadata(filepath)
    print(f"\n元数据:\n{json.dumps(loaded_metadata, indent=2)}")
    
    # 列出文件
    raw_files = storage.list_raw_data(symbol='ES')
    print(f"\n原始数据文件: {len(raw_files)} 个")
    for file in raw_files[:3]:
        print(f"  - {file['filename']} ({file['size']} bytes)")
    
    # 获取存储信息
    info = storage.get_storage_info()
    print(f"\n存储信息:")
    print(f"  原始数据: {info['raw_data']['count']} 个文件, "
          f"{info['raw_data']['size_mb']:.2f} MB")
    print(f"  处理后数据: {info['processed_data']['count']} 个文件, "
          f"{info['processed_data']['size_mb']:.2f} MB")
    print(f"  总计: {info['total_size_mb']:.2f} MB")