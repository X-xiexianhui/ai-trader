"""
IB Gateway数据模块
提供与Interactive Brokers Gateway的连接、数据下载和实时数据流功能
"""

from .ib_connector import (
    IBConnector,
    create_futures_contract,
    create_stock_contract,
    create_continuous_futures_contract
)

from .ib_historical_data import (
    IBHistoricalDataDownloader,
    download_futures_data
)

from .ib_realtime_data import (
    IBRealtimeDataStreamer,
    RealtimeDataRecorder
)

from .data_storage import DataStorage

__all__ = [
    # 连接器
    'IBConnector',
    'create_futures_contract',
    'create_stock_contract',
    'create_continuous_futures_contract',
    
    # 历史数据
    'IBHistoricalDataDownloader',
    'download_futures_data',
    
    # 实时数据
    'IBRealtimeDataStreamer',
    'RealtimeDataRecorder',
    
    # 数据存储
    'DataStorage',
]

__version__ = '1.0.0'
__author__ = 'AI Trader Team'