"""
IB实时数据流处理器
支持实时K线数据订阅和处理
"""

import pandas as pd
from typing import Optional, Callable, Dict, List
from datetime import datetime
import logging
from ib_insync import IB, Contract, RealTimeBarList, RealTimeBar
from collections import deque
import threading

from .ib_connector import IBConnector

logger = logging.getLogger(__name__)


class IBRealtimeDataStreamer:
    """IB实时数据流处理器"""
    
    def __init__(
        self,
        connector: IBConnector,
        buffer_size: int = 1000
    ):
        """
        初始化实时数据流处理器
        
        Args:
            connector: IB连接器实例
            buffer_size: 数据缓冲区大小
        """
        self.connector = connector
        self.ib = connector.ib
        self.buffer_size = buffer_size
        
        # 数据缓冲区：contract -> deque of bars
        self.buffers: Dict[str, deque] = {}
        
        # 回调函数：contract -> callback
        self.callbacks: Dict[str, Callable] = {}
        
        # 订阅管理
        self.subscriptions: Dict[str, RealTimeBarList] = {}
        
        logger.info(f"实时数据流处理器初始化完成，缓冲区大小: {buffer_size}")
    
    def subscribe_realtime_bars(
        self,
        contract: Contract,
        bar_size: int = 5,
        what_to_show: str = 'TRADES',
        use_rth: bool = False,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        订阅实时K线数据
        
        Args:
            contract: 合约对象
            bar_size: K线周期（秒），只支持5秒
            what_to_show: 数据类型 ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            use_rth: 是否只使用常规交易时间
            callback: 数据回调函数，接收参数(contract, bar)
        
        Returns:
            bool: 订阅是否成功
        """
        if not self.connector.is_connected:
            logger.error("未连接到IB Gateway")
            return False
        
        try:
            contract_key = self._get_contract_key(contract)
            
            # 检查是否已订阅
            if contract_key in self.subscriptions:
                logger.warning(f"合约 {contract_key} 已经订阅")
                return True
            
            logger.info(f"订阅实时K线: {contract_key}, bar_size={bar_size}s")
            
            # 初始化缓冲区
            self.buffers[contract_key] = deque(maxlen=self.buffer_size)
            
            # 设置回调
            if callback:
                self.callbacks[contract_key] = callback
            
            # 订阅实时K线
            bars = self.ib.reqRealTimeBars(
                contract=contract,
                barSize=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth
            )
            
            # 注册更新回调
            bars.updateEvent += lambda bars, hasNewBar: self._on_bar_update(
                contract, bars, hasNewBar
            )
            
            self.subscriptions[contract_key] = bars
            
            logger.info(f"成功订阅实时K线: {contract_key}")
            return True
            
        except Exception as e:
            logger.error(f"订阅实时K线失败: {e}")
            return False
    
    def unsubscribe_realtime_bars(self, contract: Contract) -> bool:
        """
        取消订阅实时K线
        
        Args:
            contract: 合约对象
        
        Returns:
            bool: 取消订阅是否成功
        """
        try:
            contract_key = self._get_contract_key(contract)
            
            if contract_key not in self.subscriptions:
                logger.warning(f"合约 {contract_key} 未订阅")
                return False
            
            # 取消订阅
            bars = self.subscriptions[contract_key]
            self.ib.cancelRealTimeBars(bars)
            
            # 清理
            del self.subscriptions[contract_key]
            if contract_key in self.callbacks:
                del self.callbacks[contract_key]
            
            logger.info(f"已取消订阅: {contract_key}")
            return True
            
        except Exception as e:
            logger.error(f"取消订阅失败: {e}")
            return False
    
    def _on_bar_update(
        self,
        contract: Contract,
        bars: RealTimeBarList,
        has_new_bar: bool
    ):
        """
        K线更新回调
        
        Args:
            contract: 合约对象
            bars: K线列表
            has_new_bar: 是否有新K线
        """
        if not has_new_bar or not bars:
            return
        
        contract_key = self._get_contract_key(contract)
        latest_bar = bars[-1]
        
        # 添加到缓冲区
        bar_data = {
            'date': latest_bar.time,
            'open': latest_bar.open_,
            'high': latest_bar.high,
            'low': latest_bar.low,
            'close': latest_bar.close,
            'volume': latest_bar.volume,
            'wap': latest_bar.wap,
            'count': latest_bar.count
        }
        
        self.buffers[contract_key].append(bar_data)
        
        logger.debug(f"收到新K线 {contract_key}: {latest_bar.time} "
                    f"O:{latest_bar.open_:.2f} H:{latest_bar.high:.2f} "
                    f"L:{latest_bar.low:.2f} C:{latest_bar.close:.2f} "
                    f"V:{latest_bar.volume}")
        
        # 调用用户回调
        if contract_key in self.callbacks:
            try:
                self.callbacks[contract_key](contract, bar_data)
            except Exception as e:
                logger.error(f"回调函数执行失败: {e}")
    
    def get_buffer_data(
        self,
        contract: Contract,
        as_dataframe: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取缓冲区数据
        
        Args:
            contract: 合约对象
            as_dataframe: 是否返回DataFrame格式
        
        Returns:
            pd.DataFrame或list: 缓冲区数据
        """
        contract_key = self._get_contract_key(contract)
        
        if contract_key not in self.buffers:
            logger.warning(f"合约 {contract_key} 未订阅")
            return None
        
        buffer = self.buffers[contract_key]
        
        if not buffer:
            logger.warning(f"缓冲区为空: {contract_key}")
            return None
        
        if as_dataframe:
            df = pd.DataFrame(list(buffer))
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            return list(buffer)
    
    def clear_buffer(self, contract: Contract):
        """
        清空缓冲区
        
        Args:
            contract: 合约对象
        """
        contract_key = self._get_contract_key(contract)
        
        if contract_key in self.buffers:
            self.buffers[contract_key].clear()
            logger.info(f"已清空缓冲区: {contract_key}")
    
    def get_latest_bar(self, contract: Contract) -> Optional[Dict]:
        """
        获取最新K线
        
        Args:
            contract: 合约对象
        
        Returns:
            Dict: 最新K线数据
        """
        contract_key = self._get_contract_key(contract)
        
        if contract_key not in self.buffers or not self.buffers[contract_key]:
            return None
        
        return self.buffers[contract_key][-1]
    
    def _get_contract_key(self, contract: Contract) -> str:
        """
        生成合约唯一标识
        
        Args:
            contract: 合约对象
        
        Returns:
            str: 合约标识
        """
        return f"{contract.symbol}_{contract.secType}_{contract.exchange}"
    
    def unsubscribe_all(self):
        """取消所有订阅"""
        contracts = list(self.subscriptions.keys())
        for contract_key in contracts:
            # 从key重建contract（简化版）
            parts = contract_key.split('_')
            contract = Contract()
            contract.symbol = parts[0]
            contract.secType = parts[1]
            contract.exchange = parts[2]
            
            self.unsubscribe_realtime_bars(contract)
        
        logger.info("已取消所有订阅")


class RealtimeDataRecorder:
    """实时数据记录器，将实时数据保存到文件"""
    
    def __init__(
        self,
        streamer: IBRealtimeDataStreamer,
        save_interval: int = 100
    ):
        """
        初始化数据记录器
        
        Args:
            streamer: 实时数据流处理器
            save_interval: 保存间隔（K线数量）
        """
        self.streamer = streamer
        self.save_interval = save_interval
        
        # 记录计数器
        self.counters: Dict[str, int] = {}
        
        logger.info(f"数据记录器初始化完成，保存间隔: {save_interval}")
    
    def start_recording(
        self,
        contract: Contract,
        filepath: str,
        bar_size: int = 5,
        what_to_show: str = 'TRADES'
    ) -> bool:
        """
        开始记录数据
        
        Args:
            contract: 合约对象
            filepath: 保存文件路径
            bar_size: K线周期（秒）
            what_to_show: 数据类型
        
        Returns:
            bool: 是否成功开始记录
        """
        contract_key = self.streamer._get_contract_key(contract)
        
        # 创建回调函数
        def save_callback(contract, bar_data):
            self._on_new_bar(contract, bar_data, filepath)
        
        # 订阅数据
        success = self.streamer.subscribe_realtime_bars(
            contract=contract,
            bar_size=bar_size,
            what_to_show=what_to_show,
            callback=save_callback
        )
        
        if success:
            self.counters[contract_key] = 0
            logger.info(f"开始记录数据到: {filepath}")
        
        return success
    
    def _on_new_bar(self, contract: Contract, bar_data: Dict, filepath: str):
        """
        新K线回调
        
        Args:
            contract: 合约对象
            bar_data: K线数据
            filepath: 保存路径
        """
        contract_key = self.streamer._get_contract_key(contract)
        
        # 增加计数
        self.counters[contract_key] += 1
        
        # 达到保存间隔时保存数据
        if self.counters[contract_key] >= self.save_interval:
            self._save_buffer(contract, filepath)
            self.counters[contract_key] = 0
    
    def _save_buffer(self, contract: Contract, filepath: str):
        """
        保存缓冲区数据
        
        Args:
            contract: 合约对象
            filepath: 保存路径
        """
        try:
            df = self.streamer.get_buffer_data(contract)
            
            if df is not None and not df.empty:
                # 追加模式保存
                import os
                mode = 'a' if os.path.exists(filepath) else 'w'
                header = not os.path.exists(filepath)
                
                df.to_csv(filepath, mode=mode, header=header, index=False)
                logger.info(f"已保存 {len(df)} 条数据到: {filepath}")
        
        except Exception as e:
            logger.error(f"保存数据失败: {e}")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from .ib_connector import create_futures_contract
    import time
    
    # 创建连接
    with IBConnector(host='127.0.0.1', port=4001) as connector:
        # 创建合约
        contract = create_futures_contract('ES', 'CME')
        qualified = connector.qualify_contract(contract)
        
        if qualified:
            # 创建实时数据流
            streamer = IBRealtimeDataStreamer(connector)
            
            # 定义回调函数
            def on_new_bar(contract, bar_data):
                print(f"\n新K线: {bar_data['date']}")
                print(f"OHLC: {bar_data['open']:.2f}, {bar_data['high']:.2f}, "
                      f"{bar_data['low']:.2f}, {bar_data['close']:.2f}")
                print(f"成交量: {bar_data['volume']}")
            
            # 订阅实时数据
            streamer.subscribe_realtime_bars(
                contract=qualified,
                bar_size=5,
                callback=on_new_bar
            )
            
            # 运行60秒
            print("\n开始接收实时数据，运行60秒...")
            time.sleep(60)
            
            # 获取缓冲区数据
            df = streamer.get_buffer_data(qualified)
            if df is not None:
                print(f"\n缓冲区数据: {len(df)} 条")
                print(df.tail())
            
            # 取消订阅
            streamer.unsubscribe_all()