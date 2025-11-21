"""
IB历史数据下载器
支持下载各种时间周期的历史数据
"""

import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging
from ib_insync import IB, Contract, BarDataList
import time

from .ib_connector import IBConnector

logger = logging.getLogger(__name__)


class IBHistoricalDataDownloader:
    """IB历史数据下载器"""
    
    # IB支持的时间周期映射
    BAR_SIZE_MAP = {
        '1s': '1 secs',
        '5s': '5 secs',
        '10s': '10 secs',
        '15s': '15 secs',
        '30s': '30 secs',
        '1m': '1 min',
        '2m': '2 mins',
        '3m': '3 mins',
        '5m': '5 mins',
        '10m': '10 mins',
        '15m': '15 mins',
        '20m': '20 mins',
        '30m': '30 mins',
        '1h': '1 hour',
        '2h': '2 hours',
        '3h': '3 hours',
        '4h': '4 hours',
        '8h': '8 hours',
        '1d': '1 day',
        '1w': '1 week',
        '1M': '1 month'
    }
    
    # IB支持的数据类型
    WHAT_TO_SHOW_OPTIONS = [
        'TRADES',           # 成交数据
        'MIDPOINT',         # 中间价
        'BID',              # 买价
        'ASK',              # 卖价
        'BID_ASK',          # 买卖价
        'HISTORICAL_VOLATILITY',  # 历史波动率
        'OPTION_IMPLIED_VOLATILITY',  # 隐含波动率
        'ADJUSTED_LAST',    # 调整后的最新价
    ]
    
    def __init__(self, connector: IBConnector):
        """
        初始化历史数据下载器
        
        Args:
            connector: IB连接器实例
        """
        self.connector = connector
        self.ib = connector.ib
        
        logger.info("历史数据下载器初始化完成")
    
    def download_historical_data(
        self,
        contract: Contract,
        end_datetime: Optional[datetime] = None,
        duration: str = '1 D',
        bar_size: str = '5m',
        what_to_show: str = 'TRADES',
        use_rth: bool = False,
        format_date: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        下载历史数据
        
        Args:
            contract: 合约对象
            end_datetime: 结束时间，None表示当前时间（连续期货合约必须为None）
            duration: 持续时间 (如 '1 D', '2 W', '1 M', '1 Y')
            bar_size: K线周期 (使用BAR_SIZE_MAP中的键)
            what_to_show: 数据类型
            use_rth: 是否只使用常规交易时间 (Regular Trading Hours)
            format_date: 日期格式 (1=yyyyMMdd HH:mm:ss, 2=epoch time)
        
        Returns:
            pd.DataFrame: 历史数据，包含列: date, open, high, low, close, volume, average, barCount
        """
        if not self.connector.is_connected:
            logger.error("未连接到IB Gateway")
            return None
        
        try:
            # 转换bar_size
            ib_bar_size = self.BAR_SIZE_MAP.get(bar_size, bar_size)
            
            # 对于连续期货合约(CONTFUT)，必须使用空字符串作为endDateTime
            if contract.secType == 'CONTFUT':
                end_datetime_str = ''
                logger.info(f"检测到连续期货合约，使用当前时间作为结束时间")
            elif end_datetime is None:
                end_datetime_str = ''
            else:
                end_datetime_str = end_datetime.strftime('%Y%m%d %H:%M:%S')
            
            logger.info(f"开始下载历史数据: {contract.symbol} ({contract.secType}), "
                       f"duration={duration}, bar_size={ib_bar_size}, "
                       f"end={end_datetime_str if end_datetime_str else 'now'}")
            
            # 请求历史数据
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_datetime_str,
                durationStr=duration,
                barSizeSetting=ib_bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date
            )
            
            if not bars:
                logger.warning("未获取到历史数据")
                return None
            
            # 转换为DataFrame
            df = self._bars_to_dataframe(bars)
            
            logger.info(f"成功下载 {len(df)} 条历史数据")
            return df
            
        except Exception as e:
            logger.error(f"下载历史数据失败: {e}")
            return None
    
    def download_historical_data_range(
        self,
        contract: Contract,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = '5m',
        what_to_show: str = 'TRADES',
        use_rth: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        下载指定日期范围的历史数据
        由于IB API限制，可能需要分段下载
        
        注意：对于连续期货合约(CONTFUT)，只能下载从当前时间往回的数据，
        end_date参数会被忽略，使用当前时间。
        
        Args:
            contract: 合约对象
            start_date: 开始日期
            end_date: 结束日期（连续期货合约会被忽略）
            bar_size: K线周期
            what_to_show: 数据类型
            use_rth: 是否只使用常规交易时间
        
        Returns:
            pd.DataFrame: 历史数据
        """
        if not self.connector.is_connected:
            logger.error("未连接到IB Gateway")
            return None
        
        try:
            # 对于连续期货合约，只能从当前时间往回下载
            if contract.secType == 'CONTFUT':
                logger.info("连续期货合约：从当前时间往回下载数据")
                
                # 计算需要下载的天数
                total_days = (datetime.now() - start_date).days
                
                # 根据bar_size确定每次请求的最大天数
                max_days_per_request = self._get_max_days_per_request(bar_size)
                
                # 计算需要的duration
                if total_days <= max_days_per_request:
                    duration = f"{total_days} D"
                else:
                    # 如果超过限制，使用最大值
                    duration = f"{max_days_per_request} D"
                    logger.warning(f"请求天数({total_days})超过限制({max_days_per_request})，"
                                 f"将只下载最近{max_days_per_request}天的数据")
                
                # 下载数据（end_datetime会被自动设为空字符串）
                df = self.download_historical_data(
                    contract=contract,
                    end_datetime=None,  # 连续期货必须为None
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth
                )
                
                if df is None or df.empty:
                    logger.warning("未获取到任何历史数据")
                    return None
                
                # 过滤到指定开始日期（确保start_date没有时区信息）
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                df = df[df['date'] >= start_date]
                
                logger.info(f"成功下载连续期货历史数据: {len(df)} 条记录")
                return df
            
            else:
                # 普通合约：可以指定结束时间，支持分段下载
                total_days = (end_date - start_date).days
                max_days_per_request = self._get_max_days_per_request(bar_size)
                
                all_data = []
                current_end = end_date
                
                while current_end > start_date:
                    # 计算本次请求的持续时间
                    days_to_request = min(max_days_per_request, (current_end - start_date).days)
                    duration = f"{days_to_request} D"
                    
                    logger.info(f"下载数据段: 结束于 {current_end}, 持续 {duration}")
                    
                    # 下载数据
                    df = self.download_historical_data(
                        contract=contract,
                        end_datetime=current_end,
                        duration=duration,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        use_rth=use_rth
                    )
                    
                    if df is not None and not df.empty:
                        all_data.append(df)
                    
                    # 更新下一次请求的结束时间
                    current_end = current_end - timedelta(days=days_to_request)
                    
                    # 避免请求过快
                    time.sleep(1)
                
                if not all_data:
                    logger.warning("未获取到任何历史数据")
                    return None
                
                # 合并所有数据
                result = pd.concat(all_data, ignore_index=True)
                
                # 去重并排序
                result = result.drop_duplicates(subset=['date'])
                result = result.sort_values('date').reset_index(drop=True)
                
                # 过滤到指定日期范围（确保日期没有时区信息）
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
                    
                result = result[
                    (result['date'] >= start_date) &
                    (result['date'] <= end_date)
                ]
                
                logger.info(f"成功下载完整历史数据: {len(result)} 条记录")
                return result
            
        except Exception as e:
            logger.error(f"下载历史数据范围失败: {e}")
            return None
    
    def _bars_to_dataframe(self, bars: BarDataList) -> pd.DataFrame:
        """
        将IB的BarDataList转换为DataFrame
        
        Args:
            bars: IB返回的K线数据
        
        Returns:
            pd.DataFrame: 转换后的数据
        """
        data = []
        for bar in bars:
            data.append({
                'date': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'average': bar.average,
                'barCount': bar.barCount
            })
        
        df = pd.DataFrame(data)
        
        # 确保date列是datetime类型，并移除时区信息以便比较
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            # 如果有时区信息，转换为UTC后移除时区
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        return df
    
    def _get_max_days_per_request(self, bar_size: str) -> int:
        """
        根据K线周期确定每次请求的最大天数
        IB API对不同周期有不同的限制
        
        Args:
            bar_size: K线周期
        
        Returns:
            int: 最大天数
        """
        # IB API的限制（近似值）
        limits = {
            '1s': 1,
            '5s': 1,
            '10s': 1,
            '15s': 1,
            '30s': 1,
            '1m': 7,
            '2m': 7,
            '3m': 7,
            '5m': 30,
            '10m': 30,
            '15m': 30,
            '20m': 30,
            '30m': 30,
            '1h': 365,
            '2h': 365,
            '3h': 365,
            '4h': 365,
            '8h': 365,
            '1d': 365,
            '1w': 365 * 2,
            '1M': 365 * 5
        }
        
        return limits.get(bar_size, 30)
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        include_index: bool = False
    ):
        """
        保存数据到CSV文件
        
        Args:
            df: 数据DataFrame
            filepath: 文件路径
            include_index: 是否包含索引
        """
        try:
            df.to_csv(filepath, index=include_index)
            logger.info(f"数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
    
    def save_to_parquet(
        self,
        df: pd.DataFrame,
        filepath: str
    ):
        """
        保存数据到Parquet文件（更高效的存储格式）
        
        Args:
            df: 数据DataFrame
            filepath: 文件路径
        """
        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"数据已保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存Parquet文件失败: {e}")


def download_futures_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    bar_size: str = '5m',
    exchange: str = 'CME',
    host: str = '127.0.0.1',
    port: int = 4001,
    save_path: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    便捷函数：下载期货历史数据
    
    Args:
        symbol: 期货代码 (如 'ES', 'NQ')
        start_date: 开始日期
        end_date: 结束日期
        bar_size: K线周期
        exchange: 交易所
        host: IB Gateway主机
        port: IB Gateway端口
        save_path: 保存路径（可选）
    
    Returns:
        pd.DataFrame: 历史数据
    """
    from .ib_connector import create_futures_contract
    
    with IBConnector(host=host, port=port) as connector:
        # 创建合约
        contract = create_futures_contract(symbol, exchange)
        qualified = connector.qualify_contract(contract)
        
        if not qualified:
            logger.error(f"无法验证合约: {symbol}")
            return None
        
        # 下载数据
        downloader = IBHistoricalDataDownloader(connector)
        df = downloader.download_historical_data_range(
            contract=qualified,
            start_date=start_date,
            end_date=end_date,
            bar_size=bar_size
        )
        
        # 保存数据
        if df is not None and save_path:
            if save_path.endswith('.parquet'):
                downloader.save_to_parquet(df, save_path)
            else:
                downloader.save_to_csv(df, save_path)
        
        return df


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 下载ES期货最近30天的5分钟数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    df = download_futures_data(
        symbol='ES',
        start_date=start_date,
        end_date=end_date,
        bar_size='5m',
        exchange='CME',
        save_path='data/raw/ES_5m.csv'
    )
    
    if df is not None:
        print(f"\n下载完成！共 {len(df)} 条数据")
        print(f"\n数据预览:")
        print(df.head())
        print(f"\n数据统计:")
        print(df.describe())