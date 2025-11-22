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
        format_date: int = 1,
        target_tz: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载历史数据
        
        Args:
            contract: 合约对象
            end_datetime: 结束时间，None表示当前时间（连续期货合约必须为None）
            duration: 持续时间 (如 '1 D', '2 W', '1 M', '10 Y')
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
            df = self._bars_to_dataframe(bars, target_tz=target_tz)
            
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
        use_rth: bool = False,
        target_tz: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载指定日期范围的历史数据
        智能处理超长期数据下载，支持最多68年
        
        注意：对于连续期货合约(CONTFUT)，只能从当前时间往回下载，
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
                logger.info(f"总天数: {total_days} 天 ({total_days/365:.1f} 年)")
                
                # 对于连续期货，IB实际限制较严格，需要分段下载
                # 每次最多下载90天（约3个月）
                max_days = 90
                
                if total_days <= max_days:
                    # 单次请求即可
                    duration = self._calculate_optimal_duration(total_days)
                    logger.info(f"单次请求: duration={duration}")
                    
                    df = self.download_historical_data(
                        contract=contract,
                        end_datetime=None,
                        duration=duration,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        use_rth=use_rth,
                        target_tz=target_tz
                    )
                else:
                    # 分段下载
                    logger.info(f"需要分段下载: 每段{max_days}天")
                    all_data = []
                    years_needed = (total_days // 365) + 1
                    
                    for year in range(years_needed):
                        logger.info(f"\n[年份 {year + 1}/{years_needed}] 下载第{year + 1}年的数据")
                        
                        # 计算这一段的天数
                        remaining_days = total_days - (year * max_days)
                        days_this_segment = min(max_days, remaining_days)
                        
                        if days_this_segment <= 0:
                            break
                        
                        duration = self._calculate_optimal_duration(days_this_segment)
                        logger.info(f"Duration: {duration}")
                        
                        # 注意：连续期货只能用空字符串作为end_datetime
                        # 但我们无法指定历史的结束时间，所以只能下载最近的数据
                        # 这是连续期货的限制
                        if year == 0:
                            # 第一段：从现在往回
                            df_segment = self.download_historical_data(
                                contract=contract,
                                end_datetime=None,
                                duration=duration,
                                bar_size=bar_size,
                                what_to_show=what_to_show,
                                use_rth=use_rth,
                                target_tz=target_tz
                            )
                        else:
                            # 后续段：连续期货无法指定历史结束时间
                            # 这是IB API的限制
                            logger.warning(f"连续期货合约无法下载{year}年前的历史数据")
                            logger.warning("建议使用具体合约方式下载长期历史数据")
                            break
                        
                        if df_segment is not None and not df_segment.empty:
                            all_data.append(df_segment)
                            logger.info(f"[年份 {year + 1}] 获取 {len(df_segment)} 条记录")
                        
                        # 避免请求过快
                        if year < years_needed - 1:
                            time.sleep(2)
                    
                    if not all_data:
                        logger.warning("未获取到任何历史数据")
                        return None
                    
                    # 合并所有数据
                    df = pd.concat(all_data, ignore_index=True)
                    df = df.drop_duplicates(subset=['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                
                if df is None or df.empty:
                    logger.warning("未获取到任何历史数据")
                    return None
                
                # 过滤到指定开始日期
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                df = df[df['date'] >= start_date]
                
                logger.info(f"成功下载连续期货历史数据: {len(df)} 条记录")
                logger.info(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
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
                    
                    # 确保至少请求1天的数据
                    if days_to_request <= 0:
                        logger.warning(f"计算的天数无效: {days_to_request}，跳过")
                        break
                    
                    duration = self._calculate_optimal_duration(days_to_request)
                    
                    logger.info(f"下载数据段: 结束于 {current_end}, 持续 {duration}")
                    
                    # 下载数据
                    df = self.download_historical_data(
                        contract=contract,
                        end_datetime=current_end,
                        duration=duration,
                        bar_size=bar_size,
                        what_to_show=what_to_show,
                        use_rth=use_rth,
                        target_tz=target_tz
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
    
    def _calculate_optimal_duration(self, total_days: int) -> str:
        """
        根据天数智能选择最优的duration字符串
        基于IB API支持的duration单位：S(秒), D(天), W(周), M(月), Y(年)
        
        根据IB API文档，5分钟K线支持：
        - Max Second Duration: 86400 S
        - Max Day Duration: 365 D
        - Max Week Duration: 52 W
        - Max Month Duration: 12 M
        - Max Year Duration: 68 Y
        
        Args:
            total_days: 总天数
        
        Returns:
            str: 最优的duration字符串 (如 "10 Y", "6 M", "30 D")
        """
        # 优先使用更大的单位以提高效率
        years = total_days // 365
        if years > 0 and years <= 68:  # 最多68年
            return f"{years} Y"
        
        # 如果超过68年，使用68年
        if total_days > 365 * 68:
            logger.warning(f"请求天数({total_days})超过最大限制(68年)，将使用68年")
            return "68 Y"
        
        # 对于不足1年的，使用月/周/天
        months = total_days // 30
        if months > 0 and months <= 12:  # 最多12个月
            return f"{months} M"
        
        weeks = total_days // 7
        if weeks > 0 and weeks <= 52:  # 最多52周
            return f"{weeks} W"
        
        # 默认使用天数
        if total_days <= 365:
            return f"{total_days} D"
        
        # 兜底：使用365天
        return "365 D"
    
    def _bars_to_dataframe(self, bars: BarDataList, target_tz: str = None) -> pd.DataFrame:
        """
        将IB的BarDataList转换为DataFrame
        
        Args:
            bars: IB返回的K线数据
            target_tz: 目标时区（如 'US/Eastern'），None 则保持原时区
        
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
        
        # 确保date列是datetime类型
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            
            # 如果指定了目标时区，进行转换
            if target_tz and df['date'].dt.tz is not None:
                import pytz
                df['date'] = df['date'].dt.tz_convert(target_tz).dt.tz_localize(None)
            elif df['date'].dt.tz is not None:
                # 如果没有指定目标时区，移除时区信息
                df['date'] = df['date'].dt.tz_localize(None)
        
        return df
    
    def _get_max_days_per_request(self, bar_size: str) -> int:
        """
        根据K线周期确定每次请求的最大天数
        基于IB API官方文档的限制
        
        参考：https://interactivebrokers.github.io/tws-api/historical_limitations.html
        
        对于5分钟K线，Max Year Duration = 68 Y，因此实际上可以一次请求68年的数据
        
        Args:
            bar_size: K线周期
        
        Returns:
            int: 最大天数
        """
        # IB API的限制（基于官方文档）
        # 大多数bar size支持68年（约24820天）
        limits = {
            '1s': 1,      # Max: 2000 S (约0.5天)
            '5s': 365 * 68,    # Max: 68 Y
            '10s': 365 * 68,   # Max: 68 Y
            '15s': 365 * 68,   # Max: 68 Y
            '30s': 365 * 68,   # Max: 68 Y
            '1m': 365 * 68,    # Max: 68 Y
            '2m': 365 * 68,    # Max: 68 Y
            '3m': 365 * 68,    # Max: 68 Y
            '5m': 365 * 68,    # Max: 68 Y
            '10m': 365 * 68,   # Max: 68 Y
            '15m': 365 * 68,   # Max: 68 Y
            '20m': 365 * 68,   # Max: 68 Y
            '30m': 365 * 68,   # Max: 68 Y
            '1h': 365 * 68,    # Max: 68 Y
            '2h': 365 * 68,    # Max: 68 Y
            '3h': 365 * 68,    # Max: 68 Y
            '4h': 365 * 68,    # Max: 68 Y
            '8h': 365 * 68,    # Max: 68 Y
            '1d': 365 * 68,    # Max: 68 Y
            '1w': 365 * 68,    # Max: 68 Y
            '1M': 365 * 68     # Max: 68 Y
        }
        
        return limits.get(bar_size, 365 * 68)
    
    def download_historical_data_streaming(
        self,
        contract: Contract,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = '5m',
        what_to_show: str = 'TRADES',
        use_rth: bool = False,
        save_path: Optional[str] = None,
        chunk_days: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        流式下载历史数据，边下载边保存
        适合下载长期历史数据，避免内存溢出
        
        Args:
            contract: 合约对象
            start_date: 开始日期
            end_date: 结束日期
            bar_size: K线周期
            what_to_show: 数据类型
            use_rth: 是否只使用常规交易时间
            save_path: 保存路径（如果提供，会边下载边保存）
            chunk_days: 每次下载的天数（None则自动根据bar_size确定）
        
        Returns:
            pd.DataFrame: 完整的历史数据（如果内存允许）
        """
        if not self.connector.is_connected:
            logger.error("未连接到IB Gateway")
            return None
        
        try:
            # 对于连续期货合约的特殊处理
            if contract.secType == 'CONTFUT':
                logger.warning("连续期货合约不支持流式下载，使用标准方法")
                return self.download_historical_data_range(
                    contract=contract,
                    start_date=start_date,
                    end_date=end_date,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth
                )
            
            # 确定每次下载的天数
            if chunk_days is None:
                chunk_days = self._get_max_days_per_request(bar_size)
            
            logger.info(f"开始流式下载历史数据: {contract.symbol}")
            logger.info(f"时间范围: {start_date} 到 {end_date}")
            logger.info(f"分块大小: {chunk_days} 天")
            
            # 准备保存文件
            import os
            file_exists = False
            if save_path:
                file_exists = os.path.exists(save_path)
                logger.info(f"数据将保存到: {save_path}")
            
            all_data = []
            current_end = end_date
            chunk_count = 0
            total_records = 0
            
            # 移除时区信息
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if current_end.tzinfo is not None:
                current_end = current_end.replace(tzinfo=None)
            
            while current_end > start_date:
                chunk_count += 1
                
                # 计算本次请求的持续时间
                days_to_request = min(chunk_days, (current_end - start_date).days)
                if days_to_request <= 0:
                    break
                
                duration = self._calculate_optimal_duration(days_to_request)
                
                logger.info(f"[块 {chunk_count}] 下载数据: 结束于 {current_end}, 持续 {duration}")
                
                # 下载数据块
                df_chunk = self.download_historical_data(
                    contract=contract,
                    end_datetime=current_end,
                    duration=duration,
                    bar_size=bar_size,
                    what_to_show=what_to_show,
                    use_rth=use_rth
                )
                
                if df_chunk is not None and not df_chunk.empty:
                    # 过滤到指定日期范围
                    df_chunk = df_chunk[
                        (df_chunk['date'] >= start_date) &
                        (df_chunk['date'] <= current_end)
                    ]
                    
                    chunk_records = len(df_chunk)
                    total_records += chunk_records
                    logger.info(f"[块 {chunk_count}] 获取 {chunk_records} 条记录，累计 {total_records} 条")
                    
                    # 立即保存到文件
                    if save_path:
                        mode = 'a' if file_exists else 'w'
                        header = not file_exists
                        
                        if save_path.endswith('.csv'):
                            df_chunk.to_csv(save_path, mode=mode, header=header, index=False)
                        elif save_path.endswith('.parquet'):
                            # Parquet不支持追加，需要特殊处理
                            if file_exists:
                                # 读取现有数据，合并后保存
                                existing_df = pd.read_parquet(save_path)
                                combined_df = pd.concat([existing_df, df_chunk], ignore_index=True)
                                combined_df.to_parquet(save_path, index=False)
                            else:
                                df_chunk.to_parquet(save_path, index=False)
                        
                        file_exists = True
                        logger.info(f"[块 {chunk_count}] 数据已保存")
                    
                    # 保存到内存（可选）
                    all_data.append(df_chunk)
                else:
                    logger.warning(f"[块 {chunk_count}] 未获取到数据")
                
                # 更新下一次请求的结束时间
                current_end = current_end - timedelta(days=days_to_request)
                
                # 避免请求过快
                time.sleep(1)
            
            logger.info(f"流式下载完成！共 {chunk_count} 个块，{total_records} 条记录")
            
            # 如果需要返回完整数据
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                result = result.drop_duplicates(subset=['date'])
                result = result.sort_values('date').reset_index(drop=True)
                return result
            else:
                logger.warning("未获取到任何数据")
                return None
            
        except Exception as e:
            logger.error(f"流式下载失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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