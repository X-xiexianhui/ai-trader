"""
期货合约历史数据下载器
支持下载具体合约并拼接成连续数据
"""

import logging
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

from .ib_connector import IBConnector
from .ib_historical_data import IBHistoricalDataDownloader

logger = logging.getLogger(__name__)


class FuturesContractDownloader:
    """期货合约下载器"""
    
    def __init__(
        self,
        connector: IBConnector,
        downloader: IBHistoricalDataDownloader
    ):
        """
        初始化期货合约下载器
        
        Args:
            connector: IB连接器
            downloader: 历史数据下载器
        """
        self.connector = connector
        self.downloader = downloader
    
    def generate_quarterly_contracts(
        self,
        symbol: str,
        start_year: int,
        end_year: int,
        start_month: int = 1
    ) -> List[Tuple[str, datetime, datetime]]:
        """
        生成季度期货合约列表
        
        季度合约月份：3月(H)、6月(M)、9月(U)、12月(Z)
        每个合约通常在到期前3个月开始活跃
        
        Args:
            symbol: 期货代码（如'MES', 'ES'）
            start_year: 开始年份
            end_year: 结束年份
            start_month: 开始月份（默认1月）
        
        Returns:
            List of (contract_code, start_date, end_date)
        """
        contracts = []
        months = [3, 6, 9, 12]
        
        for year in range(start_year, end_year + 1):
            for month in months:
                # 跳过起始日期之前的合约
                if year == start_year and month < start_month:
                    continue
                
                # 合约代码：YYYYMM
                contract_code = f"{year}{month:02d}"
                
                # 计算合约到期日（第三个周五，简化为第15日）
                expiry_date = datetime(year, month, 15)
                
                # 合约活跃期：到期前3个月开始活跃
                start_date = expiry_date - timedelta(days=90)
                end_date = expiry_date
                
                contracts.append((contract_code, start_date, end_date))
        
        logger.info(f"生成 {len(contracts)} 个{symbol}合约")
        return contracts
    
    def download_specific_contract(
        self,
        symbol: str,
        contract_code: str,
        start_date: datetime,
        end_date: datetime,
        exchange: str = 'CME',
        bar_size: str = '5m',
        use_rth: bool = True,
        target_tz: str = None
    ) -> pd.DataFrame:
        """
        下载特定合约的历史数据
        
        Args:
            symbol: 期货代码（如'MES', 'ES'）
            contract_code: 合约代码 (YYYYMM格式)
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            bar_size: K线周期
            use_rth: 是否只使用常规交易时间（默认True，避免收盘时段缺失被误删）
        
        Returns:
            pd.DataFrame: 历史数据
        """
        logger.info(f"=" * 60)
        logger.info(f"下载合约: {symbol}{contract_code}")
        logger.info(f"时间范围: {start_date.date()} 到 {end_date.date()}")
        
        try:
            # 创建期货合约
            from ib_insync import Contract
            contract = Contract()
            contract.symbol = symbol
            contract.secType = 'FUT'
            contract.exchange = exchange
            contract.currency = 'USD'
            contract.lastTradeDateOrContractMonth = contract_code
            
            # 判断是否为当前活跃合约
            contract_date = datetime.strptime(contract_code, '%Y%m')
            now = datetime.now()
            
            # 计算当前应该活跃的合约（最近的未来季度月）
            current_year = now.year
            current_month = now.month
            
            # 找到当前活跃的合约月份（下一个季度月）
            quarter_months = [3, 6, 9, 12]
            active_contract_month = None
            active_contract_year = current_year
            
            for month in quarter_months:
                if month > current_month:
                    active_contract_month = month
                    break
            
            if active_contract_month is None:
                # 如果当前月份已过12月，则活跃合约是下一年的3月
                active_contract_month = 3
                active_contract_year = current_year + 1
            
            # 判断是否为当前活跃合约
            is_active_contract = (contract_date.year == active_contract_year and
                                 contract_date.month == active_contract_month)
            
            if is_active_contract:
                # 当前活跃合约：不设置includeExpired
                logger.info(f"当前活跃合约 {contract_code}，不设置includeExpired")
            else:
                # 历史合约或未来合约：需要includeExpired
                contract.includeExpired = True
                logger.info(f"历史合约 {contract_code}，设置includeExpired=True")
            
            # 验证合约
            qualified = self.connector.qualify_contract(contract)
            
            if not qualified:
                logger.warning(f"无法验证合约 {symbol}{contract_code}，可能该合约不存在或数据不可用")
                return pd.DataFrame()
            
            logger.info(f"合约验证成功: {qualified}")
            
            # 下载数据
            df = self.downloader.download_historical_data_range(
                contract=qualified,
                start_date=start_date,
                end_date=end_date,
                bar_size=bar_size,
                what_to_show='TRADES',
                use_rth=use_rth,
                target_tz=target_tz
            )
            
            if df is not None and not df.empty:
                # 添加合约标识
                df['contract'] = f"{symbol}{contract_code}"
                logger.info(f"成功下载 {len(df)} 条数据")
                return df
            else:
                logger.warning(f"未获取到数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"下载合约 {symbol}{contract_code} 失败: {e}")
            return pd.DataFrame()
    
    def stitch_contracts(
        self,
        all_data: List[pd.DataFrame],
        method: str = 'ratio'
    ) -> pd.DataFrame:
        """
        拼接多个合约的数据，处理换月跳空
        
        Args:
            all_data: 所有合约的数据列表
            method: 拼接方法
                - 'simple': 简单拼接，保留原始价格
                - 'ratio': 比例调整法（Panama Canal方法）
        
        Returns:
            pd.DataFrame: 拼接后的连续数据
        """
        if not all_data:
            return pd.DataFrame()
        
        logger.info(f"开始拼接 {len(all_data)} 个合约的数据，方法: {method}")
        
        if method == 'simple':
            # 简单拼接，不做调整
            result = pd.concat(all_data, ignore_index=True)
            result = result.sort_values('date').reset_index(drop=True)
            result = result.drop_duplicates(subset=['date'], keep='first')
            logger.info(f"简单拼接完成，共 {len(result)} 条记录")
            return result
        
        elif method == 'ratio':
            # 比例调整法（向后调整）
            result = all_data[-1].copy()  # 从最新合约开始
            
            for i in range(len(all_data) - 2, -1, -1):
                current_df = all_data[i].copy()
                next_df = result.copy()
                
                # 找到换月点（当前合约的最后一天）
                if current_df.empty or next_df.empty:
                    continue
                
                rollover_date = current_df['date'].max()
                
                # 找到换月时两个合约的价格
                current_last_price = current_df[current_df['date'] == rollover_date]['close'].iloc[-1]
                next_first_price = next_df[next_df['date'] >= rollover_date]['close'].iloc[0]
                
                # 计算调整比例
                if current_last_price > 0:
                    ratio = next_first_price / current_last_price
                    
                    # 调整当前合约的所有价格
                    price_cols = ['open', 'high', 'low', 'close', 'average']
                    for col in price_cols:
                        if col in current_df.columns:
                            current_df[col] = current_df[col] * ratio
                    
                    logger.info(f"合约 {current_df['contract'].iloc[0]} 调整比例: {ratio:.6f}")
                
                # 合并数据
                result = pd.concat([current_df, result], ignore_index=True)
            
            result = result.sort_values('date').reset_index(drop=True)
            result = result.drop_duplicates(subset=['date'], keep='first')
            logger.info(f"比例调整拼接完成，共 {len(result)} 条记录")
            return result
        
        else:
            logger.error(f"不支持的拼接方法: {method}")
            return pd.DataFrame()
    
    def download_contracts_range(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        exchange: str = 'CME',
        bar_size: str = '5m',
        stitch_method: str = 'ratio',
        save_dir: Optional[str] = None,
        max_future_months: int = 3,
        use_rth: bool = True,
        target_tz: str = None
    ) -> Optional[pd.DataFrame]:
        """
        下载指定日期范围的期货合约数据并拼接
        
        Args:
            symbol: 期货代码（如'MES', 'ES'）
            start_date: 开始日期
            end_date: 结束日期
            exchange: 交易所
            bar_size: K线周期
            stitch_method: 拼接方法
            save_dir: 保存目录（可选）
            max_future_months: 最多下载未来几个月的合约
            use_rth: 是否只使用常规交易时间（默认True，推荐）
                    True: 只下载交易时段数据，避免收盘时段缺失被数据清洗误删
                    False: 下载24小时数据，包括盘前盘后
        
        Returns:
            pd.DataFrame: 拼接后的连续数据
        """
        logger.info("=" * 80)
        logger.info(f"开始下载{symbol}期货历史数据（具体合约方式）")
        logger.info(f"时间范围: {start_date.date()} 到 {end_date.date()}")
        logger.info(f"K线周期: {bar_size}")
        logger.info(f"拼接方法: {stitch_method}")
        logger.info("=" * 80)
        
        try:
            # 生成合约列表
            contracts = self.generate_quarterly_contracts(
                symbol=symbol,
                start_year=start_date.year,
                end_year=end_date.year + 1,
                start_month=start_date.month
            )
            
            # 过滤到指定日期范围内的合约
            now = datetime.now()
            relevant_contracts = []
            
            for code, start, end in contracts:
                # 跳过太远的未来合约
                contract_date = datetime.strptime(code, '%Y%m')
                months_ahead = (contract_date.year - now.year) * 12 + (contract_date.month - now.month)
                
                if months_ahead > max_future_months:
                    logger.info(f"跳过未来合约 {code}（{months_ahead}个月后到期）")
                    continue
                
                # 检查是否在日期范围内
                if end >= start_date and start <= end_date:
                    # 对于已过期的合约，下载完整活跃期
                    # 对于未来合约，使用用户指定的日期范围
                    if contract_date < now:
                        # 已过期合约：下载完整活跃期
                        actual_start = start
                        actual_end = end
                        logger.info(f"合约 {code} 已过期，下载完整活跃期: {actual_start.date()} 到 {actual_end.date()}")
                    else:
                        # 未来合约：使用用户指定的日期范围
                        actual_start = max(start, start_date)
                        actual_end = min(end, end_date)
                        logger.info(f"合约 {code} 未来合约，下载指定范围: {actual_start.date()} 到 {actual_end.date()}")
                    
                    # 确保start < end
                    if actual_start < actual_end:
                        relevant_contracts.append((code, actual_start, actual_end))
                        logger.info(f"将下载合约 {code}，时间范围: {actual_start.date()} 到 {actual_end.date()}")
                    else:
                        logger.warning(f"跳过合约 {code}：时间范围无效")
            
            logger.info(f"需要下载 {len(relevant_contracts)} 个合约")
            
            if not relevant_contracts:
                logger.warning("没有需要下载的合约")
                return None
            
            # 下载所有合约
            all_data = []
            for i, (contract_code, contract_start, contract_end) in enumerate(relevant_contracts, 1):
                logger.info(f"\n进度: {i}/{len(relevant_contracts)}")
                
                df = self.download_specific_contract(
                    symbol=symbol,
                    contract_code=contract_code,
                    start_date=contract_start,
                    end_date=contract_end,
                    exchange=exchange,
                    bar_size=bar_size,
                    use_rth=use_rth,
                    target_tz=target_tz
                )
                
                if not df.empty:
                    all_data.append(df)
                    
                    # 保存单个合约数据
                    if save_dir:
                        import os
                        os.makedirs(save_dir, exist_ok=True)
                        contract_file = os.path.join(save_dir, f'{symbol}_{contract_code}_{bar_size}.csv')
                        df.to_csv(contract_file, index=False)
                        logger.info(f"合约数据已保存: {contract_file}")
                
                # 避免请求过快
                if i < len(relevant_contracts):
                    logger.info("等待1秒...")
                    import time
                    time.sleep(1)
            
            if not all_data:
                logger.error("未能下载到任何数据")
                return None
            
            # 拼接所有合约数据
            logger.info("\n" + "=" * 80)
            logger.info("开始拼接合约数据...")
            stitched_df = self.stitch_contracts(all_data, method=stitch_method)
            
            if stitched_df.empty:
                logger.error("拼接失败")
                return None
            
            # 保存拼接后的数据
            if save_dir:
                import os
                output_file = os.path.join(save_dir, f'{symbol}_stitched_{bar_size}_{stitch_method}.csv')
                stitched_df.to_csv(output_file, index=False)
                logger.info(f"拼接数据已保存到: {output_file}")
            
            logger.info("=" * 80)
            logger.info("下载和拼接完成！")
            logger.info(f"总记录数: {len(stitched_df):,}")
            logger.info(f"时间范围: {stitched_df['date'].min()} 到 {stitched_df['date'].max()}")
            logger.info("=" * 80)
            
            return stitched_df
            
        except Exception as e:
            logger.error(f"下载过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None