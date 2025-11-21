"""
IB Gateway连接器
提供与Interactive Brokers Gateway的连接和基础功能
"""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from ib_insync import IB, Contract, util
import nest_asyncio

# 允许嵌套事件循环
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class IBConnector:
    """IB Gateway连接器基础类"""
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 4001,  # TWS Paper Trading默认端口
        client_id: int = 1,
        timeout: int = 10
    ):
        """
        初始化IB连接器
        
        Args:
            host: IB Gateway主机地址
            port: IB Gateway端口 (4001=TWS Live, 4001=TWS Paper, 4003=Gateway Live, 4004=Gateway Paper)
            client_id: 客户端ID
            timeout: 连接超时时间(秒)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        
        self.ib = IB()
        self.is_connected = False
        
        # 设置错误处理
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnected
        
        logger.info(f"IB连接器初始化完成: {host}:{port}, client_id={client_id}")
    
    def connect(self) -> bool:
        """
        连接到IB Gateway
        
        Returns:
            bool: 连接是否成功
        """
        try:
            if self.is_connected:
                logger.warning("已经连接到IB Gateway")
                return True
            
            logger.info(f"正在连接到IB Gateway: {self.host}:{self.port}")
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            
            self.is_connected = True
            logger.info("成功连接到IB Gateway")
            
            # 获取账户信息
            accounts = self.ib.managedAccounts()
            logger.info(f"可用账户: {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"连接IB Gateway失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开与IB Gateway的连接"""
        if self.is_connected:
            try:
                self.ib.disconnect()
                self.is_connected = False
                logger.info("已断开与IB Gateway的连接")
            except Exception as e:
                logger.error(f"断开连接时出错: {e}")
    
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract):
        """错误处理回调"""
        logger.error(f"IB错误 [reqId={reqId}, code={errorCode}]: {errorString}")
        if contract:
            logger.error(f"相关合约: {contract}")
    
    def _on_disconnected(self):
        """断开连接回调"""
        self.is_connected = False
        logger.warning("与IB Gateway的连接已断开")
    
    def create_contract(
        self,
        symbol: str,
        sec_type: str = 'FUT',
        exchange: str = 'CME',
        currency: str = 'USD',
        last_trade_date: Optional[str] = None,
        multiplier: Optional[str] = None
    ) -> Contract:
        """
        创建合约对象
        
        Args:
            symbol: 合约代码 (如 'ES', 'NQ')
            sec_type: 证券类型 ('STK'=股票, 'FUT'=期货, 'CONTFUT'=连续期货, 'OPT'=期权, 'CASH'=外汇)
            exchange: 交易所 ('CME', 'GLOBEX', 'SMART'等)
            currency: 货币
            last_trade_date: 最后交易日 (期货合约到期日，格式: YYYYMMDD，CONTFUT不需要)
            multiplier: 合约乘数
        
        Returns:
            Contract: IB合约对象
        """
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        if last_trade_date:
            contract.lastTradeDateOrContractMonth = last_trade_date
        
        if multiplier:
            contract.multiplier = multiplier
        
        return contract
    
    def qualify_contract(self, contract: Contract) -> Optional[Contract]:
        """
        验证并补全合约信息
        
        Args:
            contract: 待验证的合约
        
        Returns:
            Contract: 验证后的合约，如果失败返回None
        """
        if not self.is_connected:
            logger.error("未连接到IB Gateway")
            return None
        
        try:
            qualified = self.ib.qualifyContracts(contract)
            if qualified:
                logger.info(f"合约验证成功: {qualified[0]}")
                return qualified[0]
            else:
                logger.error(f"合约验证失败: {contract}")
                return None
        except Exception as e:
            logger.error(f"合约验证出错: {e}")
            return None
    
    def get_contract_details(self, contract: Contract) -> List[Any]:
        """
        获取合约详细信息
        
        Args:
            contract: 合约对象
        
        Returns:
            List: 合约详细信息列表
        """
        if not self.is_connected:
            logger.error("未连接到IB Gateway")
            return []
        
        try:
            details = self.ib.reqContractDetails(contract)
            logger.info(f"获取到 {len(details)} 个合约详情")
            return details
        except Exception as e:
            logger.error(f"获取合约详情失败: {e}")
            return []
    
    def get_current_time(self) -> Optional[datetime]:
        """
        获取IB服务器当前时间
        
        Returns:
            datetime: 服务器时间，失败返回None
        """
        if not self.is_connected:
            logger.error("未连接到IB Gateway")
            return None
        
        try:
            server_time = self.ib.reqCurrentTime()
            logger.info(f"IB服务器时间: {server_time}")
            return server_time
        except Exception as e:
            logger.error(f"获取服务器时间失败: {e}")
            return None
    
    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()
    
    def __del__(self):
        """析构函数"""
        self.disconnect()


def create_futures_contract(
    symbol: str,
    exchange: str = 'CME',
    expiry: Optional[str] = None
) -> Contract:
    """
    便捷函数：创建期货合约
    
    Args:
        symbol: 期货代码 (如 'ES', 'NQ')
        exchange: 交易所
        expiry: 到期日 (格式: YYYYMM 或 YYYYMMDD)
    
    Returns:
        Contract: 期货合约对象
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'FUT'
    contract.exchange = exchange
    contract.currency = 'USD'
    
    if expiry:
        contract.lastTradeDateOrContractMonth = expiry
    
    return contract


def create_stock_contract(
    symbol: str,
    exchange: str = 'SMART',
    currency: str = 'USD'
) -> Contract:
    """
    便捷函数：创建股票合约
    
    Args:
        symbol: 股票代码
        exchange: 交易所 (通常使用'SMART'自动路由)
        currency: 货币
    
    Returns:
        Contract: 股票合约对象
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = exchange
    contract.currency = currency
    
    return contract


def create_continuous_futures_contract(
    symbol: str,
    exchange: str = 'CME'
) -> Contract:
    """
    便捷函数：创建连续期货合约
    连续期货合约会自动滚动到最活跃的合约月份
    
    Args:
        symbol: 期货代码 (如 'ES', 'NQ')
        exchange: 交易所
    
    Returns:
        Contract: 连续期货合约对象
    """
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'CONTFUT'
    contract.exchange = exchange
    contract.currency = 'USD'
    
    return contract


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 使用上下文管理器
    with IBConnector(host='127.0.0.1', port=4001) as connector:
        # 获取服务器时间
        server_time = connector.get_current_time()
        
        # 创建ES期货合约
        es_contract = create_futures_contract('ES', 'CME')
        qualified = connector.qualify_contract(es_contract)
        
        if qualified:
            # 获取合约详情
            details = connector.get_contract_details(qualified)
            for detail in details:
                print(f"合约: {detail.contract}")
                print(f"交易时间: {detail.tradingHours}")