# IB Gateway数据模块

这个模块提供了与Interactive Brokers Gateway的完整集成，包括连接管理、历史数据下载、实时数据流和数据存储功能。

## 功能特性

- ✅ **连接管理**: 稳定的IB Gateway连接，支持自动重连
- ✅ **历史数据**: 下载各种时间周期的历史K线数据
- ✅ **实时数据**: 订阅和处理实时K线数据流
- ✅ **数据存储**: 统一的数据存储接口，支持CSV、Parquet、HDF5格式
- ✅ **合约支持**: 支持期货(FUT)、连续期货(CONTFUT)、股票(STK)等多种合约类型

## 安装依赖

```bash
pip install ib_insync nest_asyncio pandas pyarrow
```

## 快速开始

### 1. 连接到IB Gateway

```python
from src.data import IBConnector

# 创建连接（使用上下文管理器自动管理连接）
with IBConnector(host='127.0.0.1', port=4001) as connector:
    # 获取服务器时间
    server_time = connector.get_current_time()
    print(f"IB服务器时间: {server_time}")
```

### 2. 下载历史数据

```python
from src.data import download_futures_data
from datetime import datetime, timedelta

# 下载ES期货最近30天的5分钟数据
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

df = download_futures_data(
    symbol='ES',
    start_date=start_date,
    end_date=end_date,
    bar_size='5m',
    exchange='CME',
    save_path='data/raw/ES_5m.parquet'
)

print(f"下载完成！共 {len(df)} 条数据")
print(df.head())
```

### 3. 订阅实时数据

```python
from src.data import IBConnector, IBRealtimeDataStreamer, create_futures_contract
import time

with IBConnector(host='127.0.0.1', port=4001) as connector:
    # 创建合约
    contract = create_futures_contract('ES', 'CME')
    qualified = connector.qualify_contract(contract)
    
    # 创建实时数据流
    streamer = IBRealtimeDataStreamer(connector)
    
    # 定义回调函数
    def on_new_bar(contract, bar_data):
        print(f"新K线: {bar_data['date']}")
        print(f"OHLC: {bar_data['open']:.2f}, {bar_data['high']:.2f}, "
              f"{bar_data['low']:.2f}, {bar_data['close']:.2f}")
    
    # 订阅实时数据
    streamer.subscribe_realtime_bars(
        contract=qualified,
        bar_size=5,  # 5秒K线
        callback=on_new_bar
    )
    
    # 运行60秒
    time.sleep(60)
    
    # 获取缓冲区数据
    df = streamer.get_buffer_data(qualified)
    print(f"缓冲区数据: {len(df)} 条")
```

### 4. 使用数据存储

```python
from src.data import DataStorage
import pandas as pd

# 创建存储管理器
storage = DataStorage(base_dir='data')

# 保存原始数据
metadata = {
    'source': 'IB Gateway',
    'symbol': 'ES',
    'interval': '5m',
    'records': len(df)
}

filepath = storage.save_raw_data(
    df=df,
    symbol='ES',
    interval='5m',
    format='parquet',
    metadata=metadata
)

# 加载最新数据
df_loaded = storage.load_latest_raw_data(symbol='ES', interval='5m')

# 列出所有文件
files = storage.list_raw_data(symbol='ES')
for file in files:
    print(f"{file['filename']}: {file['size']} bytes, {file['modified']}")

# 获取存储信息
info = storage.get_storage_info()
print(f"总存储空间: {info['total_size_mb']:.2f} MB")
```

## 合约类型

### 期货合约 (FUT)

```python
from src.data import create_futures_contract

# 创建指定到期日的期货合约
contract = create_futures_contract(
    symbol='ES',
    exchange='CME',
    expiry='202503'  # 2025年3月到期
)
```

### 连续期货合约 (CONTFUT)

```python
from src.data import create_continuous_futures_contract

# 创建连续期货合约（自动滚动到最活跃合约）
contract = create_continuous_futures_contract(
    symbol='ES',
    exchange='CME'
)
```

### 股票合约 (STK)

```python
from src.data import create_stock_contract

# 创建股票合约
contract = create_stock_contract(
    symbol='AAPL',
    exchange='SMART',  # 智能路由
    currency='USD'
)
```

## 支持的时间周期

历史数据下载支持以下时间周期：

- **秒级**: `1s`, `5s`, `10s`, `15s`, `30s`
- **分钟级**: `1m`, `2m`, `3m`, `5m`, `10m`, `15m`, `20m`, `30m`
- **小时级**: `1h`, `2h`, `3h`, `4h`, `8h`
- **日级**: `1d`
- **周级**: `1w`
- **月级**: `1M`

## 高级用法

### 批量下载多个品种

```python
from src.data import IBConnector, IBHistoricalDataDownloader, create_futures_contract
from datetime import datetime, timedelta

symbols = ['ES', 'NQ', 'YM', 'RTY']  # 标普、纳指、道指、罗素
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

with IBConnector(host='127.0.0.1', port=4001) as connector:
    downloader = IBHistoricalDataDownloader(connector)
    
    for symbol in symbols:
        contract = create_futures_contract(symbol, 'CME')
        qualified = connector.qualify_contract(contract)
        
        if qualified:
            df = downloader.download_historical_data_range(
                contract=qualified,
                start_date=start_date,
                end_date=end_date,
                bar_size='5m'
            )
            
            if df is not None:
                filepath = f'data/raw/{symbol}_5m.parquet'
                downloader.save_to_parquet(df, filepath)
                print(f"{symbol}: {len(df)} 条数据已保存")
```

### 实时数据记录到文件

```python
from src.data import IBConnector, IBRealtimeDataStreamer, RealtimeDataRecorder
from src.data import create_futures_contract
import time

with IBConnector(host='127.0.0.1', port=4001) as connector:
    # 创建合约
    contract = create_futures_contract('ES', 'CME')
    qualified = connector.qualify_contract(contract)
    
    # 创建流处理器和记录器
    streamer = IBRealtimeDataStreamer(connector, buffer_size=1000)
    recorder = RealtimeDataRecorder(streamer, save_interval=100)
    
    # 开始记录
    recorder.start_recording(
        contract=qualified,
        filepath='data/raw/ES_realtime.csv',
        bar_size=5
    )
    
    # 运行一段时间
    print("开始记录实时数据...")
    time.sleep(3600)  # 记录1小时
    
    # 停止
    streamer.unsubscribe_all()
    print("记录完成")
```

## 配置说明

### IB Gateway端口

- **4001**: TWS Live Trading（实盘）
- **4001**: TWS Paper Trading（模拟盘）
- **4003**: IB Gateway Live Trading（实盘）
- **4004**: IB Gateway Paper Trading（模拟盘）

### 数据类型 (what_to_show)

- `TRADES`: 成交数据（默认）
- `MIDPOINT`: 中间价
- `BID`: 买价
- `ASK`: 卖价
- `BID_ASK`: 买卖价
- `HISTORICAL_VOLATILITY`: 历史波动率
- `OPTION_IMPLIED_VOLATILITY`: 隐含波动率

## 注意事项

1. **连接前提**: 确保IB Gateway或TWS已启动并配置好API连接
2. **数据限制**: IB对历史数据请求有频率限制，建议添加适当延迟
3. **市场数据**: 需要订阅相应的市场数据才能获取实时数据
4. **时区处理**: 所有时间默认使用UTC时区
5. **错误处理**: 建议在生产环境中添加完善的错误处理和重连机制

## 故障排查

### 连接失败

```python
# 检查IB Gateway是否运行
# 检查端口号是否正确
# 检查API设置中是否启用了Socket客户端
```

### 数据为空

```python
# 检查合约是否有效
qualified = connector.qualify_contract(contract)
if not qualified:
    print("合约验证失败，请检查合约参数")

# 检查市场数据订阅
details = connector.get_contract_details(contract)
print(details)
```

### 实时数据不更新

```python
# 检查市场是否开盘
# 检查是否有市场数据订阅权限
# 检查回调函数是否正确设置
```

## 更多示例

查看各模块文件底部的`if __name__ == '__main__':`部分获取更多使用示例。

## 支持

如有问题，请查看：
- [IB API文档](https://interactivebrokers.github.io/tws-api/)
- [ib_insync文档](https://ib-insync.readthedocs.io/)