# 数据采集模块文档

## 目录

- [概述](#概述)
- [模块架构](#模块架构)
- [核心组件](#核心组件)
- [使用指南](#使用指南)
- [API参考](#api参考)
- [配置说明](#配置说明)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

---

## 概述

数据采集模块是AI交易系统的基础组件，负责从各种数据源获取、处理、存储和管理金融市场数据。

### 主要功能

- ✅ **数据下载**: 支持Yahoo Finance等多个数据源
- ✅ **智能缓存**: Parquet格式存储，支持压缩和版本控制
- ✅ **增量更新**: 自动检测并下载新数据
- ✅ **数据验证**: 完整性检查和异常值检测
- ✅ **数据清洗**: 时区处理、交易时段过滤
- ✅ **多品种管理**: 统一管理多个交易品种
- ✅ **版本控制**: 完整的数据版本历史

### 支持的数据源

- Yahoo Finance (yfinance)
- 未来可扩展: Alpha Vantage, IEX Cloud, Polygon.io等

### 支持的品种

- 期货: 能源(原油、天然气)、金属(黄金、白银、铜)、农产品(玉米、大豆、小麦)、指数(S&P 500, Nasdaq, Dow Jones)
- ETF: SPY, QQQ, DIA, GLD, USO等
- 股票: 任意美股代码

---

## 模块架构

```
src/data/
├── downloader.py      # 数据下载器
├── cache.py           # 数据缓存管理
├── updater.py         # 增量更新器
├── validator.py       # 数据验证器
├── cleaner.py         # 数据清洗器
└── manager.py         # 多品种数据管理器
```

### 数据流程

```
下载 → 验证 → 清洗 → 缓存 → 版本控制
  ↓      ↓      ↓      ↓        ↓
Yahoo  完整性  时区   Parquet  历史记录
Finance 检查   转换   存储     追踪
```

---

## 核心组件

### 1. YahooFinanceDownloader

数据下载器，负责从Yahoo Finance获取OHLCV数据。

**特性:**
- 支持多种时间频率(1m, 5m, 15m, 30m, 1h, 1d)
- 自动重试机制(最多3次)
- 速率限制(60次/分钟, 2000次/小时)
- 错误处理和日志记录

**示例:**
```python
from src.data.downloader import YahooFinanceDownloader

downloader = YahooFinanceDownloader()

# 下载单个品种
data = downloader.download(
    symbol="AAPL",
    start="2024-01-01",
    end="2024-12-31",
    interval="5m"
)

# 批量下载
symbols = ["AAPL", "MSFT", "GOOGL"]
results = downloader.download_multiple(
    symbols=symbols,
    start="2024-01-01",
    end="2024-12-31",
    interval="1d",
    delay=1.0
)
```

### 2. DataCache

数据缓存管理器，提供高效的本地存储。

**特性:**
- Parquet格式存储(支持Snappy压缩)
- MD5校验和验证
- 缓存过期检查
- 元数据管理

**示例:**
```python
from src.data.cache import DataCache

cache = DataCache()

# 保存数据
cache.save(data, symbol="AAPL", interval="5m")

# 加载数据
data = cache.load(symbol="AAPL", interval="5m")

# 检查缓存
exists = cache.exists(symbol="AAPL", interval="5m")

# 获取元数据
metadata = cache.get_metadata(symbol="AAPL", interval="5m")

# 获取缓存统计
stats = cache.get_cache_size()
print(f"缓存大小: {stats['total_size_mb']} MB")
```

### 3. DataUpdater

增量数据更新器，智能管理数据更新。

**特性:**
- 自动检测本地最新数据时间戳
- 仅下载新增数据
- 自动合并到现有数据集
- 避免重复数据

**示例:**
```python
from src.data.updater import DataUpdater

updater = DataUpdater()

# 增量更新(自动检测需要更新的部分)
data = updater.update(
    symbol="AAPL",
    interval="5m",
    force=False  # False表示增量更新
)

# 强制全量更新
data = updater.update(
    symbol="AAPL",
    interval="5m",
    start="2024-01-01",
    force=True
)

# 批量更新
symbols = ["AAPL", "MSFT"]
results = updater.update_multiple(
    symbols=symbols,
    interval="5m",
    delay=1.0
)

# 检查更新状态
status = updater.get_update_status("AAPL", "5m")
print(f"需要更新: {status['needs_update']}")
```

### 4. DataValidator

数据验证器，确保数据质量。

**特性:**
- 缺失值检查(阈值<1%)
- 时间连续性检查
- OHLC一致性验证
- 价格异常值检测(5σ阈值)
- 成交量异常值检测(10倍MA阈值)
- 自动修复功能

**示例:**
```python
from src.data.validator import DataValidator

validator = DataValidator()

# 验证数据
validated_data, report = validator.validate(
    data=data,
    symbol="AAPL",
    fix_issues=True  # 自动修复问题
)

# 查看报告
print(f"质量分数: {report['quality_score']:.2f}")
print(f"发现问题: {report['issues_found']}")
print(f"已修复: {report['issues_fixed']}")

# 单独检查
missing_check = validator.check_missing_values(data)
ohlc_check = validator.check_ohlc_consistency(data)
outliers = validator.detect_price_outliers(data)
```

### 5. DataCleaner

数据清洗器，处理时区和交易时段。

**特性:**
- 时区转换(市场时区 → UTC)
- 交易时段过滤
- 数据标准化
- 完整的清洗管道

**示例:**
```python
from src.data.cleaner import DataCleaner, DataCleaningPipeline

cleaner = DataCleaner()

# 时区转换
converted_data = cleaner.convert_timezone(
    data,
    from_tz='America/New_York',
    to_tz='UTC'
)

# 交易时段过滤
filtered_data = cleaner.filter_trading_hours(
    data,
    start_time='09:30',
    end_time='16:00',
    include_extended=False
)

# 完整清洗流程
cleaned_data, report = cleaner.clean(
    data,
    symbol="AAPL",
    convert_timezone=True,
    filter_trading_hours=True,
    validate=True,
    fix_issues=True
)

# 使用清洗管道
pipeline = DataCleaningPipeline()
result = pipeline.transform(data, symbol="AAPL")
```

### 6. MultiSymbolDataManager

多品种数据管理器，统一管理所有数据。

**特性:**
- 多品种统一管理
- 批量操作支持
- 版本控制集成
- 数据索引和查询

**示例:**
```python
from src.data.manager import MultiSymbolDataManager

manager = MultiSymbolDataManager()

# 添加品种
manager.add_symbol(
    symbol="AAPL",
    interval="5m",
    start="2024-01-01",
    force_download=True
)

# 获取品种数据
data = manager.get_symbol_data(
    symbol="AAPL",
    interval="5m",
    start="2024-01-01",
    end="2024-12-31"
)

# 批量获取
symbols = ["AAPL", "MSFT", "GOOGL"]
results = manager.get_multiple_symbols(
    symbols=symbols,
    interval="5m"
)

# 更新所有品种
update_results = manager.update_all_symbols(
    interval="5m",
    delay=1.0
)

# 获取品种信息
info = manager.get_symbol_info("AAPL", "5m")
print(f"缓存: {info['cached']}")
print(f"版本数: {len(info['versions'])}")

# 获取统计信息
stats = manager.get_statistics()
print(f"总品种数: {stats['total_symbols']}")
print(f"总记录数: {stats['total_records']}")
print(f"缓存大小: {stats['cache_size']['total_size_mb']} MB")

# 列出所有品种
symbols = manager.list_symbols(interval="5m")
print(f"已缓存品种: {symbols}")
```

---

## 使用指南

### 快速开始

#### 1. 下载单个品种数据

```python
from src.data.manager import MultiSymbolDataManager

# 创建管理器
manager = MultiSymbolDataManager()

# 添加品种(自动下载、清洗、缓存)
manager.add_symbol(
    symbol="AAPL",
    interval="5m",
    start="2024-01-01",
    end="2024-12-31"
)

# 获取数据
data = manager.get_symbol_data("AAPL", "5m")
print(f"数据: {len(data)} 条记录")
print(data.head())
```

#### 2. 批量下载多个品种

```python
# 定义品种列表
symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

# 批量添加
for symbol in symbols:
    manager.add_symbol(
        symbol=symbol,
        interval="1d",
        start="2020-01-01"
    )

# 批量获取
results = manager.get_multiple_symbols(symbols, interval="1d")
for symbol, data in results.items():
    print(f"{symbol}: {len(data)} 条记录")
```

#### 3. 增量更新数据

```python
from src.data.updater import DataUpdater

updater = DataUpdater()

# 检查更新状态
status = updater.get_update_status("AAPL", "5m")
if status['needs_update']:
    print("需要更新")
    
    # 执行增量更新
    data = updater.update("AAPL", "5m")
    print(f"更新后: {len(data)} 条记录")
```

#### 4. 数据验证和清洗

```python
from src.data.validator import DataValidator
from src.data.cleaner import DataCleaningPipeline

# 验证数据
validator = DataValidator()
validated_data, report = validator.validate(data, symbol="AAPL")
print(f"质量分数: {report['quality_score']:.2f}")

# 清洗数据
pipeline = DataCleaningPipeline()
cleaned_data = pipeline.transform(data, symbol="AAPL")
```

### 高级用法

#### 1. 自定义配置

```python
from src.utils.config_loader import ConfigLoader

# 加载配置
config = ConfigLoader("configs/data_config.yaml")

# 修改配置
config.set('cache.format', 'parquet')
config.set('quality.missing_values.max_missing_ratio', 0.02)

# 保存配置
config.save("configs/my_data_config.yaml")

# 使用自定义配置
from src.data.manager import MultiSymbolDataManager
manager = MultiSymbolDataManager("configs/my_data_config.yaml")
```

#### 2. 版本控制

```python
from src.data.manager import DataVersionControl

version_control = DataVersionControl()

# 创建版本
version_id = version_control.create_version(
    symbol="AAPL",
    interval="5m",
    data=data,
    metadata={'note': '初始版本'}
)

# 获取版本列表
versions = version_control.get_versions("AAPL", "5m")
for v in versions:
    print(f"版本: {v['version_id']}, 记录数: {v['records']}")

# 获取最新版本
latest = version_control.get_latest_version("AAPL", "5m")
print(f"最新版本: {latest['version_id']}")
```

#### 3. 数据质量监控

```python
from src.data.validator import DataValidator

validator = DataValidator()

# 批量验证
symbols = ["AAPL", "MSFT", "GOOGL"]
quality_scores = {}

for symbol in symbols:
    data = manager.get_symbol_data(symbol, "5m")
    _, report = validator.validate(data, symbol=symbol)
    quality_scores[symbol] = report['quality_score']

# 输出质量报告
for symbol, score in quality_scores.items():
    print(f"{symbol}: {score:.2f}")
```

---

## API参考

### YahooFinanceDownloader

#### `download(symbol, start, end, interval, **kwargs)`
下载指定品种的OHLCV数据。

**参数:**
- `symbol` (str): 品种代码
- `start` (str|datetime): 开始日期
- `end` (str|datetime): 结束日期
- `interval` (str): 数据频率

**返回:** `pd.DataFrame`

#### `download_multiple(symbols, start, end, interval, delay, **kwargs)`
批量下载多个品种。

**参数:**
- `symbols` (List[str]): 品种代码列表
- `delay` (float): 品种间延迟(秒)

**返回:** `Dict[str, pd.DataFrame]`

### DataCache

#### `save(data, symbol, interval, metadata)`
保存数据到缓存。

**参数:**
- `data` (pd.DataFrame): 数据
- `symbol` (str): 品种代码
- `interval` (str): 数据频率
- `metadata` (Dict): 额外元数据

**返回:** `bool`

#### `load(symbol, interval, check_expiry)`
从缓存加载数据。

**参数:**
- `symbol` (str): 品种代码
- `interval` (str): 数据频率
- `check_expiry` (bool): 是否检查过期

**返回:** `pd.DataFrame | None`

### DataValidator

#### `validate(data, symbol, fix_issues)`
执行完整的数据验证。

**参数:**
- `data` (pd.DataFrame): 数据
- `symbol` (str): 品种代码
- `fix_issues` (bool): 是否自动修复

**返回:** `Tuple[pd.DataFrame, Dict]`

### MultiSymbolDataManager

#### `add_symbol(symbol, interval, start, end, force_download)`
添加品种数据。

**参数:**
- `symbol` (str): 品种代码
- `interval` (str): 数据频率
- `start` (str|datetime): 开始日期
- `end` (str|datetime): 结束日期
- `force_download` (bool): 是否强制下载

**返回:** `bool`

#### `get_symbol_data(symbol, interval, start, end)`
获取品种数据。

**返回:** `pd.DataFrame | None`

---

## 配置说明

### 数据源配置

```yaml
data_sources:
  yahoo_finance:
    enabled: true
    api:
      timeout: 30
      max_retries: 3
      retry_delay: 5
    rate_limit:
      requests_per_minute: 60
      requests_per_hour: 2000
```

### 缓存配置

```yaml
cache:
  directory: "data/cache"
  format: "parquet"  # parquet, csv, pickle
  strategy:
    enabled: true
    compression: "snappy"
  expiry:
    intraday: 3600  # 1小时
    daily: 86400    # 24小时
```

### 数据质量配置

```yaml
quality:
  missing_values:
    max_missing_ratio: 0.01  # 1%
    max_consecutive_missing: 5
  outliers:
    price_jump:
      threshold_sigma: 5
    volume_spike:
      threshold_multiplier: 10
      ma_window: 20
```

---

## 最佳实践

### 1. 数据下载

- ✅ 使用增量更新而非全量下载
- ✅ 设置合理的延迟避免速率限制
- ✅ 使用try-except处理下载错误
- ✅ 定期检查数据质量

### 2. 缓存管理

- ✅ 定期清理过期缓存
- ✅ 监控缓存大小
- ✅ 使用Parquet格式节省空间
- ✅ 启用压缩

### 3. 数据验证

- ✅ 每次下载后验证数据
- ✅ 自动修复常见问题
- ✅ 记录质量分数
- ✅ 设置质量阈值

### 4. 版本控制

- ✅ 为重要更新创建版本
- ✅ 保留足够的历史版本
- ✅ 记录版本元数据
- ✅ 定期清理旧版本

---

## 常见问题

### Q1: 如何处理下载失败?

**A**: 系统自动重试3次，如果仍然失败会抛出异常。可以捕获异常并记录日志：

```python
try:
    data = downloader.download("AAPL", "2024-01-01", "2024-12-31")
except RuntimeError as e:
    logger.error(f"下载失败: {str(e)}")
```

### Q2: 如何清理缓存?

**A**: 使用DataCache的清理方法：

```python
cache = DataCache()

# 删除单个缓存
cache.delete("AAPL", "5m")

# 清空所有缓存
cache.clear_all()
```

### Q3: 如何处理时区问题?

**A**: 使用DataCleaner自动转换时区：

```python
cleaner = DataCleaner()
converted_data = cleaner.convert_timezone(
    data,
    from_tz='America/New_York',
    to_tz='UTC'
)
```

### Q4: 如何提高下载速度?

**A**: 
1. 使用缓存避免重复下载
2. 使用增量更新
3. 批量下载时设置合理延迟
4. 考虑使用多线程(需要注意速率限制)

### Q5: 数据质量分数如何计算?

**A**: 质量分数基于多个因素：
- 缺失值比例(每1%扣10分)
- 时间不连续(每个间隔扣2分)
- OHLC不一致(按比例扣分)
- 价格异常值(按比例扣分)
- 成交量异常值(按比例扣分)

满分100分，分数越高质量越好。

---

## 更新日志

### v1.0.0 (2025-01-20)
- ✅ 初始版本发布
- ✅ 支持Yahoo Finance数据源
- ✅ 实现完整的数据采集流程
- ✅ 支持多品种管理
- ✅ 实现版本控制

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [项目地址]
- Email: [邮箱地址]

---

**文档版本**: v1.0.0  
**最后更新**: 2025-01-20