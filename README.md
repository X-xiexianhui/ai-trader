# AI Trader - 智能交易系统

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

基于TS2Vec-Transformer-PPO的智能量化交易系统

## 📋 项目简介

AI Trader是一个先进的量化交易系统，结合了深度学习和强化学习技术：

- **TS2Vec**: 时序数据的自监督表征学习
- **Transformer**: 强大的序列建模能力
- **PPO**: 稳定的强化学习策略优化

该系统能够自动学习市场模式，生成交易信号，并通过强化学习优化交易策略。

## ✨ 主要特性

### 核心功能
- 🔄 **自动化数据采集**: 支持多数据源，自动更新和缓存
- 🧹 **智能数据清洗**: 异常值检测、缺失值处理、数据标准化
- 🎯 **高级特征工程**: 27维手工特征 + TS2Vec学习特征
- 🤖 **深度学习模型**: TS2Vec + Transformer + PPO三层架构
- 📊 **完整回测系统**: 基于Backtrader，支持GPU加速
- 📈 **全面评估体系**: Walk-Forward验证、多指标评估

### 技术亮点
- ⚡ **GPU加速**: 支持CUDA和ROCm，显著提升训练和回测速度
- 🔧 **模块化设计**: 清晰的代码结构，易于扩展和维护
- 📝 **完整日志系统**: 多级别日志，支持轮转和彩色输出
- ⚙️ **灵活配置管理**: YAML配置文件，支持环境变量覆盖
- 🧪 **实验跟踪**: 集成MLflow和TensorBoard

## 🚀 快速开始

### 环境要求

- **Python 3.11.14** (必需)
- **PyTorch 2.9.0** with **CUDA 12.6** 或 **ROCm 6.0**
- 8GB+ RAM (推荐16GB+)
- **GPU** (推荐用于训练和回测加速):
  - NVIDIA GPU with CUDA 12.6
  - AMD GPU with ROCm 6.0

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader
```

2. **创建虚拟环境**
```bash
# 确保使用Python 3.11.14
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装PyTorch**

**NVIDIA GPU (CUDA 12.6):**
```bash
pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

**AMD GPU (ROCm 6.0):**
```bash
pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

**CPU版本 (无GPU):**
```bash
pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

4. **安装其他依赖**
```bash
pip install -r requirements.txt
```
> 📖 **详细安装指南**: 如需完整的安装说明（包括Python 3.11.14安装、CUDA配置、常见问题等），请参考 [INSTALL.md](INSTALL.md)


5. **配置系统**
```bash
# 复制配置文件模板
cp configs/base_config.yaml configs/my_config.yaml

# 根据需要修改配置
vim configs/my_config.yaml
```

### 基本使用

#### 方式1: 一键训练（推荐）

```bash
# 运行完整训练流程（数据采集 → 特征提取 → 模型训练）
python train.py

# 使用自定义配置
python train.py --config configs/my_config.yaml

# 只训练特定模型
python train.py --model ts2vec      # 只训练TS2Vec
python train.py --model transformer # 只训练Transformer
python train.py --model ppo         # 只训练PPO
```

训练完成后，模型将保存在：
- `models/ts2vec/best_model.pt`
- `models/transformer/best_model.pt`
- `models/ppo/best_model.pt`

#### 方式2: 运行已训练模型

```bash
# 单次预测
python run.py --mode once --symbol ES=F

# 持续运行（每5分钟预测一次）
python run.py --mode continuous --symbol ES=F --interval 300

# 回测模式
python run.py --mode backtest --symbol ES=F --start 2023-01-01 --end 2023-12-31

# 使用CPU运行
python run.py --device cpu --mode once
```

**输出示例**:
```
============================================================
TRADING SIGNAL
============================================================
Symbol:        ES=F
Current Price: $4521.50
Direction:     LONG
Position Size: 45.00%
Stop Loss:     2.50%
Take Profit:   5.00%
Confidence:    78.50%
Latency:       15.23ms
Timestamp:     2023-11-20T21:30:00
============================================================
```

#### 方式3: 编程方式使用

##### 1. 数据采集
```python
from src.data.downloader import YahooFinanceDownloader
from src.data.cleaner import DataCleaningPipeline

# 下载数据
downloader = YahooFinanceDownloader()
data = downloader.download(symbol="ES=F", start="2020-01-01", end="2024-12-31")

# 清洗数据
cleaner = DataCleaningPipeline()
clean_data = cleaner.transform(data)
```

##### 2. 特征工程
```python
from src.features.pipeline import FeatureEngineeringPipeline

# 计算特征
feature_pipeline = FeatureEngineeringPipeline()
features = feature_pipeline.transform(clean_data)
```

##### 3. 模型推理
```python
from src.api.inference_service import LocalInferenceService

# 初始化推理服务
service = LocalInferenceService(model_dir="models", device="auto")

# 执行推理
signal = service.predict(market_data, features)

print(f"Direction: {signal['direction']}")
print(f"Position Size: {signal['position_size']:.2%}")
print(f"Confidence: {signal['confidence']:.2%}")
```

##### 4. 回测评估
```python
from src.backtest.engine import BacktestEngine

# 运行回测
engine = BacktestEngine(config)
results = engine.run(strategy, data)

# 生成报告
engine.generate_report(results)
```

## 📁 项目结构

```
ai-trader/
├── configs/                 # 配置文件
│   ├── base_config.yaml    # 基础配置
│   ├── experiment_config.yaml  # 实验配置
│   └── ...
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── cache/              # 缓存数据
├── models/                  # 模型保存目录
│   ├── ts2vec/             # TS2Vec模型
│   ├── transformer/        # Transformer模型
│   └── ppo/                # PPO模型
├── src/                     # 源代码
│   ├── data/               # 数据处理模块
│   │   ├── downloader.py   # 数据下载
│   │   └── cleaner.py      # 数据清洗
│   ├── features/           # 特征工程模块
│   │   ├── price_features.py
│   │   ├── technical_features.py
│   │   └── pipeline.py
│   ├── models/             # 模型模块
│   │   ├── ts2vec/         # TS2Vec实现
│   │   ├── transformer/    # Transformer实现
│   │   └── ppo/            # PPO实现
│   ├── backtest/           # 回测模块
│   │   ├── engine.py       # 回测引擎
│   │   └── metrics.py      # 性能指标
│   ├── evaluation/         # 评估模块
│   │   └── walk_forward.py # Walk-Forward验证
│   ├── utils/              # 工具模块
│   │   ├── config_loader.py  # 配置加载
│   │   ├── logger.py       # 日志系统
│   │   └── helpers.py      # 辅助函数
│   └── api/                # API接口
├── logs/                    # 日志文件
├── notebooks/              # Jupyter笔记本
├── tests/                  # 测试文件
├── scalers/                # 归一化器保存目录
├── requirements.txt        # 项目依赖
├── task.md                 # 任务文档
├── design_document.md      # 设计文档
└── README.md               # 项目说明
```

## 🔧 配置说明

### 基础配置 (configs/base_config.yaml)

主要配置项：

```yaml
# 环境配置
environment:
  mode: "development"  # development, production, testing
  seed: 42
  device:
    type: "auto"  # auto, cuda, rocm, cpu

# 数据配置
data:
  frequency: 5  # 数据频率（分钟）
  split:
    train: 0.7
    validation: 0.15
    test: 0.15

# 训练配置
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
```

### 环境变量覆盖

可以通过环境变量覆盖配置：

```bash
# 设置批次大小
export AI_TRADER_TRAINING__BATCH_SIZE=64

# 设置设备类型
export AI_TRADER_ENVIRONMENT__DEVICE__TYPE=cuda
```

## 📊 性能指标

系统评估指标包括：

- **Sharpe Ratio**: 风险调整后收益
- **CAGR**: 复合年增长率
- **Max Drawdown**: 最大回撤
- **Win Rate**: 胜率
- **Profit Factor**: 盈亏比

## 🧪 测试

运行测试：

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_experiment_tracking.py

# 生成覆盖率报告
pytest --cov=src tests/
```

## 📚 文档

详细文档请参考：

- [设计文档](design_document.md) - 系统架构和设计
- [任务文档](task.md) - 开发任务和进度
- [需求文档](requirements.md) - 项目需求说明

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👥 作者

AI Trader Team

## 🙏 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习库
- [Backtrader](https://www.backtrader.com/) - 回测框架
- [MLflow](https://mlflow.org/) - 实验跟踪
- [yfinance](https://github.com/ranaroussi/yfinance) - 金融数据获取

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](https://github.com/yourusername/ai-trader/issues)
- 发送邮件至: your.email@example.com

---

⭐ 如果这个项目对你有帮助，请给它一个星标！