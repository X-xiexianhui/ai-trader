# AI交易系统 - 基于TS2Vec-Transformer-PPO

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个融合"形态识别 → 状态建模 → 动作决策"的三层智能交易系统，用于5分钟级别期货交易。

## 🎯 项目目标

构建一个以**快速训练验证模型效果**为核心目标的智能交易系统，按照**最简单的路线**实现核心功能。

## 🏗️ 系统架构

```
原始OHLC数据
    ↓
[数据层] 数据清洗 + 27维手工特征
    ↓
[TS2Vec层] 形态识别 → 128维embedding
    ↓
[Transformer层] 状态建模 → 256维状态向量
    ↓
[PPO层] 动作决策 → 交易信号
    ↓
[回测层] 性能评估
```

## ✨ 核心特性

### 1. 三层架构设计
- **TS2Vec形态编码器**: 无监督学习提取时间序列形态特征
- **Transformer状态建模器**: 融合embedding和手工特征，建模市场状态
- **PPO强化学习**: 基于状态向量进行交易决策

### 2. 完整的数据处理流程
- 数据清洗（缺失值、异常值、时间对齐）
- 27维手工特征（价格、波动率、技术指标、成交量、K线形态、时间周期）
- 特征归一化（StandardScaler + RobustScaler）
- 特征验证（信息量测试、置换重要性、相关性检测、VIF）

### 3. 端到端训练和评估
- 统一的训练脚本（支持TS2Vec、Transformer、PPO）
- 多种评估模式（回测、Walk-forward、过拟合检测、市场状态分析）
- 完善的日志系统（多级别、彩色输出、TensorBoard集成）

### 4. 实时推理能力
- 低延迟数据管道（<50ms）
- 高效的缓冲区管理
- 支持单条和批量处理

## 📦 安装

### 环境要求
- Python 3.11+
- CUDA 12.6+ (可选，用于GPU加速)
- 8GB+ RAM

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/yourusername/ai-trader.git
cd ai-trader

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 🔄 项目同步

### 同步到远程服务器

项目提供了自动同步脚本，可以快速将本地代码变更同步到远程服务器：

```bash
# 运行同步脚本
./sync_to_server.sh
```

**特性**：
- ✅ 交互式确认，避免误操作
- ✅ 实时显示传输进度
- ✅ 自动排除不必要的文件（缓存、日志、数据文件等）
- ✅ 彩色输出，清晰易读
- ✅ 错误检测和提示

**详细使用说明**：请参阅 [`SYNC_GUIDE.md`](SYNC_GUIDE.md:1)

**服务器信息**：
- 服务器：xj-member.bitahub.com
- 端口：42052
- 远程路径：~/ai-trader/

```

## 🚀 快速开始

### 1. 配置文件

编辑 `configs/config.yaml` 设置参数：

```yaml
data:
  symbols: ['ES=F']  # 标普500期货
  interval: '5m'
  start_date: '2020-01-01'
  end_date: '2024-12-31'

ts2vec:
  window_length: 256
  hidden_dim: 64
  output_dim: 128

transformer:
  sequence_length: 64
  d_model: 256
  nhead: 8
  num_layers: 6

ppo:
  n_steps: 2048
  learning_rate: 0.0003
  gamma: 0.99
```

### 2. 训练模型

```bash
# 训练所有模型
python scripts/train.py --config configs/config.yaml --model all

# 或分步训练
python scripts/train.py --model ts2vec      # 1. 训练TS2Vec
python scripts/train.py --model transformer # 2. 训练Transformer
python scripts/train.py --model ppo         # 3. 训练PPO
```

### 3. 评估模型

```bash
# 评估所有模型
python scripts/evaluate.py --model all

# PPO多种评估模式
python scripts/evaluate.py --model ppo --mode backtest      # 回测
python scripts/evaluate.py --model ppo --mode walk_forward  # Walk-forward验证
python scripts/evaluate.py --model ppo --mode overfitting   # 过拟合检测
python scripts/evaluate.py --model ppo --mode market_state  # 市场状态分析
```

### 4. 查看结果

```bash
# 查看日志
tail -f logs/train.log
tail -f logs/evaluate.log

# 启动TensorBoard
tensorboard --logdir logs/
```

## 📊 示例代码

### 数据处理示例

```python
from src.pipeline.training_pipeline import TrainingDataPipeline

# 创建数据管道
pipeline = TrainingDataPipeline(
    ts2vec_model_path='models/checkpoints/ts2vec/ts2vec_final.pth',
    scaler_path='models/scalers/feature_scaler.pkl',
    config=config
)

# 处理数据
train_data, val_data, test_data = pipeline.process(df)
```

### 实时推理示例

```python
from src.pipeline.inference_pipeline import InferenceDataPipeline

# 创建推理管道
pipeline = InferenceDataPipeline(
    ts2vec_model_path='models/checkpoints/ts2vec/ts2vec_final.pth',
    scaler_path='models/scalers/feature_scaler.pkl',
    config=config
)

# 预热
pipeline.warmup(historical_data)

# 处理新数据
model_input = pipeline.process_new_bar(new_bar)
```

### 回测示例

```python
from src.backtest.engine import BacktestEngine

# 创建回测引擎
engine = BacktestEngine(
    agent=ppo_agent,
    transformer_model=transformer_model,
    config=config
)

# 运行回测
results = engine.run(test_data)
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

## 📁 项目结构

```
ai-trader/
├── configs/                    # 配置文件
│   └── config.yaml
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后数据
├── docs/                       # 文档
│   ├── MODULE_3_COMPLETION_SUMMARY.md
│   ├── MODULE_7_COMPLETION_SUMMARY.md
│   └── ...
├── examples/                   # 示例代码
│   ├── data_pipeline_demo.py
│   ├── ts2vec_training_demo.py
│   ├── transformer_training_demo.py
│   └── ppo_training_demo.py
├── logs/                       # 日志目录
├── models/                     # 模型保存目录
│   ├── checkpoints/
│   └── scalers/
├── scripts/                    # 脚本
│   ├── train.py               # 训练脚本
│   └── evaluate.py            # 评估脚本
├── src/                        # 源代码
│   ├── data/                  # 数据模块
│   │   ├── downloader.py
│   │   └── storage.py
│   ├── features/              # 特征模块
│   │   ├── data_cleaner.py
│   │   ├── feature_calculator.py
│   │   └── feature_scaler.py
│   ├── models/                # 模型模块
│   │   ├── ts2vec/
│   │   ├── transformer/
│   │   └── ppo/
│   ├── pipeline/              # 数据管道
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   ├── backtest/              # 回测模块
│   │   ├── engine.py
│   │   ├── strategy.py
│   │   └── recorder.py
│   ├── evaluation/            # 评估模块
│   │   ├── walk_forward.py
│   │   ├── overfitting_detection.py
│   │   └── market_state.py
│   └── utils/                 # 工具模块
│       └── logger.py
├── tests/                      # 测试
├── requirements.txt            # 依赖
├── task.md                     # 任务文档
├── design_document.md          # 设计文档
└── README.md                   # 本文件
```

## 🔬 技术栈

### 深度学习
- **PyTorch 2.0+**: 深度学习框架
- **TS2Vec**: 时间序列对比学习
- **Transformer**: 序列建模
- **PPO**: 强化学习算法

### 数据处理
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **ta**: 技术指标计算

### 回测与评估
- **Backtrader**: 回测框架
- **scikit-learn**: 机器学习工具

### 工程化
- **TensorBoard**: 训练可视化
- **colorlog**: 彩色日志
- **PyYAML**: 配置管理

## 📈 性能指标

### 模型性能
- **TS2Vec**: 
  - 对比损失 < 0.5
  - 正样本相似度 > 0.8
  - 线性探测准确率 > 55%

- **Transformer**:
  - 回归MSE < 0.01
  - 分类准确率 > 55%
  - 状态向量方差 > 0.1

- **PPO**:
  - 夏普比率 > 1.5
  - 最大回撤 < 20%
  - 年化收益率 > 15%
  - 胜率 > 50%

### 系统性能
- 训练数据管道: ~1000条/秒
- 推理延迟: <50ms
- 内存占用: <2GB（训练）, <500MB（推理）

## 📚 文档

- [任务拆分文档](task.md) - 详细的任务列表和依赖关系
- [设计文档](design_document.md) - 系统设计和技术细节
- [模块3完成总结](docs/MODULE_3_COMPLETION_SUMMARY.md) - Transformer模块
- [模块7完成总结](docs/MODULE_7_COMPLETION_SUMMARY.md) - 工程化模块

## 🧪 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_data_cleaning.py
pytest tests/test_features.py
pytest tests/test_module5.py
pytest tests/test_module6.py

# 查看覆盖率
pytest --cov=src tests/
```

## 🛠️ 开发指南

### 代码风格
- 使用Black格式化代码
- 使用flake8检查代码质量
- 使用mypy进行类型检查

```bash
black src/
flake8 src/
mypy src/
```

### 添加新特征
1. 在 `src/features/feature_calculator.py` 中添加计算方法
2. 在 `configs/config.yaml` 中配置参数
3. 添加单元测试
4. 更新文档

### 添加新模型
1. 在 `src/models/` 下创建新目录
2. 实现模型、训练器、评估器
3. 在训练和评估脚本中集成
4. 添加示例代码

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [TS2Vec](https://github.com/yuezhihan/ts2vec) - 时间序列对比学习
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - 强化学习库
- [Backtrader](https://github.com/mementum/backtrader) - 回测框架

## 📧 联系方式

- 项目主页: [https://github.com/yourusername/ai-trader](https://github.com/yourusername/ai-trader)
- 问题反馈: [Issues](https://github.com/yourusername/ai-trader/issues)

## 🗺️ 路线图

- [x] 模块1: 数据层
- [x] 模块2: TS2Vec形态编码器
- [x] 模块3: Transformer状态建模器
- [x] 模块4: PPO强化学习
- [x] 模块5: 测试层
- [x] 模块6: 评估层
- [x] 模块7: 工程化
- [ ] 模块8: 部署（可选）
- [ ] 模块9: 监控告警（可选）
- [ ] 模块10: 超参数优化（可选）

---

**注意**: 本项目仅用于学习和研究目的，不构成任何投资建议。实盘交易需谨慎，风险自负。