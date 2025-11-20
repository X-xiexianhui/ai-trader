# AI交易系统 - 基于TS2Vec-Transformer-PPO

一个融合"形态识别 → 状态建模 → 动作决策"的三层智能交易系统，用于5分钟级别期货交易。

## 📋 项目概述

本项目实现了一个完整的深度强化学习交易系统，包含三个核心模块：

1. **TS2Vec形态编码器** - 使用对比学习提取时间序列的深层特征
2. **Transformer状态建模器** - 建模市场状态的时序依赖关系
3. **PPO强化学习** - 学习最优交易策略

## 🏗️ 项目结构

```
ai-trader-demo/
├── src/                      # 源代码
│   ├── data/                 # 数据层
│   │   ├── cleaning.py       # 数据清洗（任务1.1.1-1.1.4）
│   │   ├── features.py       # 特征计算（任务1.2.1-1.2.7）
│   │   └── normalization.py  # 特征归一化（任务1.3.1-1.3.3）
│   ├── models/               # 模型层
│   │   ├── ts2vec/          # TS2Vec模型
│   │   ├── transformer/     # Transformer模型
│   │   └── ppo/             # PPO模型
│   ├── features/            # 特征工程
│   ├── environment/         # 交易环境
│   ├── evaluation/          # 评估模块
│   ├── training/            # 训练脚本
│   └── utils/               # 工具函数
├── configs/                 # 配置文件
│   └── config.yaml          # 主配置文件
├── data/                    # 数据目录
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── models/                  # 模型保存
│   ├── checkpoints/         # 训练检查点
│   └── scalers/             # 归一化器
├── logs/                    # 日志文件
├── tests/                   # 单元测试
├── examples/                # 示例代码
│   └── data_pipeline_demo.py
├── requirements.txt         # 依赖包
└── README.md               # 项目文档
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd ai-trader-demo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行数据处理示例

```bash
cd examples
python data_pipeline_demo.py
```

这个示例演示了完整的数据处理流程：
- 数据清洗（缺失值处理、异常值检测）
- 特征计算（27维手工特征）
- 特征归一化（StandardScaler和RobustScaler）

## 📊 数据层功能

### 数据清洗模块 (DataCleaner)

实现了以下功能：

1. **缺失值处理** (任务1.1.1)
   - 前向填充用于价格数据
   - 零填充用于成交量数据
   - 线性插值用于短期缺失
   - 删除连续缺失超过5根K线的数据段

2. **异常值检测与处理** (任务1.1.2)
   - 使用3σ原则检测价格跳变
   - 区分真实跳空和数据错误
   - 修正OHLC一致性约束

3. **时间对齐与时区处理** (任务1.1.3)
   - 转换时区到UTC或指定时区
   - 处理夏令时切换
   - 过滤非交易时段数据
   - 重采样确保严格5分钟间隔

4. **数据质量验证** (任务1.1.4)
   - 完整性检查：缺失值比例<1%
   - 一致性检查：High>=max(O,C), Low<=min(O,C)
   - 异常值检查：价格跳变<5σ
   - 时间检查：间隔严格、无重复

### 特征计算模块 (FeatureCalculator)

计算27维手工特征，分为6组：

1. **价格与收益特征** (5维) - 任务1.2.1
   - ret_1, ret_5, ret_20: 对数收益率
   - price_slope_20: 价格线性回归斜率
   - C_div_MA20: 收盘价/20周期均线

2. **波动率特征** (5维) - 任务1.2.2
   - ATR14_norm: 归一化ATR
   - vol_20: 20周期标准差
   - range_20_norm: 归一化价格范围
   - BB_width_norm: 归一化布林带宽度
   - parkinson_vol: Parkinson波动率

3. **技术指标特征** (4维) - 任务1.2.3
   - EMA20: 指数移动平均
   - stoch: 随机指标
   - MACD: MACD线
   - VWAP: 成交量加权平均价

4. **成交量特征** (4维) - 任务1.2.4
   - volume: 原始成交量
   - volume_zscore: 成交量Z-score
   - volume_change_1: 成交量变化率
   - OBV_slope_20: OBV斜率

5. **K线形态特征** (7维) - 任务1.2.5
   - pos_in_range_20: 在20周期范围内的相对位置
   - dist_to_HH20_norm, dist_to_LL20_norm: 到高低点的距离
   - body_ratio: 实体比例
   - upper_shadow_ratio, lower_shadow_ratio: 影线比例
   - FVG: 公允价值缺口 (任务1.2.7)

6. **时间周期特征** (2维) - 任务1.2.6
   - sin_tod, cos_tod: 时间的正弦余弦编码

### 特征归一化模块

提供三种归一化器：

1. **StandardScaler** (任务1.3.1)
   - 用于收益率、价格斜率等特征
   - z-score标准化：z = (x - μ) / σ

2. **RobustScaler** (任务1.3.2)
   - 用于波动率、技术指标等对异常值敏感的特征
   - 鲁棒标准化：z = (x - median) / IQR

3. **FeatureScaler** (任务1.3.3)
   - 自动为不同特征组选择合适的scaler
   - 支持保存和加载

## 📈 使用示例

### 数据清洗

```python
from src.data.cleaning import DataCleaner

# 创建清洗器
cleaner = DataCleaner(max_consecutive_missing=5)

# 执行完整清洗流程
df_clean, report = cleaner.clean_pipeline(
    df_raw,
    target_timezone='UTC',
    trading_hours=(9, 16)  # 可选：指定交易时段
)
```

### 特征计算

```python
from src.data.features import FeatureCalculator

# 创建特征计算器
feature_calc = FeatureCalculator()

# 计算所有特征
df_features = feature_calc.calculate_all_features(df_clean)

# 获取特征名称和分组
feature_names = feature_calc.get_feature_names()
feature_groups = feature_calc.get_feature_groups()
```

### 特征归一化

```python
from src.data.normalization import FeatureScaler

# 创建归一化器
scaler = FeatureScaler()

# 拟合并转换训练集
X_train_scaled = scaler.fit_transform(X_train, feature_groups)

# 转换测试集
X_test_scaled = scaler.transform(X_test)

# 保存scaler
scaler.save("models/scalers")

# 加载scaler
scaler = FeatureScaler.load("models/scalers")
```

## 🎯 开发进度

### 里程碑1: 数据基础设施 ✅ (已完成)

- [x] 任务1.1.1: OHLC数据缺失值处理
- [x] 任务1.1.2: 价格异常值检测与处理
- [x] 任务1.1.3: 时间对齐与时区处理
- [x] 任务1.1.4: 数据质量验证器
- [x] 任务1.2.1: 价格与收益特征计算（5维）
- [x] 任务1.2.2: 波动率特征计算（5维）
- [x] 任务1.2.3: 技术指标特征计算（4维）
- [x] 任务1.2.4: 成交量特征计算（4维）
- [x] 任务1.2.5: K线形态特征计算（7维）
- [x] 任务1.2.6: 时间周期特征计算（2维）
- [x] 任务1.2.7: FVG公允价值缺口计算
- [x] 任务1.3.1: StandardScaler归一化器
- [x] 任务1.3.2: RobustScaler归一化器
- [x] 任务1.3.3: 归一化器的保存与加载

### 里程碑2: TS2Vec形态编码器 (进行中)

- [ ] 任务2.1.1-2.1.5: 数据准备模块
- [ ] 任务2.2.1-2.2.4: 模型实现
- [ ] 任务2.3.1-2.3.4: 训练流程
- [ ] 任务2.4.1-2.4.5: 评估指标

### 里程碑3-7: 待开发

详见 [task.md](task.md) 文件

## 🔧 技术栈

- **深度学习**: PyTorch 2.0+
- **数据处理**: Pandas, NumPy, Pandas-TA
- **强化学习**: Stable-Baselines3
- **回测框架**: Backtrader
- **数据源**: yfinance

## 📝 核心设计原则

1. ✅ 以验证模型效果为核心目标
2. ✅ 按照最简单的路线实现
3. ✅ 不拓展其他复杂功能
4. ✅ 使用面向对象风格
5. ✅ 要求高内聚低耦合

## 📖 文档

- [任务拆分文档](task.md) - 详细的87个任务列表
- [设计文档](design_document.md) - 系统架构设计

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 👨‍💻 作者

Kilo Code

---

**注意**: 本项目仅用于学习和研究目的，不构成任何投资建议。