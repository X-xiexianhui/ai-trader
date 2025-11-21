# AI Trader 训练流程说明

本目录包含完整的数据处理和模型训练流程脚本。

## 快速开始 - MES Demo

如果你想快速体验完整流程，可以运行MES（微型标普500期货）的demo：

```bash
python training/demo_mes_pipeline.py
```

这个demo会：
- **智能数据管理**：
  - 首先检查本地是否已有数据（`data/processed/MES_F.parquet`）
  - 如果本地有数据，直接加载使用（几秒钟）
  - 如果本地无数据，下载最近60天的5分钟K线数据
    - **注意**：yfinance的5分钟数据只能获取最近60天
  - 下载后立即清洗并保存到本地
- 执行步骤1-4：数据清洗、归一化、特征验证、核心特征选择
- 生成详细的报告和核心特征JSON文件
- 输出保存在 `training/output/demo/` 目录

**预计运行时间**:
- 首次运行（需下载数据）: 1-2分钟
- 后续运行（从本地加载）: 10-30秒

**输出文件**:
- `data/processed/MES_F.parquet` - 清洗后的历史数据（可重复使用）
- `training/output/demo/mes_core_features.json` - 核心特征列表
- `training/output/demo/mes_demo_report.txt` - 详细验证报告

**注意**:
- yfinance的5分钟数据只能获取最近60天
- 如需重新下载数据，删除 `data/processed/MES_F.parquet` 文件即可
- 如需更长历史数据，请使用日线数据（interval="1d"）

---

## 流程概述

整个训练流程分为5个步骤，每个步骤对应一个独立的Python脚本：

```
01_data_acquisition_and_cleaning.py  → 数据获取和清洗
02_feature_normalization.py          → 特征归一化
03_feature_validation.py             → 手工特征验证
04_core_feature_selection.py         → 核心特征选择
05_model_training_with_core_features.py → 模型训练
```

## 详细说明

### 步骤1: 数据获取和清洗
**脚本**: `01_data_acquisition_and_cleaning.py`

**功能**:
- 从yfinance下载OHLCV数据
- 执行数据清洗（缺失值、异常值、时间对齐）
- 计算27维手工特征
- 保存清洗后的数据

**输出**:
- `data/raw/*.parquet` - 原始数据
- `data/processed/*_with_features.parquet` - 带特征的数据
- `training/output/01_data_acquisition_report.txt` - 数据报告
- `training/output/feature_names.txt` - 特征名称列表

**运行**:
```bash
python training/01_data_acquisition_and_cleaning.py
```

---

### 步骤2: 特征归一化
**脚本**: `02_feature_normalization.py`

**功能**:
- 加载带特征的数据
- 划分训练集和测试集（80/20）
- 对27维手工特征进行归一化
  - 价格和收益特征: StandardScaler
  - 波动率和技术指标: RobustScaler
  - 时间特征: 无需归一化
- 保存归一化后的数据和scaler

**输出**:
- `data/processed/*_train_normalized.parquet` - 归一化训练集
- `data/processed/*_test_normalized.parquet` - 归一化测试集
- `models/scalers/*` - 归一化器
- `training/output/02_normalization_report.txt` - 归一化报告

**运行**:
```bash
python training/02_feature_normalization.py
```

---

### 步骤3: 手工特征验证
**脚本**: `03_feature_validation.py`

**功能**:
- 加载归一化后的训练数据
- 计算目标变量（未来收益）
- 执行特征验证测试：
  - 单特征信息量测试（R²和互信息）
  - 置换重要性测试（p值检验）
  - 特征相关性检测（Pearson相关）
  - VIF多重共线性检测
- **保留所有27个手工特征用于验证**

**输出**:
- `training/output/03_validation_*_report.txt` - 各品种验证报告
- `training/output/03_validation_summary.txt` - 综合验证报告
- `training/output/03_validation_results.pkl` - 验证结果（供下一步使用）
- `feature_correlation_heatmap.png` - 相关性热力图

**运行**:
```bash
python training/03_feature_validation.py
```

---

### 步骤4: 核心特征选择
**脚本**: `04_core_feature_selection.py`

**功能**:
- 加载特征验证结果
- 基于验证结果选择核心特征：
  - 保留显著且重要的特征（p<0.05）
  - 移除高度相关的冗余特征（|ρ|>0.85）
  - 移除高VIF的多重共线性特征（VIF>10）
  - 控制特征数量在10-20个之间
- **将核心特征列表输出到JSON文件**

**输出**:
- `training/output/core_features.json` - **核心特征列表（JSON格式）**
- `training/output/04_feature_selection_report.txt` - 特征选择报告

**运行**:
```bash
python training/04_core_feature_selection.py
```

**核心特征JSON格式**:
```json
{
  "metadata": {
    "created_at": "2024-01-01 12:00:00",
    "description": "基于特征验证结果选择的核心特征",
    "total_symbols": 2
  },
  "core_features": {
    "ES=F": ["ret_1", "ret_5", "ATR14_norm", ...],
    "NQ=F": ["ret_1", "vol_20", "MACD", ...]
  }
}
```

---

### 步骤5: 模型训练
**脚本**: `05_model_training_with_core_features.py`

**功能**:
- **从JSON文件加载核心特征列表**
- 加载归一化后的训练数据
- 提取核心特征进行模型训练
- 训练多种模型（Ridge, Lasso, Random Forest）
- 保存训练好的模型

**重要说明**:
- ✅ **训练模型时只使用核心特征**（从JSON加载）
- ✅ **特征验证时保留所有27个手工特征**

**输出**:
- `models/checkpoints/core_feature_models/*_model.pkl` - 训练好的模型
- `models/checkpoints/core_feature_models/feature_config.json` - 特征配置
- `training/output/05_training_report.txt` - 训练报告

**运行**:
```bash
python training/05_model_training_with_core_features.py
```

---

## 完整流程运行

按顺序执行以下脚本：

```bash
# 步骤1: 数据获取和清洗
python training/01_data_acquisition_and_cleaning.py

# 步骤2: 特征归一化
python training/02_feature_normalization.py

# 步骤3: 手工特征验证
python training/03_feature_validation.py

# 步骤4: 核心特征选择
python training/04_core_feature_selection.py

# 步骤5: 模型训练
python training/05_model_training_with_core_features.py
```

---

## 输出目录结构

```
training/output/
├── 01_data_acquisition_report.txt      # 数据获取报告
├── 02_normalization_report.txt         # 归一化报告
├── 03_validation_ES=F_report.txt       # ES=F验证报告
├── 03_validation_NQ=F_report.txt       # NQ=F验证报告
├── 03_validation_summary.txt           # 综合验证报告
├── 03_validation_results.pkl           # 验证结果（pickle）
├── 04_feature_selection_report.txt     # 特征选择报告
├── core_features.json                  # 核心特征列表（JSON）★
├── 05_training_report.txt              # 训练报告
└── feature_names.txt                   # 所有特征名称

data/
├── raw/                                # 原始数据
│   ├── ES=F.parquet
│   └── NQ=F.parquet
└── processed/                          # 处理后的数据
    ├── ES=F_with_features.parquet
    ├── ES=F_train_normalized.parquet
    ├── ES=F_test_normalized.parquet
    ├── NQ=F_with_features.parquet
    ├── NQ=F_train_normalized.parquet
    └── NQ=F_test_normalized.parquet

models/
├── scalers/                            # 归一化器
│   ├── ES=F/
│   └── NQ=F/
└── checkpoints/
    └── core_feature_models/            # 训练好的模型
        ├── ES=F_ridge_model.pkl
        ├── ES=F_lasso_model.pkl
        ├── ES=F_rf_model.pkl
        ├── NQ=F_ridge_model.pkl
        ├── NQ=F_lasso_model.pkl
        ├── NQ=F_rf_model.pkl
        └── feature_config.json         # 特征配置

logs/
├── 01_data_acquisition.log
├── 02_feature_normalization.log
├── 03_feature_validation.log
├── 04_core_feature_selection.log
└── 05_model_training.log
```

---

## 关键特性

### 1. 核心特征管理
- ✅ 核心特征存储在JSON文件中（`core_features.json`）
- ✅ 训练时从JSON加载核心特征
- ✅ 验证时保留所有27个手工特征

### 2. 特征归一化策略
- **StandardScaler**: 价格收益特征、K线形态特征
- **RobustScaler**: 波动率特征、技术指标、成交量特征
- **无需归一化**: 时间特征（已在[-1,1]范围）

### 3. 特征验证方法
- **单特征信息量**: R²和互信息
- **置换重要性**: 统计显著性检验
- **相关性检测**: Pearson相关系数
- **多重共线性**: VIF（方差膨胀因子）

### 4. 27维手工特征
1. **价格与收益特征（5维）**: ret_1, ret_5, ret_20, price_slope_20, C_div_MA20
2. **波动率特征（5维）**: ATR14_norm, vol_20, range_20_norm, BB_width_norm, parkinson_vol
3. **技术指标特征（4维）**: EMA20, stoch, MACD, VWAP
4. **成交量特征（4维）**: volume, volume_zscore, volume_change_1, OBV_slope_20
5. **K线形态特征（7维）**: pos_in_range_20, dist_to_HH20_norm, dist_to_LL20_norm, body_ratio, upper_shadow_ratio, lower_shadow_ratio, FVG
6. **时间周期特征（2维）**: sin_tod, cos_tod

---

## 依赖要求

确保已安装所有必需的Python包：

```bash
pip install -r requirements.txt
```

主要依赖：
- pandas
- numpy
- scikit-learn
- yfinance
- ta (技术分析库)
- scipy
- statsmodels
- matplotlib
- seaborn
- joblib
- pyyaml

---

## 配置文件

训练流程使用 `configs/config.yaml` 中的配置：

```yaml
data:
  symbols: ["ES=F", "NQ=F"]
  interval: "5m"
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  max_consecutive_missing: 5
  outlier_sigma: 3.0
  target_timezone: "UTC"
```

---

## 注意事项

1. **数据下载**: 首次运行可能需要较长时间下载数据
2. **内存使用**: 处理大量数据时注意内存占用
3. **特征验证**: VIF计算可能较慢，可以调整样本数量
4. **模型训练**: Random Forest训练时间较长，可以减少树的数量
5. **日志文件**: 所有操作都会记录到logs目录

---

## 故障排除

### 问题1: 数据下载失败
- 检查网络连接
- 确认yfinance可以访问
- 尝试减少日期范围

### 问题2: 内存不足
- 减少数据量
- 分批处理
- 增加系统内存

### 问题3: 特征验证很慢
- 减少置换重复次数（n_repeats）
- 使用更简单的模型
- 减少样本数量

---

## 联系方式

如有问题，请查看日志文件或联系开发团队。