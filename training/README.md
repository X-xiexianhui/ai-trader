# 训练数据处理

本目录包含用于下载和处理训练数据的完整脚本。

## 1. MES数据下载

### 脚本说明

[`download_mes_data.py`](download_mes_data.py) - 下载MES（Micro E-mini S&P 500期货）的5分钟K线数据

### 使用方法

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行下载脚本（默认下载最近60天数据）
python training/download_mes_data.py
```

### 输出文件

脚本会在 `training/output/` 目录下生成以下文件：

- **mes_5m_data.csv** - CSV格式的K线数据（761KB）
- **mes_5m_data.parquet** - Parquet格式的K线数据（233KB，更高效）
- **download_mes.log** - 下载日志文件

### 数据格式

CSV文件包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| datetime | 时间戳（带时区） | 2025-09-23 00:05:00-04:00 |
| symbol | 品种代码 | MES=F |
| open | 开盘价 | 6750.75 |
| high | 最高价 | 6751.00 |
| low | 最低价 | 6750.75 |
| close | 收盘价 | 6750.75 |
| volume | 成交量 | 0 |

### 数据统计

最近一次下载的数据统计：

- **记录数量**: 11,680条
- **时间范围**: 2025-09-23 至 2025-11-20（约59天）
- **价格范围**: 6542.75 - 6952.75
- **平均成交量**: 4,847
- **数据完整性**: 100%

### 注意事项

1. **数据限制**: 雅虎金融的5分钟K线数据最多只能获取最近60天的数据
2. **品种代码**: 
   - 主要使用 `MES=F`（Micro E-mini S&P 500）
   - 如果不可用，会自动尝试 `ES=F`（标准E-mini S&P 500）或 `^GSPC`（S&P 500指数）
3. **时区**: 数据包含时区信息（美国东部时间）
4. **数据质量**: 脚本会自动验证数据的完整性和一致性

### 依赖项

```bash
pip install yfinance pandas colorlog
```

或使用项目的requirements.txt：

```bash
pip install -r requirements.txt
```

### 自定义下载

如果需要修改下载参数，可以编辑脚本中的 `download_mes_data()` 函数：

```python
# 修改下载天数（最大59天）
df = download_mes_data(days=30)

# 修改保存路径
df = download_mes_data(save_path='custom/path/data.csv')
```

### 故障排除

**问题**: 下载失败或返回空数据

**解决方案**:
1. 检查网络连接
2. 确认雅虎金融服务可用
3. 尝试减少下载天数
4. 查看日志文件 `training/output/download_mes.log` 获取详细错误信息

**问题**: ModuleNotFoundError

**解决方案**:
```bash
# 确保已激活虚拟环境
source .venv/bin/activate

# 安装缺失的依赖
pip install -r requirements.txt
```

## 2. MES数据特征处理

### 脚本说明

[`process_mes_features.py`](process_mes_features.py) - 完整的数据处理流程

执行以下步骤：
1. 读取本地mes_5m_data.csv文件
2. 数据清洗（缺失值、异常值、时间对齐）
3. 过滤非交易时段（UTC 22:00-23:00休市时段）
4. 计算27个手工特征
5. 特征归一化
6. 特征验证并生成详细报告

### 使用方法

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行特征处理脚本
python training/process_mes_features.py
```

### 输出文件

脚本会在 `training/output/` 目录下生成以下文件：

#### 处理后的数据
- **mes_features_normalized.csv** (6.6MB) - 归一化后的特征数据（CSV格式）
- **mes_features_normalized.parquet** (2.3MB) - 归一化后的特征数据（Parquet格式，推荐）

#### 归一化器
- **scalers/** - 保存的特征归一化器
  - `price_return_scaler.pkl` - 价格收益特征的StandardScaler
  - `volatility_scaler.pkl` - 波动率特征的RobustScaler
  - `technical_scaler.pkl` - 技术指标特征的RobustScaler
  - `volume_scaler.pkl` - 成交量特征的RobustScaler
  - `candlestick_scaler.pkl` - K线形态特征的StandardScaler
  - `feature_groups.pkl` - 特征分组信息

#### 报告文件
- **feature_validation_report.txt** - 详细的特征验证报告
- **processing_summary_report.txt** - 处理摘要报告
- **process_features.log** - 完整的处理日志

### 27个手工特征详解

#### 1. 价格与收益特征（5个）
- **ret_1**: 1周期对数收益率
- **ret_5**: 5周期对数收益率
- **ret_20**: 20周期对数收益率
- **price_slope_20**: 20周期价格线性回归斜率
- **C_div_MA20**: 收盘价相对20周期均线的比值

#### 2. 波动率特征（5个）
- **ATR14_norm**: 归一化ATR(14)
- **vol_20**: 20周期收盘价标准差
- **range_20_norm**: 归一化20周期价格范围
- **BB_width_norm**: 归一化布林带宽度
- **parkinson_vol**: Parkinson波动率估计（对数变换）

#### 3. 技术指标特征（4个）
- **EMA20**: 20周期指数移动平均
- **stoch**: 随机指标%K值
- **MACD**: MACD指标线
- **VWAP**: 成交量加权平均价（20周期）

#### 4. 成交量特征（4个）
- **volume**: 原始成交量
- **volume_zscore**: 成交量Z-score
- **volume_change_1**: 成交量变化率
- **OBV_slope_20**: OBV的20周期线性回归斜率

#### 5. K线形态特征（7个）
- **pos_in_range_20**: 在20周期范围内的相对位置
- **dist_to_HH20_norm**: 到最高点的归一化距离
- **dist_to_LL20_norm**: 到最低点的归一化距离
- **body_ratio**: K线实体比例
- **upper_shadow_ratio**: 上影线比例
- **lower_shadow_ratio**: 下影线比例
- **FVG**: 公允价值缺口

#### 6. 时间周期特征（2个）
- **sin_tod**: 时间的正弦编码
- **cos_tod**: 时间的余弦编码

### 处理结果统计

最近一次处理的统计信息：

#### 数据概览
- **原始数据**: 11,680行
- **清洗后数据**: 11,358行（过滤322行非交易时段）
- **最终数据**: 11,321行
- **时间范围**: 2025-09-23 至 2025-11-21（约59天）

#### 数据清洗
- **缺失值**: 0个（数据质量良好）
- **异常值**: 检测232个，修正232个尖峰
- **质量验证**: ✓ 通过

#### 特征验证结果

**Top 5 信息量最高的特征**:
1. `ret_1`: R²=0.0154, MI=0.7854
2. `parkinson_vol`: R²=0.0001, MI=0.6563
3. `dist_to_HH20_norm`: R²=0.0025, MI=0.4899
4. `ret_5`: R²=0.0058, MI=0.4271
5. `range_20_norm`: R²=0.0001, MI=0.3299

**高度相关特征对**（相关系数>0.85）:
- `vol_20` ↔ `BB_width_norm`: 0.9998
- `EMA20` ↔ `VWAP`: 0.9995
- `range_20_norm` ↔ `BB_width_norm`: 0.9524
- `vol_20` ↔ `range_20_norm`: 0.9518
- `ret_20` ↔ `price_slope_20`: 0.9259

**多重共线性特征**（VIF>10）:
- 15个特征存在多重共线性
- 最高VIF: `BB_width_norm` (8185.65), `vol_20` (8120.01)

### 特征归一化方法

不同特征组使用不同的归一化方法：

| 特征组 | 归一化方法 | 说明 |
|--------|-----------|------|
| price_return | StandardScaler | z-score标准化，适合正态分布 |
| volatility | RobustScaler | 基于中位数和IQR，对异常值鲁棒 |
| technical | RobustScaler | 基于中位数和IQR，对异常值鲁棒 |
| volume | RobustScaler | 基于中位数和IQR，对异常值鲁棒 |
| candlestick | StandardScaler | z-score标准化 |
| time | 无需归一化 | 已在[-1,1]范围内 |

### 注意事项

1. **交易时段过滤**: 自动过滤UTC 22:00-23:00的休市时段数据
2. **特征相关性**: 部分特征高度相关，建议在模型训练时考虑特征选择
3. **多重共线性**: 15个特征存在多重共线性，可能需要降维或特征选择
4. **归一化器保存**: 所有scaler已保存，可用于推理时的特征转换

### 后续步骤

处理后的数据可用于：

1. **TS2Vec训练**: 使用 [`train_ts2vec.py`](train_ts2vec.py) 训练时序表示学习模型
2. **特征选择**: 基于验证报告移除冗余特征
3. **模型评估**: 使用 [`src/evaluation/`](../src/evaluation/) 中的评估工具
4. **回测验证**: 使用 [`src/backtest/`](../src/backtest/) 进行策略回测

## 3. TS2Vec模型训练

### 脚本说明

[`train_ts2vec.py`](train_ts2vec.py) - TS2Vec时序表示学习模型训练

执行以下步骤：
1. 从本地文件读取处理后的数据
2. 提取OHLC价格数据
3. 生成滑动窗口和对比学习样本对
4. 训练TS2Vec模型（膨胀卷积编码器）
5. 保存训练好的模型和训练历史

### 使用方法

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行TS2Vec训练脚本
python training/train_ts2vec.py
```

### 输出文件

脚本会生成以下文件：

#### 模型文件
- **models/checkpoints/ts2vec/best_model.pt** - 最佳模型检查点
- **models/checkpoints/ts2vec/checkpoint_epoch_*.pt** - 各epoch的检查点（如果配置保存）

#### 训练报告
- **training/output/ts2vec_training_summary.txt** - 训练摘要报告
- **training/output/ts2vec_training_history.png** - 训练历史曲线图
- **training/output/train_ts2vec.log** - 完整的训练日志

### TS2Vec模型架构

#### 核心组件

1. **膨胀卷积编码器**
   - 10层膨胀卷积层
   - 膨胀率: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
   - 隐藏维度: 256
   - 卷积核大小: 3
   - 残差连接 + LayerNorm

2. **投影头网络**
   - 2层全连接网络
   - 输入维度: 256
   - 输出维度: 128
   - L2归一化

3. **NT-Xent对比损失**
   - 温度参数: 0.1
   - 归一化温度交叉熵损失

#### 模型参数

- **总参数量**: ~1.3M
- **可训练参数**: ~1.3M
- **输入维度**: 4 (OHLC)
- **输出维度**: 256 (编码器) / 128 (投影头)

### 训练配置

#### 数据配置
- **窗口长度**: 256个时间步（约21小时的5分钟数据）
- **滑动步长**: 1
- **训练/验证划分**: 80% / 20%
- **批次大小**: 64

#### 数据增强
- **时间遮蔽**: 20%的时间步随机遮蔽
- **时间扭曲**: ±5%的时间轴拉伸/压缩
- **幅度缩放**: 0.9-1.1倍的统一缩放

#### 训练参数
- **训练轮数**: 100 epochs
- **学习率**: 0.001
- **优化器**: Adam
- **学习率调度**: Warmup (5 epochs) + Cosine Annealing
- **早停patience**: 10 epochs
- **最小改善**: 0.001

### 训练流程

1. **数据准备**
   ```
   原始数据 (11,321行)
   ↓
   滑动窗口生成 (~11,065个窗口)
   ↓
   训练/验证划分 (8,852 / 2,213)
   ↓
   对比样本对生成 (每个窗口生成2个增强视图)
   ```

2. **训练过程**
   ```
   每个epoch:
   - 训练阶段: 遍历所有训练批次
   - 验证阶段: 评估验证集损失
   - 学习率更新: Warmup或Cosine调度
   - 检查点保存: 保存最佳模型
   - 早停检查: 监控验证损失
   ```

3. **输出生成**
   ```
   训练完成后:
   - 保存最佳模型
   - 绘制训练历史曲线
   - 生成训练摘要报告
   ```

### 预期训练结果

基于类似配置的经验值：

- **训练时间**: 约30-60分钟（CPU）/ 5-10分钟（GPU）
- **最佳验证损失**: 约0.5-1.0
- **收敛epoch**: 约30-50 epochs
- **内存占用**: 约2-4GB

### 使用训练好的模型

训练完成后，可以使用模型生成时序表示：

```python
import torch
from src.models.ts2vec.model import TS2VecModel

# 加载模型
model = TS2VecModel.load('models/checkpoints/ts2vec/best_model.pt')
model.eval()

# 生成embedding
with torch.no_grad():
    # x: [batch, seq_len, 4] OHLC数据
    embeddings = model.encode(x, return_projection=False)
    # embeddings: [batch, seq_len, 256]
```

### 配置修改

如需修改训练参数，编辑 [`configs/config.yaml`](../configs/config.yaml) 中的 `ts2vec` 部分：

```yaml
ts2vec:
  # 模型结构
  hidden_dim: 256        # 编码器隐藏维度
  num_layers: 10         # 卷积层数
  
  # 训练配置
  window_length: 256     # 窗口长度
  batch_size: 64         # 批次大小
  num_epochs: 100        # 训练轮数
  learning_rate: 0.001   # 学习率
  
  # 早停
  patience: 10           # 早停patience
```

### 注意事项

1. **计算资源**:
   - CPU训练较慢，建议使用GPU
   - 至少需要4GB内存
   
2. **数据要求**:
   - 需要先运行 `process_mes_features.py` 生成处理后的数据
   - 数据文件: `training/output/mes_features_normalized.parquet`

3. **模型保存**:
   - 只保存最佳模型（基于验证损失）
   - 可修改代码保存所有epoch的检查点

4. **训练监控**:
   - 实时查看日志: `tail -f training/output/train_ts2vec.log`
   - 训练完成后查看曲线图和摘要报告

### 故障排除

**问题**: CUDA out of memory

**解决方案**:
```yaml
# 减小批次大小
batch_size: 32  # 或更小
```

**问题**: 训练损失不下降

**解决方案**:
1. 检查数据质量
2. 调整学习率
3. 增加训练轮数
4. 检查数据增强参数

**问题**: 验证损失震荡

**解决方案**:
1. 增加早停patience
2. 调整学习率调度
3. 检查验证集大小

## 4. TS2Vec模型评估

### 脚本说明

[`evaluate_ts2vec.py`](evaluate_ts2vec.py) - 评估训练好的TS2Vec模型质量

执行以下评估任务：
1. **Embedding质量评估**: 评估正负样本对的相似度分离
2. **线性探测**: 冻结模型权重，训练线性分类器测试表示质量
3. **聚类质量**: 评估embedding的聚类效果（轮廓系数）
4. **t-SNE可视化**: 可视化embedding的分布
5. **生成评估报告**: 完整的评估结果报告

### 使用方法

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行评估脚本（需要先训练模型）
python training/evaluate_ts2vec.py
```

### 评估指标

#### 1. Embedding质量
- **正样本相似度**: 同一窗口的两个增强视图的余弦相似度
  - 目标: > 0.7
  - 越高越好，表示模型能识别相同时间序列
  
- **负样本相似度**: 不同窗口的余弦相似度
  - 目标: < 0.3
  - 越低越好，表示模型能区分不同时间序列
  
- **分离度**: 正样本相似度 - 负样本相似度
  - 目标: > 0.4
  - 越大越好，表示embedding空间结构良好

#### 2. 线性探测
- **测试准确率**: 使用线性分类器预测未来收益方向
  - 目标: > 0.52（略高于随机猜测0.5）
  - 表示embedding包含预测性信息
  
- **测试AUC**: ROC曲线下面积
  - 目标: > 0.55
  - 衡量分类器的整体性能

#### 3. 聚类质量
- **轮廓系数**: 衡量聚类的紧密度和分离度
  - 范围: [-1, 1]
  - 目标: > 0.2
  - 越高越好，表示embedding有良好的聚类结构

#### 4. t-SNE可视化
- 将高维embedding降维到2D进行可视化
- 用颜色标注涨跌方向
- 观察不同类别的分离程度

### 输出文件

- **training/output/ts2vec_evaluation_report.txt** - 完整评估报告
- **training/output/ts2vec_tsne.png** - t-SNE可视化图
- **training/output/evaluate_ts2vec.log** - 评估日志

### 评估结果解读

#### 良好的模型应该具有：
1. **高正样本相似度** (>0.7): 表示模型能学习到时间序列的不变特征
2. **低负样本相似度** (<0.3): 表示模型能区分不同的时间序列模式
3. **较高的线性探测准确率** (>0.52): 表示embedding包含有用的预测信息
4. **正的轮廓系数** (>0.2): 表示embedding有良好的聚类结构
5. **t-SNE图中不同类别有一定分离**: 表示embedding捕获了有意义的模式

#### 如果评估结果不理想：
1. **正样本相似度过低**:
   - 增加训练轮数
   - 调整数据增强强度
   - 检查数据质量

2. **负样本相似度过高**:
   - 增加对比损失的温度参数
   - 增加批次大小
   - 调整模型容量

3. **线性探测准确率低**:
   - 可能是任务本身难度高（金融数据噪声大）
   - 尝试不同的预测目标
   - 增加训练数据量

4. **聚类质量差**:
   - 调整聚类数量
   - 增加模型容量
   - 改进数据预处理

### 使用评估结果

评估完成后，可以根据结果决定：

1. **模型是否可用**: 如果所有指标都达到目标，模型可以用于下游任务
2. **是否需要重新训练**: 如果关键指标不达标，需要调整超参数重新训练
3. **如何改进**: 根据具体的评估结果调整训练策略

### 注意事项

1. **评估需要训练好的模型**: 确保 `models/checkpoints/ts2vec/best_model.pt` 存在
2. **评估时间**: 约5-10分钟（取决于数据量和设备）
3. **内存占用**: t-SNE可能占用较多内存，已限制采样数量
4. **随机性**: 某些评估（如聚类、t-SNE）有随机性，多次运行结果可能略有不同