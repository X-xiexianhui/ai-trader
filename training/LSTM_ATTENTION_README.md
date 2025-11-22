# LSTM+Attention 神经网络使用指南

## 概述

本项目实现了基于PyTorch的LSTM+Attention神经网络，用于金融市场状态分析和预测。

### 模型架构

```
输入 (OHLCV) → LSTM层 → Attention层 → 市场状态向量 → 多任务输出
                                                    ├─ 分类：市场状态
                                                    ├─ 回归：波动率预测
                                                    └─ 回归：收益率预测
```

### 主要功能

1. **市场状态向量提取**：将K线数据编码为低维状态向量
2. **市场状态分类**：预测未来5根K线的市场状态（上涨/下跌/震荡/反转）
3. **波动率预测**：预测未来5根K线的波动率
4. **收益率预测**：预测未来5根K线的收益率

## 文件结构

```
src/models/
├── lstm_attention.py      # 模型定义
└── data_loader.py         # 数据加载器

training/
├── train_lstm_attention.py    # 训练脚本
├── test_lstm_attention.py     # 测试评估脚本
└── LSTM_ATTENTION_README.md   # 本文档
```

## 使用步骤

### 1. 准备数据

首先需要准备训练、验证和测试数据集：

```bash
# 如果还没有划分数据集，运行：
python training/split_dataset.py
```

这将在 `data/processed/` 目录下生成：
- `MES_train.csv` - 训练集
- `MES_val.csv` - 验证集
- `MES_test.csv` - 测试集

### 2. 训练模型

```bash
python training/train_lstm_attention.py
```

**训练参数**（可在脚本中修改）：
- `input_size`: 5 (OHLCV)
- `hidden_size`: 128
- `num_layers`: 2
- `state_vector_size`: 64
- `seq_len`: 60 (输入序列长度)
- `future_periods`: 5 (预测未来K线数)
- `batch_size`: 64
- `num_epochs`: 100
- `learning_rate`: 0.001

**训练输出**：
- 模型检查点：`models/lstm_attention/checkpoint_best.pth`
- 训练配置：`models/lstm_attention/config.json`
- 归一化器：`models/lstm_attention/scaler.pkl`
- 训练历史：`models/lstm_attention/training_history.json`
- TensorBoard日志：`logs/tensorboard/`

**监控训练过程**：
```bash
tensorboard --logdir=logs/tensorboard
```

### 3. 测试和评估

```bash
python training/test_lstm_attention.py
```

**评估内容**：

#### 3.1 市场状态向量聚类
- 使用K-means对状态向量进行聚类（默认8个聚类）
- 分析每个聚类的市场特征
- 生成t-SNE可视化

#### 3.2 回归任务评估
对波动率和收益率预测计算：
- **MSE** (均方误差)
- **MAE** (平均绝对误差)
- **R²** (决定系数)

#### 3.3 分类任务评估
对市场状态分类计算：
- **准确率** (Accuracy)
- **精确度** (Precision)
- **召回率** (Recall)
- **F1值**
- 混淆矩阵

**评估输出**：
- 评估报告：`training/output/lstm_attention_evaluation/evaluation_report.json`
- 文本报告：`training/output/lstm_attention_evaluation/evaluation_report.txt`
- 聚类统计：`training/output/lstm_attention_evaluation/cluster_stats.json`
- 聚类模型：`training/output/lstm_attention_evaluation/kmeans_model.pkl`
- 可视化图表：
  - `cluster_visualization.png` - 聚类可视化
  - `regression_results.png` - 回归结果
  - `confusion_matrix.png` - 混淆矩阵

## 模型使用示例

### 加载训练好的模型

```python
import torch
import json
from src.models.lstm_attention import create_model

# 加载配置
with open('models/lstm_attention/config.json', 'r') as f:
    config = json.load(f)

# 创建模型
model = create_model(config)

# 加载权重
checkpoint = torch.load('models/lstm_attention/checkpoint_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 进行预测

```python
import torch
import pickle
import numpy as np

# 加载归一化器
with open('models/lstm_attention/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 准备输入数据 (60根K线的OHLCV)
# ohlcv_data shape: (60, 5)
ohlcv_normalized = scaler.transform(ohlcv_data)

# 转换为tensor
x = torch.FloatTensor(ohlcv_normalized).unsqueeze(0)  # (1, 60, 5)

# 预测
with torch.no_grad():
    outputs = model(x)
    
    # 获取结果
    state_vector = outputs['state_vector']  # 市场状态向量
    class_pred = torch.argmax(outputs['class_logits'], dim=1)  # 市场状态类别
    volatility = outputs['volatility']  # 波动率预测
    returns = outputs['returns']  # 收益率预测

print(f"市场状态: {['上涨', '下跌', '震荡', '反转'][class_pred.item()]}")
print(f"预测波动率: {volatility.item():.4f}")
print(f"预测收益率: {returns.item():.4f}")
```

### 提取市场状态向量

```python
# 仅提取状态向量（用于聚类等下游任务）
state_vector = model.get_state_vector(x)
print(f"状态向量维度: {state_vector.shape}")  # (1, 64)
```

## 模型架构详解

### 1. LSTM层
- 输入：OHLCV序列 `[batch_size, seq_len, 5]`
- 输出：隐藏状态序列 `[batch_size, seq_len, hidden_size]`
- 作用：提取时序特征

### 2. Attention层
- 输入：LSTM隐藏状态
- 输出：加权上下文向量 `[batch_size, hidden_size]`
- 作用：关注重要时间步

### 3. 状态投影层
- 输入：上下文向量
- 输出：市场状态向量 `[batch_size, state_vector_size]`
- 作用：降维并提取核心特征

### 4. 多任务头
- **分类头**：预测市场状态类别
- **波动率头**：预测未来波动率
- **收益率头**：预测未来收益率

## 损失函数

使用多任务学习损失，结合三个任务：

```
Total Loss = α × Classification Loss + β × Volatility Loss + γ × Returns Loss
```

默认权重：α = β = γ = 1.0

## 性能优化建议

### 1. 超参数调优
- 调整 `hidden_size` 和 `num_layers` 改变模型容量
- 调整 `seq_len` 改变输入窗口大小
- 调整 `state_vector_size` 改变状态向量维度

### 2. 训练技巧
- 使用学习率调度器（已实现ReduceLROnPlateau）
- 使用早停机制防止过拟合
- 使用梯度裁剪防止梯度爆炸
- 调整损失权重平衡各任务

### 3. 数据增强
- 可以尝试不同的归一化方法（standard/minmax）
- 调整市场状态标签的阈值参数
- 增加更多的技术指标作为输入

## 常见问题

### Q1: 训练时显存不足
**A**: 减小 `batch_size` 或 `hidden_size`

### Q2: 模型过拟合
**A**: 
- 增加 `dropout` 比例
- 减小模型容量
- 使用更多训练数据
- 调整早停的 `patience` 参数

### Q3: 分类准确率低
**A**:
- 检查数据集的类别分布是否平衡
- 调整市场状态标签的阈值
- 增加分类任务的损失权重 `alpha`

### Q4: 回归预测效果差
**A**:
- 检查目标变量的分布
- 尝试对目标变量进行变换（如log变换）
- 增加回归任务的损失权重 `beta` 或 `gamma`

## 扩展功能

### 1. 添加更多输入特征
修改 `data_loader.py` 中的数据处理逻辑，添加技术指标等特征。

### 2. 修改预测目标
调整 `future_periods` 参数改变预测时间跨度。

### 3. 自定义市场状态定义
修改 `MarketStateLabeler` 类中的标签生成逻辑。

### 4. 集成到交易系统
使用训练好的模型生成交易信号：
```python
# 获取市场状态向量
state_vector = model.get_state_vector(current_data)

# 使用聚类模型判断当前市场状态
cluster_label = kmeans.predict(state_vector.numpy())

# 根据聚类结果制定交易策略
if cluster_label in high_volatility_clusters:
    # 降低仓位
    pass
elif cluster_label in uptrend_clusters:
    # 做多
    pass
```

## 参考资料

- PyTorch官方文档: https://pytorch.org/docs/
- LSTM论文: Hochreiter & Schmidhuber (1997)
- Attention机制: Bahdanau et al. (2014)
- 多任务学习: Caruana (1997)

## 更新日志

- 2024-01-22: 初始版本发布
  - 实现LSTM+Attention模型
  - 实现多任务学习框架
  - 实现完整的训练和评估流程

## 联系方式

如有问题或建议，请提交Issue或Pull Request。