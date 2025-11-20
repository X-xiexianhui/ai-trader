# TS2Vec模块文档

## 概述

TS2Vec (Time Series to Vector) 是一个用于时间序列表示学习的自监督模型。它通过对比学习的方式学习时间序列的通用表示，无需标签数据。

## 模块结构

```
src/models/ts2vec/
├── __init__.py           # 模块初始化
├── model.py              # 核心模型架构
├── augmentation.py       # 数据增强策略
├── loss.py               # 对比学习损失函数
├── dataset.py            # 数据集和数据加载器
├── trainer.py            # 训练器
└── evaluator.py          # 评估器和embedding提取器
```

## 核心组件

### 1. 模型架构 (model.py)

#### DilatedCNNEncoder
- **功能**: 使用膨胀卷积提取时间序列特征
- **特点**: 
  - 10层膨胀卷积，膨胀率指数增长 (1, 2, 4, ..., 512)
  - 大感受野，能捕获长期依赖
  - 残差连接，缓解梯度消失

```python
from src.models.ts2vec import DilatedCNNEncoder

encoder = DilatedCNNEncoder(
    input_dim=27,
    hidden_dim=64,
    num_layers=10,
    kernel_size=3
)
```

#### ProjectionHead
- **功能**: 将编码器输出投影到对比学习空间
- **结构**: 256 → 128 → 128

```python
from src.models.ts2vec import ProjectionHead

proj_head = ProjectionHead(
    input_dim=256,
    hidden_dim=128,
    output_dim=128
)
```

#### TS2Vec
- **功能**: 完整的TS2Vec模型
- **输出**: 
  - Embedding (256维)
  - Projection (128维，用于对比学习)

```python
from src.models.ts2vec import TS2Vec

model = TS2Vec(
    input_dim=27,
    hidden_dim=64,
    num_layers=10,
    projection_dim=128
)

# 前向传播
embedding = model(x)  # (batch, 256)
embedding, projection = model(x, return_projection=True)  # (batch, 256), (batch, 128)
```

### 2. 数据增强 (augmentation.py)

#### TimeSeriesAugmentation
支持4种时间序列增强方法：

1. **时间遮蔽 (Temporal Masking)**
   - 随机遮蔽时间步
   - 保持时间序列结构

2. **幅度缩放 (Magnitude Scaling)**
   - 随机缩放幅度
   - 模拟不同市场条件

3. **时间平移 (Time Shifting)**
   - 随机平移时间序列
   - 增加时间不变性

4. **时间扭曲 (Time Warping)**
   - 非线性时间变换
   - 模拟速度变化

```python
from src.models.ts2vec import TimeSeriesAugmentation

augmenter = TimeSeriesAugmentation(
    temporal_masking=True,
    magnitude_scaling=True,
    time_shifting=True,
    time_warping=True
)

# 创建正样本对
view1, view2 = augmenter.create_positive_pair(x)
```

### 3. 损失函数 (loss.py)

#### NTXentLoss
- **功能**: 归一化温度缩放交叉熵损失
- **特点**: 标准的对比学习损失

```python
from src.models.ts2vec import NTXentLoss

criterion = NTXentLoss(temperature=0.07)
loss = criterion(z1, z2)
```

#### HierarchicalContrastiveLoss
- **功能**: 层次对比损失
- **特点**: 在多个时间尺度上进行对比

```python
from src.models.ts2vec import HierarchicalContrastiveLoss

criterion = HierarchicalContrastiveLoss(temperature=0.07)
loss = criterion(z1, z2)
```

#### TS2VecLoss
- **功能**: 完整的TS2Vec损失
- **组成**: NT-Xent + 层次对比损失

```python
from src.models.ts2vec import TS2VecLoss

criterion = TS2VecLoss(
    temperature=0.07,
    hierarchical_weight=0.5
)
loss = criterion(z1, z2)
```

### 4. 数据集 (dataset.py)

#### SlidingWindowDataset
- **功能**: 滑动窗口采样
- **参数**:
  - window_size: 窗口大小 (默认256)
  - stride: 滑动步长 (默认1)

```python
from src.models.ts2vec import SlidingWindowDataset

dataset = SlidingWindowDataset(
    data=features,  # (N, feature_dim)
    window_size=256,
    stride=1
)
```

#### ContrastiveDataset
- **功能**: 对比学习数据集
- **输出**: 增强后的正样本对

```python
from src.models.ts2vec import ContrastiveDataset

dataset = ContrastiveDataset(
    data=features,
    window_size=256,
    augmenter=augmenter
)
```

#### 便捷函数
```python
from src.models.ts2vec import create_ts2vec_dataloader

train_loader = create_ts2vec_dataloader(
    data=train_data,
    window_size=256,
    stride=128,
    batch_size=32,
    shuffle=True,
    augmenter=augmenter
)
```

### 5. 训练器 (trainer.py)

#### LearningRateScheduler
- **功能**: Warmup + CosineAnnealing学习率调度
- **阶段**:
  1. Warmup: 线性增长 (前5个epoch)
  2. CosineAnnealing: 余弦退火

```python
from src.models.ts2vec import LearningRateScheduler

scheduler = LearningRateScheduler(
    optimizer=optimizer,
    warmup_epochs=5,
    total_epochs=100
)
```

#### EarlyStopping
- **功能**: 早停机制
- **监控**: 验证损失

```python
from src.models.ts2vec import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.0001
)
```

#### TS2VecTrainer
- **功能**: 完整训练流程
- **特性**:
  - 自动学习率调度
  - 早停
  - 模型保存
  - 训练历史记录

```python
from src.models.ts2vec import TS2VecTrainer

trainer = TS2VecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-3,
    device='cuda'
)

# 训练
history = trainer.train(
    num_epochs=100,
    warmup_epochs=5,
    early_stopping_patience=10
)

# 保存/加载
trainer.save_checkpoint(epoch=50, is_best=True)
trainer.load_checkpoint('models/ts2vec/best_model.pt')
```

### 6. 评估器 (evaluator.py)

#### TS2VecEvaluator
- **功能**: 评估embedding质量
- **方法**:
  1. Embedding统计
  2. 聚类质量
  3. 线性探测 (分类)
  4. 线性探测 (回归)

```python
from src.models.ts2vec import TS2VecEvaluator

evaluator = TS2VecEvaluator(model, device='cuda')

# 综合评估
results = evaluator.evaluate_comprehensive(
    data=test_data,
    labels=test_labels,
    targets=test_targets,
    n_clusters=5
)

print(results['statistics'])
print(results['clustering'])
print(results['linear_probing_classification'])
print(results['linear_probing_regression'])
```

#### EmbeddingExtractor
- **功能**: 批量提取embedding
- **特性**:
  - GPU加速
  - 批处理
  - 保存/加载

```python
from src.models.ts2vec import EmbeddingExtractor

extractor = EmbeddingExtractor(model, device='cuda')

# 提取embedding
embeddings = extractor.extract(data, batch_size=128)

# 保存
extractor.extract_and_save(data, 'embeddings.npy')

# 加载
embeddings = EmbeddingExtractor.load_embeddings('embeddings.npy')
```

## 完整训练流程

```python
import numpy as np
from src.models.ts2vec import (
    TS2Vec,
    TimeSeriesAugmentation,
    create_ts2vec_dataloader,
    TS2VecTrainer,
    TS2VecEvaluator
)

# 1. 准备数据
train_data = np.load('train_features.npy')  # (N, 27)
val_data = np.load('val_features.npy')

# 2. 创建数据增强器
augmenter = TimeSeriesAugmentation(
    temporal_masking=True,
    magnitude_scaling=True,
    time_shifting=True,
    time_warping=True
)

# 3. 创建数据加载器
train_loader = create_ts2vec_dataloader(
    data=train_data,
    window_size=256,
    stride=128,
    batch_size=32,
    shuffle=True,
    augmenter=augmenter
)

val_loader = create_ts2vec_dataloader(
    data=val_data,
    window_size=256,
    stride=256,
    batch_size=32,
    shuffle=False,
    augmenter=augmenter
)

# 4. 创建模型
model = TS2Vec(
    input_dim=27,
    hidden_dim=64,
    num_layers=10,
    projection_dim=128
)

# 5. 创建训练器
trainer = TS2VecTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-3,
    weight_decay=1e-5,
    temperature=0.07,
    device='cuda'
)

# 6. 训练
history = trainer.train(
    num_epochs=100,
    warmup_epochs=5,
    early_stopping_patience=10,
    save_best=True
)

# 7. 评估
evaluator = TS2VecEvaluator(model, device='cuda')
results = evaluator.evaluate_comprehensive(
    data=val_data,
    n_clusters=5
)

print(f"Silhouette Score: {results['clustering']['silhouette_score']:.4f}")
```

## 配置文件

配置文件位于 `configs/ts2vec_config.yaml`：

```yaml
model:
  input_dim: 27
  hidden_dim: 64
  num_layers: 10
  projection_dim: 128

data:
  window_size: 256
  stride: 1
  batch_size: 32

training:
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.00001

contrastive:
  temperature: 0.07
  hierarchical_weight: 0.5
```

## 性能优化建议

1. **数据加载**
   - 使用多线程: `num_workers=4`
   - 启用pin_memory: `pin_memory=True`

2. **训练加速**
   - 使用GPU: `device='cuda'`
   - 增大batch_size (如果显存允许)
   - 使用混合精度训练 (可选)

3. **内存优化**
   - 减小window_size
   - 增大stride
   - 使用梯度累积

## 常见问题

### Q1: 训练损失不下降？
- 检查学习率是否过大/过小
- 检查数据增强是否过强
- 检查温度参数是否合适

### Q2: 显存不足？
- 减小batch_size
- 减小window_size
- 减小hidden_dim

### Q3: Embedding质量不好？
- 增加训练轮数
- 调整数据增强策略
- 调整温度参数
- 增加模型容量

## 参考文献

1. Yue et al. "TS2Vec: Towards Universal Representation of Time Series" (AAAI 2022)
2. Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (ICML 2020)

## 更新日志

### v1.0.0 (2025-11-20)
- 初始版本
- 实现完整的TS2Vec模型
- 支持多种数据增强
- 完整的训练和评估流程