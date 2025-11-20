# 防过拟合策略文档

本文档详细介绍AI交易系统中实现的各种防过拟合策略和正则化技术。

## 目录

1. [概述](#概述)
2. [早停机制](#早停机制)
3. [梯度裁剪](#梯度裁剪)
4. [Dropout正则化](#dropout正则化)
5. [权重衰减](#权重衰减)
6. [数据增强](#数据增强)
7. [模型集成](#模型集成)
8. [交叉验证](#交叉验证)
9. [使用示例](#使用示例)

---

## 概述

过拟合是机器学习中的常见问题，特别是在金融时间序列预测中。本系统实现了多种防过拟合策略：

- **早停机制**: 监控验证集性能，防止过度训练
- **梯度裁剪**: 防止梯度爆炸，稳定训练
- **Dropout**: 随机丢弃神经元，增强泛化能力
- **权重衰减**: L2正则化，惩罚大权重
- **数据增强**: 增加数据多样性
- **模型集成**: 组合多个模型，降低方差
- **交叉验证**: 评估模型泛化能力

---

## 早停机制

### 原理

早停机制通过监控验证集上的性能指标，当性能不再改善时提前停止训练，防止过拟合。

### 实现

```python
from src.training.regularization import EarlyStopping

# 创建早停器
early_stopping = EarlyStopping(
    patience=10,           # 容忍10个epoch不改善
    min_delta=0.0001,     # 最小改善幅度
    mode='min',           # 指标越小越好
    save_path='best_model.pt'
)

# 训练循环中使用
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)
    
    # 检查是否应该早停
    if early_stopping(val_loss, model, epoch):
        print(f"早停触发，最佳epoch: {early_stopping.best_epoch}")
        break

# 加载最佳模型
early_stopping.load_best_model(model)
```

### 参数说明

- `patience`: 容忍的epoch数，超过此数量仍无改善则停止
- `min_delta`: 最小改善幅度，小于此值不算改善
- `mode`: 'min'表示指标越小越好，'max'表示越大越好
- `save_path`: 最佳模型保存路径

---

## 梯度裁剪

### 原理

梯度裁剪通过限制梯度的范数或值，防止梯度爆炸，使训练更稳定。

### 实现

```python
from src.training.regularization import GradientClipper

# 创建梯度裁剪器
clipper = GradientClipper(
    max_norm=1.0,         # 最大梯度范数
    norm_type=2.0,        # L2范数
    clip_method='norm'    # 裁剪方法
)

# 训练循环中使用
optimizer.zero_grad()
loss.backward()

# 裁剪梯度
grad_norm = clipper.clip(model.parameters())
print(f"梯度范数: {grad_norm:.4f}")

optimizer.step()
```

### 裁剪方法

- **norm**: 按范数裁剪，保持梯度方向
- **value**: 按值裁剪，限制每个梯度的绝对值

---

## Dropout正则化

### 原理

Dropout在训练时随机丢弃一部分神经元，迫使网络学习更鲁棒的特征表示。

### 实现

```python
import torch.nn as nn

# 在模型中添加Dropout层
class MyModel(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        return x

# 使用Dropout调度器
from src.training.regularization import DropoutScheduler

scheduler = DropoutScheduler(
    model=model,
    initial_dropout=0.1,
    final_dropout=0.3,
    warmup_epochs=10
)

# 训练循环中更新Dropout率
for epoch in range(num_epochs):
    scheduler.step(epoch)
    train_one_epoch(model, train_loader)
```

---

## 权重衰减

### 原理

权重衰减（L2正则化）在损失函数中添加权重的L2范数惩罚项，防止权重过大。

### 实现

```python
import torch.optim as optim

# 在优化器中设置权重衰减
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001  # L2正则化系数
)

# 使用权重衰减调度器
from src.training.regularization import WeightDecayScheduler

wd_scheduler = WeightDecayScheduler(
    optimizer=optimizer,
    initial_wd=0.0001,
    final_wd=0.001,
    warmup_epochs=10
)

# 训练循环中更新权重衰减
for epoch in range(num_epochs):
    wd_scheduler.step(epoch)
    train_one_epoch(model, train_loader)
```

---

## 数据增强

### Mixup增强

Mixup通过线性插值混合样本和标签，增加数据多样性。

```python
from src.training.regularization import MixupAugmentation

mixup = MixupAugmentation(alpha=0.2)

# 训练循环中使用
for batch_x, batch_y in train_loader:
    # 应用Mixup
    mixed_x, mixed_y, lam = mixup(batch_x, batch_y)
    
    # 使用混合后的数据训练
    output = model(mixed_x)
    loss = criterion(output, mixed_y)
    loss.backward()
    optimizer.step()
```

### 标签平滑

标签平滑通过软化one-hot标签，防止模型过度自信。

```python
from src.training.regularization import LabelSmoothing

criterion = LabelSmoothing(smoothing=0.1)

# 使用标签平滑损失
output = model(batch_x)
loss = criterion(output, batch_y)
```

---

## 模型集成

### 简单集成

```python
from src.training.ensemble import ModelEnsemble

# 创建集成器
ensemble = ModelEnsemble(method='average')

# 添加多个模型
for i in range(5):
    model = create_model()
    train_model(model, train_data)
    ensemble.add_model(model)

# 集成预测
predictions, uncertainty = ensemble.predict(test_x, return_std=True)
```

### Bootstrap集成

```python
from src.training.ensemble import BootstrapEnsemble

# 创建Bootstrap集成
bootstrap = BootstrapEnsemble(
    base_model_class=MyModel,
    model_kwargs={'input_dim': 256},
    n_models=5,
    sample_ratio=0.8
)

# 训练集成
bootstrap.fit(train_dataset, train_function)

# 预测
predictions = bootstrap.predict(test_x)
```

### 快照集成

```python
from src.training.ensemble import SnapshotEnsemble

# 创建快照集成
snapshot = SnapshotEnsemble(
    model=model,
    n_snapshots=5,
    cycle_length=10
)

# 训练循环中保存快照
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)
    
    if snapshot.should_save_snapshot(epoch):
        snapshot.save_snapshot()

# 预测
predictions = snapshot.predict(test_x)
```

---

## 交叉验证

### 时间序列交叉验证

```python
from src.training.cross_validation import TimeSeriesSplit, CrossValidator

# 创建分割器
splitter = TimeSeriesSplit(
    n_splits=5,
    test_size=None,  # 自动计算
    gap=10           # 训练和测试之间的间隔
)

# 创建验证器
validator = CrossValidator(
    splitter=splitter,
    scoring_fn=lambda y_true, y_pred: -np.mean((y_true - y_pred)**2)
)

# 执行交叉验证
results = validator.validate(
    model_fn=create_model,
    X=features,
    y=labels
)

print(f"平均分数: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
```

### Walk-Forward验证

```python
from src.training.cross_validation import WalkForwardValidation

# 创建Walk-Forward验证器
wfv = WalkForwardValidation(
    train_period=500,
    test_period=100,
    step_size=100,
    gap=0
)

# 分割数据
splits = wfv.split(data)

# 对每个分割进行训练和测试
for train_idx, test_idx in splits:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = create_model()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.4f}")
```

---

## 使用示例

### 完整训练流程

```python
from src.training.regularization import RegularizationManager
from src.utils.config_loader import load_config

# 加载配置
config = load_config('configs/regularization_config.yaml')

# 创建正则化管理器
reg_manager = RegularizationManager(config)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    
    for batch_x, batch_y in train_loader:
        # 应用Mixup（如果启用）
        batch_x, batch_y, lam = reg_manager.apply_mixup(batch_x, batch_y)
        
        # 前向传播
        output = model(batch_x)
        
        # 计算损失（使用标签平滑）
        if reg_manager.label_smoothing:
            loss = reg_manager.get_label_smoothing_loss()(output, batch_y)
        else:
            loss = criterion(output, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = reg_manager.clip_gradients(model.parameters())
        
        # 更新参数
        optimizer.step()
    
    # 验证
    val_loss = validate(model, val_loader)
    
    # 检查早停
    if reg_manager.check_early_stopping(val_loss, model, epoch):
        print("早停触发")
        break
```

---

## 最佳实践

1. **组合使用**: 同时使用多种正则化技术效果更好
2. **适度正则化**: 过度正则化会导致欠拟合
3. **监控指标**: 同时监控训练集和验证集性能
4. **超参数调优**: 使用交叉验证选择最佳超参数
5. **数据质量**: 高质量数据比复杂正则化更重要

---

## 配置文件

完整的配置示例请参考 [`configs/regularization_config.yaml`](../configs/regularization_config.yaml)

---

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)
- [Time Series Cross-Validation](https://robjhyndman.com/hyndsight/tscv/)