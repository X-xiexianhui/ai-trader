# Transformer模型文档

## 目录
1. [概述](#概述)
2. [模型架构](#模型架构)
3. [核心组件](#核心组件)
4. [使用指南](#使用指南)
5. [配置说明](#配置说明)
6. [训练流程](#训练流程)
7. [评估方法](#评估方法)
8. [API参考](#api参考)

---

## 概述

Transformer模型用于时间序列状态表征学习，通过监督学习预训练，为后续的强化学习提供高质量的状态表征。

### 主要特性

- **特征融合**: 融合TS2Vec embedding (128维) 和手工特征 (27维)
- **多头注意力**: 8个注意力头，捕捉不同时间尺度的依赖关系
- **位置编码**: 正弦位置编码，保持时间顺序信息
- **因果掩码**: 防止未来信息泄露
- **多任务学习**: 同时进行回归（收益率预测）和分类（方向预测）
- **深层架构**: 6层Transformer编码器

### 技术规格

| 参数 | 值 |
|------|-----|
| 输入维度 | 155 (128 + 27) |
| 模型维度 | 256 |
| 注意力头数 | 8 |
| 编码器层数 | 6 |
| 前馈网络维度 | 1024 |
| 输出维度 | 256 (状态向量) |
| 参数量 | ~3.5M |

---

## 模型架构

### 整体架构

```
输入: TS2Vec Embedding (128维) + 手工特征 (27维)
  ↓
特征融合层 (155 → 256)
  ↓
位置编码
  ↓
Transformer编码器 × 6层
  ├─ 多头自注意力 (8头)
  ├─ 残差连接 + LayerNorm
  ├─ 前馈网络 (256 → 1024 → 256)
  └─ 残差连接 + LayerNorm
  ↓
状态向量 (256维)
  ↓
辅助任务头
  ├─ 回归头: 预测未来收益率
  └─ 分类头: 预测价格方向 (上涨/下跌/持平)
```

### 关键设计

1. **特征融合策略**
   - 简单拼接: `[TS2Vec_emb, manual_features]`
   - 线性投影: `155 → 256`
   - Dropout正则化

2. **注意力机制**
   - Scaled Dot-Product Attention
   - 因果掩码（上三角掩码）
   - 多头并行计算

3. **位置编码**
   - 正弦/余弦函数
   - 固定编码（不可学习）
   - 加法融合

---

## 核心组件

### 1. TransformerModel

完整的Transformer模型，包含所有组件。

```python
from src.models.transformer import TransformerModel

model = TransformerModel(
    ts2vec_dim=128,
    manual_dim=27,
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1,
    use_auxiliary=True
)
```

### 2. FeatureFusion

特征融合层，融合TS2Vec embedding和手工特征。

```python
from src.models.transformer import FeatureFusion

fusion = FeatureFusion(
    ts2vec_dim=128,
    manual_dim=27,
    d_model=256,
    dropout=0.1
)
```

### 3. MultiHeadAttention

多头自注意力机制。

```python
from src.models.transformer import MultiHeadAttention

attention = MultiHeadAttention(
    d_model=256,
    num_heads=8,
    dropout=0.1
)
```

### 4. PositionalEncoding

正弦位置编码。

```python
from src.models.transformer import PositionalEncoding

pos_encoder = PositionalEncoding(
    d_model=256,
    max_len=512,
    dropout=0.1
)
```

---

## 使用指南

### 快速开始

```python
import numpy as np
from src.models.transformer import (
    TransformerModel,
    TransformerDataModule,
    TransformerTrainer
)

# 1. 准备数据
train_ts2vec = np.load('data/train_ts2vec_embeddings.npy')
train_features = np.load('data/train_manual_features.npy')
train_prices = np.load('data/train_prices.npy')

# 2. 创建数据模块
data_module = TransformerDataModule(
    train_ts2vec=train_ts2vec,
    train_features=train_features,
    train_prices=train_prices,
    seq_len=64,
    batch_size=32
)

# 3. 创建模型
model = TransformerModel(
    ts2vec_dim=128,
    manual_dim=27,
    d_model=256,
    num_heads=8,
    num_layers=6
)

# 4. 创建训练器
trainer = TransformerTrainer(
    model=model,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    learning_rate=1e-4,
    max_epochs=100
)

# 5. 训练
trainer.train()
```

### 加载预训练模型

```python
import torch
from src.models.transformer import TransformerModel

# 创建模型
model = TransformerModel(
    ts2vec_dim=128,
    manual_dim=27,
    d_model=256,
    num_heads=8,
    num_layers=6
)

# 加载权重
checkpoint = torch.load('models/transformer/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 提取状态向量

```python
from src.models.transformer import StateVectorExtractor

# 创建提取器
extractor = StateVectorExtractor(model, device='cuda')

# 批量提取
states = extractor.extract(test_loader, return_last_only=True)

# 单样本提取
state = extractor.extract_single(ts2vec_emb, manual_features)
```

---

## 配置说明

配置文件位于 `configs/transformer_config.yaml`。

### 模型配置

```yaml
model:
  ts2vec_dim: 128        # TS2Vec embedding维度
  manual_dim: 27         # 手工特征维度
  d_model: 256           # 模型维度
  num_heads: 8           # 注意力头数
  num_layers: 6          # 编码器层数
  d_ff: 1024             # 前馈网络维度
  dropout: 0.1           # Dropout率
```

### 训练配置

```yaml
training:
  learning_rate: 0.0001  # 学习率
  weight_decay: 0.01     # 权重衰减
  warmup_steps: 1000     # Warmup步数
  batch_size: 32         # 批次大小
  max_epochs: 100        # 最大轮数
  grad_clip: 1.0         # 梯度裁剪
  
  # 多任务权重
  regression_weight: 1.0
  classification_weight: 0.5
```

### 数据配置

```yaml
data:
  seq_len: 64            # 序列长度
  pred_horizon: 1        # 预测时间跨度
  stride: 1              # 滑动步长
  return_threshold: 0.001 # 分类阈值
```

---

## 训练流程

### 1. 数据准备

```python
# 加载TS2Vec embeddings
from src.models.ts2vec import EmbeddingExtractor

ts2vec_model = load_ts2vec_model()
extractor = EmbeddingExtractor(ts2vec_model)
embeddings = extractor.extract(data_loader)

# 加载手工特征
from src.features import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()
features = pipeline.transform(raw_data)
```

### 2. 创建数据集

```python
from src.models.transformer import TransformerDataModule

data_module = TransformerDataModule(
    train_ts2vec=train_embeddings,
    train_features=train_features,
    train_prices=train_prices,
    val_ts2vec=val_embeddings,
    val_features=val_features,
    val_prices=val_prices,
    seq_len=64,
    batch_size=32
)
```

### 3. 训练模型

```python
from src.models.transformer import TransformerTrainer

trainer = TransformerTrainer(
    model=model,
    train_loader=data_module.train_dataloader(),
    val_loader=data_module.val_dataloader(),
    learning_rate=1e-4,
    max_epochs=100,
    patience=10
)

trainer.train()
```

### 4. 监控训练

训练过程中会自动记录：
- 训练损失（总损失、回归损失、分类损失）
- 验证损失
- 学习率变化
- 训练时间

所有指标保存在 `models/transformer/training_history.json`。

---

## 评估方法

### 1. 模型评估

```python
from src.models.transformer import TransformerEvaluator

evaluator = TransformerEvaluator(model, device='cuda')

# 评估
metrics = evaluator.evaluate(test_loader)

# 生成报告
report = evaluator.generate_report(metrics)
print(report)
```

### 2. 评估指标

**回归任务**:
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- Direction Accuracy

**分类任务**:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC

### 3. 注意力可视化

```python
# 提取注意力权重
attention_weights = evaluator.extract_attention_weights(
    ts2vec_emb, manual_features, layer_idx=-1
)

# 可视化
evaluator.visualize_attention(
    attention_weights[0],  # 第一个样本
    save_path='attention_heatmap.png'
)
```

---

## API参考

### TransformerModel

```python
class TransformerModel(nn.Module):
    def __init__(
        self,
        ts2vec_dim: int = 128,
        manual_dim: int = 27,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_auxiliary: bool = True
    )
    
    def forward(
        self,
        ts2vec_emb: torch.Tensor,
        manual_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
```

### TransformerTrainer

```python
class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_epochs: int = 100,
        grad_clip: float = 1.0,
        regression_weight: float = 1.0,
        classification_weight: float = 0.5,
        device: str = 'cuda',
        save_dir: str = 'models/transformer',
        patience: int = 10
    )
    
    def train(self)
    def save_checkpoint(self, is_best: bool = False)
    def load_checkpoint(self, checkpoint_path: str)
```

### TransformerEvaluator

```python
class TransformerEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    )
    
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict[str, float]
    
    def extract_attention_weights(
        self,
        ts2vec_emb: torch.Tensor,
        manual_features: torch.Tensor,
        layer_idx: int = -1
    ) -> np.ndarray
    
    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    )
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> str
```

### StateVectorExtractor

```python
class StateVectorExtractor:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    )
    
    def extract(
        self,
        dataloader: DataLoader,
        return_last_only: bool = True
    ) -> np.ndarray
    
    def extract_single(
        self,
        ts2vec_emb: np.ndarray,
        manual_features: np.ndarray
    ) -> np.ndarray
    
    def save_states(self, states: np.ndarray, save_path: str)
    def load_states(self, load_path: str) -> np.ndarray
```

---

## 最佳实践

### 1. 数据准备

- 确保TS2Vec embeddings质量良好
- 手工特征需要正确归一化
- 价格序列用于生成标签

### 2. 超参数调优

关键超参数：
- `learning_rate`: 建议从1e-4开始
- `warmup_steps`: 通常设为总步数的5-10%
- `regression_weight` vs `classification_weight`: 根据任务重要性调整
- `dropout`: 0.1-0.2之间

### 3. 训练技巧

- 使用Warmup + 余弦退火学习率
- 应用梯度裁剪防止梯度爆炸
- 使用早停避免过拟合
- 监控验证集性能

### 4. 评估建议

- 在多个市场状态下评估
- 关注方向准确率（比MSE更重要）
- 可视化注意力权重理解模型行为
- 与基准模型对比

---

## 常见问题

### Q1: 如何选择序列长度？

A: 序列长度取决于数据频率和预测目标：
- 5分钟数据: 64步 ≈ 5.3小时
- 建议在32-128之间
- 过长会增加计算成本

### Q2: 多任务权重如何设置？

A: 根据任务重要性：
- 回归任务权重: 1.0（基准）
- 分类任务权重: 0.3-0.7
- 可以通过验证集调优

### Q3: 如何处理类别不平衡？

A: 几种方法：
- 调整`return_threshold`
- 使用加权损失函数
- 数据重采样

### Q4: 训练需要多长时间？

A: 取决于数据量和硬件：
- 1000个样本: ~5分钟/epoch (GPU)
- 10000个样本: ~30分钟/epoch (GPU)
- 建议使用GPU加速

---

## 更新日志

### v1.0.0 (2025-11-20)
- 初始版本发布
- 实现完整的Transformer架构
- 支持多任务学习
- 提供训练和评估工具

---

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
3. Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)

---

**文档版本**: v1.0.0  
**最后更新**: 2025-11-20  
**维护者**: AI Trader Team