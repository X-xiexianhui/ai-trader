# 特征模块重组总结

## 概述
根据用户反馈，将特征验证和特征重要性分析功能按照其实际使用时机进行了模块重组。

## 重组决策

### 1. 特征验证 (任务1.4 / 6.5)
**位置**: `src/features/feature_validator.py`
**原因**: 特征验证应该在**特征工程阶段**进行，用于评估特征质量
**时机**: 在特征计算和归一化之后，模型训练之前

包含功能：
- 单特征信息量测试 (1.4.1 / 6.5.1)
- 置换重要性测试 (1.4.2 / 6.5.2)
- 特征相关性检测 (1.4.3 / 6.5.3)
- VIF多重共线性检测 (1.4.4 / 6.5.4)

### 2. 特征重要性分析 (任务6.2)
**位置**: `src/evaluation/ablation_study.py`
**原因**: 消融实验需要在**完整模型训练后**进行，用于评估特征组对最终交易性能的贡献
**时机**: 在PPO模型训练完成后，评估阶段

包含功能：
- 消融实验框架 (6.2.1)
- 特征组贡献度分析 (6.2.2)

## 文件变更

### 移动的文件
1. `src/evaluation/feature_validator.py` → `src/features/feature_validator.py`

### 保持不变的文件
1. `src/evaluation/ablation_study.py` (保持在evaluation模块)

### 更新的文件
1. `src/features/__init__.py` - 添加FeatureValidator导出
2. `src/evaluation/__init__.py` - 保持AblationStudy和FeatureGroupAnalyzer导出
3. `tests/test_module6.py` - 更新导入语句
4. `task.md` - 更新任务6.5的模块归属为"特征层"，保持6.2为"评估层"

## 使用示例

### 特征验证（特征工程阶段）
```python
from src.features import FeatureValidator

# 在特征计算后立即验证
validator = FeatureValidator()

# 1. 单特征信息量测试
info_results = validator.test_single_feature_information(X, y)

# 2. 特征相关性检测
corr_matrix, high_corr_pairs = validator.test_feature_correlation(X)

# 3. VIF多重共线性检测
vif_results = validator.test_vif_multicollinearity(X)

# 4. 生成验证报告
validator.generate_validation_report()
```

### 特征重要性分析（模型评估阶段）
```python
from src.evaluation import AblationStudy, FeatureGroupAnalyzer

# 在PPO模型训练完成后进行消融实验
feature_groups = {
    'price_features': ['ret_1', 'ret_5', 'ret_20'],
    'volatility_features': ['ATR14_norm', 'vol_20'],
    'technical_features': ['EMA20', 'MACD']
}

ablation = AblationStudy(feature_groups)
results = ablation.run_ablation(X_train, y_train, X_val, y_val, train_func, eval_func)

# 分析特征组贡献度
analyzer = FeatureGroupAnalyzer(results)
contributions = analyzer.analyze_absolute_contribution()
analyzer.plot_contribution_heatmap()
```

## 工作流程

```
数据清洗
    ↓
特征计算
    ↓
特征归一化
    ↓
【特征验证】← src/features/feature_validator.py
    ↓
TS2Vec训练
    ↓
Transformer训练
    ↓
PPO训练
    ↓
【特征重要性分析】← src/evaluation/ablation_study.py
    ↓
最终评估
```

## 关键区别

| 方面 | 特征验证 | 特征重要性分析 |
|------|---------|---------------|
| **时机** | 特征工程阶段 | 模型训练后 |
| **目的** | 评估特征质量 | 评估特征对最终性能的贡献 |
| **依赖** | 仅需要特征和标签 | 需要完整训练好的模型 |
| **模块** | src/features | src/evaluation |
| **任务** | 1.4 / 6.5 | 6.2 |

## 依赖关系更新

### task.md中的依赖调整
- 任务6.5.1-6.5.4: 依赖1.3.3（特征归一化）
- 任务6.2.1-6.2.2: 依赖4.4.4（PPO模型训练完成）

## 测试验证

测试文件 `tests/test_module6.py` 已更新导入：
- 从 `src.features` 导入: 无（特征验证测试在test_features.py中）
- 从 `src.evaluation` 导入: AblationStudy, FeatureGroupAnalyzer

## 注意事项

1. **特征验证**应该在每次特征工程迭代时运行，帮助识别和移除低质量特征
2. **特征重要性分析**应该在模型训练完成后运行，评估不同特征组对交易性能的实际贡献
3. 两者互补但不重复：特征验证关注特征本身的质量，特征重要性分析关注特征对模型的贡献

## 完成日期
2025-11-21

## 相关文档
- [任务拆分文档](../task.md)
- [设计文档](../design_document.md)
- [模块7完成总结](./MODULE_7_COMPLETION_SUMMARY.md)