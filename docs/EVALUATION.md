# 评估与验证体系文档

## 概述

评估与验证体系提供全面的模型评估工具，包括Walk-Forward验证、特征重要性分析、过拟合检测、市场状态分析、稳定性测试、压力测试和基准对比等功能。

## 模块结构

```
src/evaluation/
├── __init__.py                 # 模块初始化
├── walk_forward.py            # Walk-Forward验证
├── feature_importance.py      # 特征重要性分析
├── overfitting_detector.py    # 过拟合检测
├── market_regime.py           # 市场状态识别
├── evaluation_tools.py        # 评估工具集
└── report_generator.py        # 报告生成器
```

## 1. Walk-Forward验证

### 原理

Walk-Forward验证是一种时间序列交叉验证方法，通过滚动窗口评估模型的时间稳定性。

### 使用示例

```python
from src.evaluation import WalkForwardValidator

# 创建验证器
validator = WalkForwardValidator(
    train_window=24,  # 训练窗口24个月
    val_window=6,     # 验证窗口6个月
    test_window=6,    # 测试窗口6个月
    step_size=6       # 步长6个月
)

# 定义训练和评估函数
def train_func(train_data, val_data, **kwargs):
    # 训练模型
    model = YourModel()
    model.fit(train_data, val_data)
    return model

def eval_func(model, test_data, **kwargs):
    # 评估模型
    predictions = model.predict(test_data)
    metrics = calculate_metrics(predictions, test_data)
    return metrics

# 执行验证
results = validator.validate(
    data=your_data,
    train_func=train_func,
    eval_func=eval_func
)

# 打印摘要
validator.print_summary()

# 保存结果
validator.save_results('results/walk_forward.json')
```

### 关键参数

- `train_window`: 训练窗口大小（月）
- `val_window`: 验证窗口大小（月）
- `test_window`: 测试窗口大小（月）
- `step_size`: 滚动步长（月）
- `min_train_size`: 最小训练集大小

### 输出指标

- 各fold的训练/验证/测试指标
- 统计量（均值、标准差、最小值、最大值、中位数、CV）
- 稳定性评分

## 2. 特征重要性分析

### 2.1 置换重要性

通过随机打乱特征值来评估特征重要性。

```python
from src.evaluation import PermutationImportance

# 创建分析器
perm_importance = PermutationImportance(
    model=trained_model,
    eval_func=your_eval_func,
    n_repeats=100,
    random_state=42
)

# 执行分析
importances = perm_importance.analyze(X, y)

# 打印摘要
perm_importance.print_summary(top_n=20)

# 绘制重要性图
perm_importance.plot_importances(top_n=20, save_path='results/importance.png')
```

### 2.2 消融实验

通过逐步移除特征组来评估其重要性。

```python
from src.evaluation import AblationStudy

# 创建消融实验
ablation = AblationStudy(
    train_func=your_train_func,
    eval_func=your_eval_func
)

# 定义特征组
feature_groups = {
    'price_features': ['ret_1', 'ret_5', 'ret_20'],
    'volatility_features': ['ATR14_norm', 'vol_20'],
    'technical_features': ['EMA20', 'MACD']
}

# 执行消融实验
results = ablation.run_ablation(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_groups
)

# 打印摘要
ablation.print_summary()

# 绘制结果
ablation.plot_results(save_path='results/ablation.png')
```

## 3. 过拟合检测

### 检测方法

1. **性能差距检测**: 训练集和验证集性能差距
2. **验证集退化检测**: 验证集性能下降
3. **方差增大检测**: 验证集方差增大
4. **学习曲线分析**: 训练和验证曲线趋势

### 使用示例

```python
from src.evaluation import OverfittingDetector

# 创建检测器
detector = OverfittingDetector(
    train_metrics={'loss': train_loss_history, 'accuracy': train_acc_history},
    val_metrics={'loss': val_loss_history, 'accuracy': val_acc_history}
)

# 执行检测
results = detector.detect_all()

# 打印报告
detector.print_report()

# 绘制学习曲线
detector.plot_learning_curves(save_path='results/learning_curves.png')
```

### 判断标准

- **性能差距**: 超过10%认为过拟合
- **验证集退化**: 超过5%认为退化
- **方差增大**: 超过50%认为异常
- **学习曲线**: 训练和验证趋势差异超过50%认为异常

## 4. 市场状态识别

### 状态类型

1. **牛市(Bull)**: 明显上升趋势
2. **熊市(Bear)**: 明显下降趋势
3. **震荡市(Sideways)**: 无明显趋势
4. **高波动(High Volatility)**: 波动率异常高

### 使用示例

```python
from src.evaluation import MarketRegimeDetector, RegimeBasedEvaluator

# 创建检测器
detector = MarketRegimeDetector(
    trend_window=20,
    volatility_window=20,
    trend_threshold=0.02,
    volatility_threshold=0.03
)

# 检测市场状态
regimes = detector.detect_regimes(data, price_column='close')

# 打印摘要
detector.print_summary()

# 绘制市场状态
detector.plot_regimes(data, save_path='results/regimes.png')

# 按状态评估
evaluator = RegimeBasedEvaluator(detector)
results = evaluator.evaluate_by_regime(
    data, predictions, actuals, eval_func
)

# 打印对比
evaluator.print_comparison()
```

## 5. 稳定性测试

### 多Seed测试

使用多个随机种子训练模型，评估结果稳定性。

```python
from src.evaluation import MultiSeedStabilityTest

# 创建测试
stability_test = MultiSeedStabilityTest(
    train_func=your_train_func,
    eval_func=your_eval_func,
    n_seeds=10,
    base_seed=42
)

# 执行测试
statistics = stability_test.run_test(train_data, test_data)

# 打印摘要
stability_test.print_summary()

# 绘制分布
stability_test.plot_distributions(save_path='results/stability.png')
```

### 稳定性标准

- **变异系数(CV) < 0.1**: 认为稳定
- **CV 0.1-0.3**: 中等稳定
- **CV > 0.3**: 不稳定

## 6. 压力测试

### 测试场景

1. **历史极端事件**: 2008金融危机、2020疫情等
2. **高波动期**: 波动率超过90分位
3. **大幅下跌期**: 收益率低于5分位
4. **连续下跌期**: 连续多日下跌

### 使用示例

```python
from src.evaluation import StressTest

# 创建压力测试
stress_test = StressTest()

# 定义极端事件
extreme_events = [
    ('2008金融危机', '2008-09-01', '2009-03-31'),
    ('2020新冠疫情', '2020-02-01', '2020-04-30')
]

# 执行测试
results = stress_test.run_stress_tests(
    model=trained_model,
    data=full_data,
    eval_func=your_eval_func,
    extreme_events=extreme_events
)

# 打印摘要
stress_test.print_summary()
```

## 7. 基准策略对比

### 基准策略

1. **Buy & Hold**: 买入持有
2. **MA Crossover**: 移动平均交叉
3. **Momentum**: 动量策略

### 使用示例

```python
from src.evaluation import BenchmarkComparison

# 创建对比
benchmark = BenchmarkComparison()

# 添加AI策略
benchmark.add_strategy('AI Strategy', ai_returns, ai_trades)

# 添加基准策略
buy_hold_returns = benchmark.create_buy_and_hold(prices)
benchmark.add_strategy('Buy & Hold', buy_hold_returns)

ma_returns = benchmark.create_ma_crossover(prices, fast_window=10, slow_window=30)
benchmark.add_strategy('MA Crossover', ma_returns)

momentum_returns = benchmark.create_momentum(prices, lookback=20)
benchmark.add_strategy('Momentum', momentum_returns)

# 打印对比
benchmark.print_comparison()

# 绘制对比图
benchmark.plot_comparison(save_path='results/benchmark.png')
```

## 8. 评估报告生成

### 综合报告

```python
from src.evaluation import create_comprehensive_report

# 创建综合报告
report = create_comprehensive_report(
    walk_forward_results=wf_results,
    feature_importance=importance_results,
    ablation_results=ablation_results,
    overfitting_detection=overfitting_results,
    stability_test=stability_results,
    regime_analysis=regime_results,
    stress_test=stress_results,
    benchmark_comparison=benchmark_results,
    output_dir='results/evaluation'
)
```

### 报告格式

- **Markdown**: 适合版本控制和文档
- **HTML**: 适合浏览器查看，包含样式
- **JSON**: 适合程序处理

## 配置文件

配置文件位于 `configs/evaluation_config.yaml`，包含所有评估参数的默认值。

### 主要配置项

```yaml
# Walk-Forward验证
walk_forward:
  train_window: 24
  val_window: 6
  test_window: 6
  step_size: 6

# 特征重要性
feature_importance:
  n_repeats: 100
  top_n: 20

# 过拟合检测
overfitting_detection:
  performance_gap_threshold: 0.10
  degradation_threshold: 0.05

# 稳定性测试
stability_test:
  n_seeds: 10
  cv_threshold: 0.10

# 市场状态
market_regime:
  trend_window: 20
  volatility_window: 20
```

## 最佳实践

### 1. 评估流程

```python
# 1. Walk-Forward验证
validator = WalkForwardValidator(...)
wf_results = validator.validate(...)

# 2. 特征重要性分析
perm_importance = PermutationImportance(...)
importance_results = perm_importance.analyze(...)

# 3. 过拟合检测
detector = OverfittingDetector(...)
overfitting_results = detector.detect_all()

# 4. 稳定性测试
stability_test = MultiSeedStabilityTest(...)
stability_results = stability_test.run_test(...)

# 5. 市场状态分析
regime_detector = MarketRegimeDetector(...)
regime_results = regime_detector.detect_regimes(...)

# 6. 压力测试
stress_test = StressTest()
stress_results = stress_test.run_stress_tests(...)

# 7. 基准对比
benchmark = BenchmarkComparison()
benchmark_results = benchmark.print_comparison()

# 8. 生成报告
report = create_comprehensive_report(...)
```

### 2. 注意事项

1. **时间顺序**: 确保数据按时间顺序排列
2. **数据泄露**: 避免使用未来信息
3. **样本量**: 确保每个fold有足够的样本
4. **计算资源**: 某些评估（如多Seed测试）需要较多计算资源
5. **统计显著性**: 注意p值和置信区间

### 3. 性能优化

1. **并行计算**: 使用多进程加速评估
2. **缓存结果**: 保存中间结果避免重复计算
3. **采样**: 对大数据集进行采样
4. **GPU加速**: 利用GPU加速模型推理

## 故障排查

### 常见问题

1. **内存不足**
   - 减少n_repeats
   - 使用数据采样
   - 分批处理

2. **计算时间过长**
   - 减少fold数量
   - 减少seed数量
   - 使用并行计算

3. **结果不稳定**
   - 增加n_repeats
   - 增加n_seeds
   - 检查数据质量

## 参考资料

1. Walk-Forward Analysis: [链接]
2. Permutation Importance: Breiman (2001)
3. Market Regime Detection: [链接]
4. Cross-Validation for Time Series: [链接]

## 更新日志

### v1.0.0 (2025-11-20)
- 初始版本
- 实现Walk-Forward验证
- 实现特征重要性分析
- 实现过拟合检测
- 实现市场状态识别
- 实现稳定性测试
- 实现压力测试
- 实现基准对比
- 实现报告生成