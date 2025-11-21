# PPO强化学习模型

## 概述

本模块实现了完整的PPO (Proximal Policy Optimization)强化学习算法，专为交易策略训练设计。采用面向对象设计，所有组件整合在统一的模块结构中。

## 模块结构

```
src/models/ppo/
├── __init__.py          # 模块导出
├── model.py            # PPO模型（Actor-Critic网络、经验缓冲区）
├── trainer.py          # PPO训练器
├── metrics.py          # 性能评估指标
├── environment/        # 交易环境
│   ├── __init__.py
│   ├── action_space.py
│   ├── reward_function.py
│   ├── state_space.py
│   └── trading_env.py
└── README.md           # 本文档
```

## 核心类

### 1. PPOModel (`model.py`)

统一的PPO模型接口，整合了Actor-Critic网络和经验缓冲区。

**主要组件**:
- `ActorNetwork`: 策略网络，输出混合动作空间
- `CriticNetwork`: 价值网络，估计状态价值
- `ExperienceBuffer`: 经验缓冲区，存储和处理轨迹数据
- `PPOModel`: 统一模型接口

**使用示例**:
```python
from src.models.ppo import PPOModel

# 创建模型
model = PPOModel(
    state_dim=263,
    hidden_dim=512,
    lr_actor=1e-4,
    lr_critic=3e-4,
    device='cuda'
)

# 选择动作
action, log_prob, value = model.select_action(state)

# 保存/加载模型
model.save('model.pt')
model.load('model.pt')

# 获取模型信息
info = model.get_model_info()
```

### 2. PPOTrainer (`trainer.py`)

完整的PPO训练流程实现。

**主要功能**:
- 经验收集
- GAE优势函数计算
- PPO损失计算（策略损失、价值损失、熵正则化）
- 策略更新
- 训练监控和日志
- 模型保存

**使用示例**:
```python
from src.models.ppo import PPOModel, PPOTrainer

# 创建训练器
trainer = PPOTrainer(
    model=model,
    env=env,
    config={
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10
    }
)

# 开始训练
trainer.train(
    n_iterations=100,
    save_dir='models/ppo'
)

# 评估策略
eval_stats = trainer.evaluate(n_episodes=50, verbose=True)
```

### 3. TrainingMetrics (`metrics.py`)

性能评估和可视化工具。

**主要功能**:
- 风险调整收益指标（夏普比率、索提诺比率、卡玛比率）
- 交易统计（胜率、盈亏比）
- 训练曲线可视化
- 性能报告生成

**使用示例**:
```python
from src.models.ppo import TrainingMetrics

# 计算夏普比率
sharpe = TrainingMetrics.compute_sharpe_ratio(returns)

# 生成性能报告
report = TrainingMetrics.generate_performance_report(
    episode_rewards, episode_lengths
)

# 绘制训练曲线
TrainingMetrics.plot_training_curves(
    history=trainer.history,
    save_path='curves.png'
)
```

## 快速开始

### 完整训练示例

```python
from src.models.ppo import PPOModel, PPOTrainer, TradingEnvironment

# 1. 创建环境
env = TradingEnvironment(
    data=market_data,
    transformer_states=transformer_features,
    initial_balance=100000.0
)

# 2. 创建模型
model = PPOModel(
    state_dim=263,
    hidden_dim=512,
    device='cuda'
)

# 3. 创建训练器
trainer = PPOTrainer(model=model, env=env)

# 4. 训练
trainer.train(n_iterations=100, save_dir='models/ppo')

# 5. 评估
eval_stats = trainer.evaluate(n_episodes=50, verbose=True)
```

## 配置参数

### PPO超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| gamma | 0.99 | 折扣因子 |
| gae_lambda | 0.95 | GAE λ参数 |
| clip_epsilon | 0.2 | PPO裁剪范围 |
| value_clip | 0.2 | 价值函数裁剪 |
| entropy_coef | 0.01 | 熵系数 |
| value_coef | 0.5 | 价值损失系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_steps | 2048 | 每次迭代收集步数 |
| batch_size | 64 | 批次大小 |
| n_epochs | 10 | 每次更新的epoch数 |
| buffer_capacity | 10000 | 缓冲区容量 |
| save_interval | 10 | 保存间隔 |
| log_interval | 1 | 日志间隔 |

## 性能指标

### 风险调整收益
- **夏普比率**: 单位风险的超额收益（良好值 > 1.0）
- **索提诺比率**: 只考虑下行风险（良好值 > 1.5）
- **卡玛比率**: 年化收益/最大回撤（良好值 > 1.0）
- **最大回撤**: 峰值到谷底的最大跌幅（良好值 < 20%）

### 交易统计
- **胜率**: 盈利交易占比（良好值 > 50%）
- **盈亏比**: 总盈利/总亏损（良好值 > 1.5）

## 面向对象设计优势

1. **模块化**: 所有PPO相关代码集中管理
2. **封装性**: 清晰的职责分离和接口设计
3. **可复用性**: 组件可独立使用
4. **可测试性**: 便于单元测试和集成测试
5. **可扩展性**: 易于添加新功能和改进

## 示例脚本

完整的训练示例请参考：
- `examples/ppo_training_demo.py` - 完整训练流程演示

## 文档

详细文档请参考：
- `docs/PPO_TRAINING_SYSTEM.md` - 系统架构和使用指南

## 依赖

- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- Pandas >= 1.2.0

## 许可

本项目采用MIT许可证。