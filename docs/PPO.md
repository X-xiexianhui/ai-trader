# PPO强化学习模块使用文档

## 目录
- [概述](#概述)
- [模块架构](#模块架构)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [配置说明](#配置说明)
- [训练流程](#训练流程)
- [评估和测试](#评估和测试)
- [常见问题](#常见问题)

## 概述

PPO（Proximal Policy Optimization）强化学习模块是AI交易系统的决策执行层，负责根据市场状态做出交易决策。该模块基于Actor-Critic架构，使用PPO算法进行策略优化。

### 主要特性

- **混合动作空间**：支持离散动作（方向）和连续动作（仓位、止损止盈）
- **GAE优势估计**：使用广义优势估计提高训练稳定性
- **完整的交易环境**：模拟真实交易场景，包括手续费、滑点等
- **灵活的奖励函数**：多组件奖励设计，平衡盈利、风险和稳定性
- **全面的评估工具**：提供详细的性能指标和可视化

### 系统架构

```
TS2Vec → Transformer → PPO → 交易执行
  ↓          ↓          ↓
特征提取   状态编码   决策制定
```

## 模块架构

### 核心组件

1. **TradingEnvironment** (`environment.py`)
   - 交易环境模拟
   - 符合Gymnasium接口
   - 支持多种交易操作

2. **ActorCritic** (`policy.py`)
   - 策略网络（Actor）
   - 价值网络（Critic）
   - 混合动作空间支持

3. **RolloutBuffer** (`buffer.py`)
   - 经验存储
   - GAE计算
   - 批量数据生成

4. **PPOTrainer** (`trainer.py`)
   - 训练流程管理
   - 损失函数计算
   - 模型更新

5. **PPOEvaluator** (`evaluator.py`)
   - 性能评估
   - 指标计算
   - 报告生成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

首先需要训练Transformer模型以生成状态向量：

```python
# 训练Transformer模型
from src.models.transformer import TransformerTrainer

# ... 训练代码 ...
```

### 3. 训练PPO模型

```python
import torch
import yaml
from src.models.ppo import TradingEnvironment, ActorCritic, PPOTrainer
from src.models.transformer import TransformerModel

# 加载配置
with open('configs/ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载Transformer模型
transformer = TransformerModel.load('models/transformer/best_model.pt')
transformer.eval()

# 生成状态向量
with torch.no_grad():
    states = transformer.encode(data)  # (num_steps, 256)

# 创建交易环境
env = TradingEnvironment(
    states=states.numpy(),
    prices=prices,
    initial_balance=config['environment']['initial_balance'],
    max_position_size=config['environment']['max_position_size']
)

# 创建Actor-Critic模型
model = ActorCritic(
    state_dim=config['model']['state_dim'],
    hidden_dim=config['model']['hidden_dim']
)

# 创建训练器
trainer = PPOTrainer(
    env=env,
    model=model,
    config=config['training'],
    save_dir=config['save']['model_dir'],
    device=config['training']['device']
)

# 开始训练
trainer.train(
    total_timesteps=config['training']['total_timesteps'],
    log_interval=config['training']['log_interval'],
    save_interval=config['training']['save_interval']
)
```

### 4. 评估模型

```python
from src.models.ppo import PPOEvaluator

# 加载训练好的模型
model = ActorCritic.load('models/ppo/final_model.pt')

# 创建评估器
evaluator = PPOEvaluator(
    model=model,
    env=env,
    device='cuda'
)

# 评估性能
metrics = evaluator.evaluate(num_episodes=20, deterministic=True)

# 分析动作分布
action_analysis = evaluator.analyze_actions(num_episodes=10)

# 生成报告
report = evaluator.generate_report(
    metrics=metrics,
    action_analysis=action_analysis,
    save_path='reports/ppo_evaluation.txt'
)

print(report)
```

## 详细使用

### 交易环境

#### 创建环境

```python
from src.models.ppo import TradingEnvironment

env = TradingEnvironment(
    states=states,              # 状态向量 (num_steps, 256)
    prices=prices,              # 价格序列 (num_steps,)
    initial_balance=100000.0,   # 初始资金
    max_position_size=0.5,      # 最大仓位比例
    commission_rate=0.0003,     # 手续费率
    slippage=0.0001,           # 滑点
    max_steps=1000             # 最大步数
)
```

#### 环境接口

```python
# 重置环境
obs, info = env.reset()

# 执行动作
action = {
    'direction': 1,              # 0=平仓, 1=做多, 2=做空
    'position_size': [0.3],      # 仓位大小 [0, 1]
    'stop_loss': [0.02],         # 止损比例 [0.001, 0.05]
    'take_profit': [0.05]        # 止盈比例 [0.002, 0.10]
}

obs, reward, terminated, truncated, info = env.step(action)
```

#### 观察空间

观察空间维度：263
- 状态向量：256维（来自Transformer）
- 持仓信息：4维
  - 是否持仓（0或1）
  - 持仓方向（-1=空，0=无，1=多）
  - 持仓大小（归一化）
  - 持仓时间（归一化）
- 风险参数：3维
  - 当前止损比例
  - 当前止盈比例
  - 账户余额变化率

### 策略网络

#### 网络架构

```python
from src.models.ppo import ActorCritic

model = ActorCritic(
    state_dim=263,
    hidden_dim=512,
    dropout=0.1,
    share_features=False  # 是否共享特征层
)

# 查看参数量
print(f"参数量: {model.count_parameters():,}")
```

#### 动作采样

```python
import torch

state = torch.randn(1, 263)

# 随机策略
action, log_prob, entropy, value = model(state, deterministic=False)

# 确定性策略
action, log_prob, entropy, value = model(state, deterministic=True)
```

#### 动作评估

```python
# 评估给定状态-动作对
log_prob, entropy, value = model.evaluate(state, action)
```

### 经验缓冲区

#### 创建缓冲区

```python
from src.models.ppo import RolloutBuffer

buffer = RolloutBuffer(
    buffer_size=2048,
    state_dim=263,
    gamma=0.99,
    gae_lambda=0.95
)
```

#### 添加经验

```python
buffer.add(
    state=state,
    action=action,
    reward=reward,
    value=value,
    log_prob=log_prob,
    done=done
)
```

#### 完成轨迹

```python
# 计算GAE优势和回报
buffer.finish_path(last_value=0.0)
```

#### 获取数据

```python
# 获取所有经验（自动标准化优势）
data = buffer.get()

# 重置缓冲区
buffer.reset()
```

### 训练器

#### 自定义训练配置

```python
config = {
    'n_steps': 2048,
    'n_epochs': 10,
    'mini_batch_size': 256,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_lr': 1e-4,
    'value_lr': 3e-4
}

trainer = PPOTrainer(env, model, config)
```

#### 训练控制

```python
# 开始训练
trainer.train(
    total_timesteps=1000000,
    log_interval=10,        # 每10次更新记录日志
    save_interval=100,      # 每100次更新保存模型
    eval_interval=50        # 每50次更新评估模型
)
```

#### 保存和加载

```python
# 保存检查点
trainer.save_checkpoint('checkpoint_1000.pt')

# 加载检查点
trainer.load_checkpoint('checkpoint_1000.pt')

# 保存训练历史
trainer.save_training_history()
```

### 评估器

#### 性能评估

```python
from src.models.ppo import PPOEvaluator

evaluator = PPOEvaluator(model, env)

# 评估模型
metrics = evaluator.evaluate(
    num_episodes=20,
    deterministic=True,
    render=False
)

print(f"平均奖励: {metrics['mean_episode_reward']:.4f}")
print(f"平均回报: {metrics['mean_return']:.4%}")
print(f"夏普率: {metrics['mean_sharpe_ratio']:.4f}")
print(f"最大回撤: {metrics['mean_max_drawdown']:.4%}")
print(f"胜率: {metrics['mean_win_rate']:.4%}")
```

#### 动作分析

```python
# 分析动作分布
action_analysis = evaluator.analyze_actions(num_episodes=10)

print("方向分布:")
print(f"  平仓: {action_analysis['direction_distribution']['close']:.2%}")
print(f"  做多: {action_analysis['direction_distribution']['long']:.2%}")
print(f"  做空: {action_analysis['direction_distribution']['short']:.2%}")
```

#### 生成报告

```python
# 生成评估报告
report = evaluator.generate_report(
    metrics=metrics,
    action_analysis=action_analysis,
    save_path='reports/evaluation.txt'
)
```

#### 绘制性能曲线

```python
# 绘制训练曲线
evaluator.plot_performance(
    metrics_history=trainer.training_history,
    save_path='reports/training_curves.png'
)
```

## 配置说明

### 配置文件结构

配置文件位于 `configs/ppo_config.yaml`，包含以下部分：

1. **模型配置** (`model`)
   - `state_dim`: 状态维度
   - `hidden_dim`: 隐藏层维度
   - `dropout`: Dropout率
   - `share_features`: 是否共享特征层

2. **环境配置** (`environment`)
   - `initial_balance`: 初始资金
   - `max_position_size`: 最大仓位
   - `commission_rate`: 手续费率
   - `slippage`: 滑点
   - `reward_weights`: 奖励权重

3. **训练配置** (`training`)
   - `total_timesteps`: 总训练步数
   - `n_steps`: Rollout步数
   - `n_epochs`: 更新轮数
   - `mini_batch_size`: 小批量大小
   - PPO算法参数
   - 优化器参数

4. **评估配置** (`evaluation`)
   - `num_episodes`: 评估episode数
   - `deterministic`: 是否确定性策略

5. **数据配置** (`data`)
   - 模型路径
   - 数据路径

### 关键超参数

#### PPO算法参数

- **gamma** (0.95-0.99): 折扣因子，控制未来奖励的重要性
- **gae_lambda** (0.90-0.98): GAE参数，平衡偏差和方差
- **clip_range** (0.1-0.3): PPO裁剪范围，控制策略更新幅度
- **ent_coef** (0.001-0.01): 熵系数，鼓励探索
- **vf_coef** (0.5-1.0): 价值函数损失系数

#### 学习率

- **policy_lr** (1e-5 - 1e-3): 策略网络学习率
- **value_lr** (3e-5 - 3e-3): 价值网络学习率

建议：价值网络学习率通常是策略网络的2-3倍

#### 训练参数

- **n_steps** (1024-4096): 每次rollout的步数
- **n_epochs** (3-15): 每次更新的轮数
- **mini_batch_size** (64-512): 小批量大小

## 训练流程

### 完整训练流程

```python
import yaml
import torch
from pathlib import Path

# 1. 加载配置
with open('configs/ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. 准备数据
from src.data import DataManager
from src.models.transformer import TransformerModel

data_manager = DataManager(config['data'])
train_data = data_manager.load_train_data()

# 3. 加载Transformer模型
transformer = TransformerModel.load(
    config['data']['transformer_model_path']
)

# 4. 生成状态向量
states = transformer.encode(train_data['features'])
prices = train_data['prices']

# 5. 创建环境
from src.models.ppo import TradingEnvironment

env = TradingEnvironment(
    states=states.numpy(),
    prices=prices,
    **config['environment']
)

# 6. 创建模型
from src.models.ppo import ActorCritic

model = ActorCritic(**config['model'])

# 7. 创建训练器
from src.models.ppo import PPOTrainer

trainer = PPOTrainer(
    env=env,
    model=model,
    config=config['training'],
    save_dir=config['save']['model_dir'],
    device=config['training']['device']
)

# 8. 开始训练
trainer.train(
    total_timesteps=config['training']['total_timesteps'],
    log_interval=config['training']['log_interval'],
    save_interval=config['training']['save_interval']
)

# 9. 保存最终模型
trainer.save_checkpoint('final_model.pt')
```

### 训练监控

训练过程中会输出以下信息：

```
Update 10/500 | Timesteps: 20480 | Mean Reward: 0.0234 | Policy Loss: 0.0123 | Value Loss: 0.0456
Update 20/500 | Timesteps: 40960 | Mean Reward: 0.0345 | Policy Loss: 0.0098 | Value Loss: 0.0389
...
```

### 训练技巧

1. **从小规模开始**
   - 先用少量数据测试
   - 确保代码正常运行
   - 逐步增加数据量

2. **监控关键指标**
   - 平均奖励趋势
   - 策略损失和价值损失
   - KL散度（检测策略变化）
   - 裁剪比例（检测更新幅度）

3. **调整超参数**
   - 如果训练不稳定，减小学习率或clip_range
   - 如果收敛太慢，增加学习率或n_epochs
   - 如果过拟合，增加熵系数或dropout

4. **使用学习率调度**
   - 训练后期降低学习率
   - 有助于收敛到更好的解

## 评估和测试

### 评估指标

1. **收益指标**
   - 平均回报率
   - 累计收益
   - 单笔交易收益

2. **风险指标**
   - 最大回撤
   - 夏普率
   - 波动率

3. **交易指标**
   - 胜率
   - 盈亏比
   - 交易频率

### 回测流程

```python
# 1. 加载测试数据
test_data = data_manager.load_test_data()
test_states = transformer.encode(test_data['features'])
test_prices = test_data['prices']

# 2. 创建测试环境
test_env = TradingEnvironment(
    states=test_states.numpy(),
    prices=test_prices,
    **config['environment']
)

# 3. 加载最佳模型
model = ActorCritic.load('models/ppo/best_model.pt')

# 4. 评估
evaluator = PPOEvaluator(model, test_env)
metrics = evaluator.evaluate(num_episodes=50, deterministic=True)

# 5. 生成报告
action_analysis = evaluator.analyze_actions(num_episodes=20)
report = evaluator.generate_report(metrics, action_analysis)
print(report)
```

## 常见问题

### Q1: 训练不收敛怎么办？

**A:** 尝试以下方法：
1. 降低学习率（policy_lr和value_lr）
2. 减小clip_range（如0.1）
3. 增加n_steps（如4096）
4. 检查奖励函数设计
5. 确保数据质量

### Q2: 模型过拟合怎么办？

**A:** 
1. 增加dropout率
2. 使用L2正则化
3. 增加训练数据
4. 减少模型复杂度
5. 使用早停

### Q3: 训练速度太慢怎么办？

**A:**
1. 使用GPU训练（设置device='cuda'）
2. 减小mini_batch_size
3. 减少n_epochs
4. 使用更小的hidden_dim
5. 并行环境（num_envs > 1）

### Q4: 如何选择合适的超参数？

**A:**
1. 从推荐值开始
2. 使用网格搜索或贝叶斯优化
3. 监控训练曲线
4. 参考相关论文
5. 进行消融实验

### Q5: 如何处理不平衡的动作分布？

**A:**
1. 调整奖励函数
2. 增加熵系数
3. 使用动作平衡技术
4. 检查环境设计
5. 分析数据分布

### Q6: 如何提高交易策略的稳定性？

**A:**
1. 增加stability_reward权重
2. 使用更保守的止损止盈
3. 限制最大仓位
4. 增加风险惩罚
5. 使用集成方法

## 参考资料

- [PPO论文](https://arxiv.org/abs/1707.06347)
- [Gymnasium文档](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## 更新日志

- **v1.0.0** (2025-11-20)
  - 初始版本发布
  - 实现完整的PPO训练流程
  - 支持混合动作空间
  - 提供全面的评估工具