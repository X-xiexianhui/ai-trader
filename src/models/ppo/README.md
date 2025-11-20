# PPO强化学习模块

## 模块概述

PPO（Proximal Policy Optimization）强化学习模块是AI交易系统的决策执行层，负责根据市场状态做出交易决策。该模块基于Actor-Critic架构，使用PPO算法进行策略优化。

## 模块结构

```
src/models/ppo/
├── __init__.py          # 模块初始化
├── environment.py       # 交易环境实现
├── policy.py           # 策略网络和价值网络
├── buffer.py           # 经验缓冲区
├── trainer.py          # PPO训练器
└── evaluator.py        # 模型评估器
```

## 核心组件

### 1. TradingEnvironment (environment.py)

交易环境模拟器，符合Gymnasium接口标准。

**主要特性：**
- 混合动作空间（离散+连续）
- 完整的交易逻辑（开仓、平仓、止损止盈）
- 真实的交易成本模拟（手续费、滑点）
- 多组件奖励函数设计

**动作空间：**
- `direction`: 离散动作 (0=平仓, 1=做多, 2=做空)
- `position_size`: 连续动作 [0, 1]
- `stop_loss`: 连续动作 [0.001, 0.05]
- `take_profit`: 连续动作 [0.002, 0.10]

**观察空间：**
- 状态向量：256维（来自Transformer）
- 持仓信息：4维
- 风险参数：3维
- 总计：263维

### 2. ActorCritic (policy.py)

Actor-Critic网络架构，包含策略网络和价值网络。

**网络结构：**
- 策略网络（Actor）：
  - 共享层：263 → 512 → 256
  - 离散动作头：输出direction的logits
  - 连续动作头：输出position_size的分布
  - 止损止盈头：输出stop_loss和take_profit
  
- 价值网络（Critic）：
  - 网络：263 → 512 → 256 → 1
  - 输出：状态价值V(s)

**主要方法：**
- `forward()`: 前向传播，采样动作
- `evaluate()`: 评估状态-动作对
- `get_value()`: 获取状态价值

### 3. RolloutBuffer (buffer.py)

经验缓冲区，用于存储和管理训练数据。

**主要功能：**
- 经验存储和检索
- GAE（Generalized Advantage Estimation）计算
- 优势函数标准化
- 批量数据生成

**相关类：**
- `RolloutBuffer`: 主缓冲区
- `MiniBatchSampler`: 小批量采样器
- `EpisodeBuffer`: Episode统计缓冲区

### 4. PPOTrainer (trainer.py)

PPO训练器，实现完整的训练流程。

**训练流程：**
1. 收集rollout经验
2. 计算GAE优势和回报
3. 多轮策略更新
4. 保存检查点和日志

**PPO损失函数：**
- 策略损失：PPO clip目标
- 价值损失：MSE或clipped MSE
- 熵损失：鼓励探索

**主要方法：**
- `collect_rollouts()`: 收集经验
- `update_policy()`: 更新策略
- `train()`: 完整训练流程
- `save_checkpoint()`: 保存模型

### 5. PPOEvaluator (evaluator.py)

模型评估器，提供全面的性能分析。

**评估指标：**
- 收益指标：平均回报、累计收益
- 风险指标：最大回撤、夏普率
- 交易指标：胜率、盈亏比

**主要方法：**
- `evaluate()`: 评估模型性能
- `analyze_actions()`: 分析动作分布
- `generate_report()`: 生成评估报告
- `plot_performance()`: 绘制性能曲线

## 配置文件

配置文件位于 `configs/ppo_config.yaml`，包含：

- **模型配置**: 网络结构参数
- **环境配置**: 交易环境参数
- **训练配置**: PPO算法超参数
- **评估配置**: 评估设置
- **数据配置**: 数据路径

## 使用示例

### 快速开始

```python
import yaml
from src.models.ppo import TradingEnvironment, ActorCritic, PPOTrainer

# 加载配置
with open('configs/ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建环境
env = TradingEnvironment(
    states=states,
    prices=prices,
    **config['environment']
)

# 创建模型
model = ActorCritic(**config['model'])

# 创建训练器
trainer = PPOTrainer(env, model, config['training'])

# 开始训练
trainer.train(total_timesteps=1000000)
```

### 完整示例

参见 `examples/train_ppo.py`

## 测试

运行单元测试：

```bash
python -m pytest tests/test_ppo.py -v
```

或者：

```bash
python tests/test_ppo.py
```

## 文档

详细文档请参见：`docs/PPO.md`

## 技术特点

1. **混合动作空间支持**
   - 同时处理离散和连续动作
   - 灵活的动作表示

2. **GAE优势估计**
   - 平衡偏差和方差
   - 提高训练稳定性

3. **PPO算法优化**
   - Clip目标函数
   - 自适应KL散度控制
   - 梯度裁剪

4. **完整的交易模拟**
   - 真实的交易成本
   - 止损止盈机制
   - 风险管理

5. **全面的评估工具**
   - 多维度性能指标
   - 动作分布分析
   - 可视化报告

## 性能优化

1. **GPU加速**
   - 支持CUDA训练
   - 批量数据处理

2. **内存优化**
   - 高效的缓冲区管理
   - 增量式数据更新

3. **训练加速**
   - 小批量采样
   - 并行环境支持（可扩展）

## 依赖项

- PyTorch >= 2.0.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- tqdm >= 4.65.0

## 版本历史

- **v1.0.0** (2025-11-20)
  - 初始版本发布
  - 实现完整的PPO训练流程
  - 支持混合动作空间
  - 提供全面的评估工具

## 作者

AI Trader Team

## 许可证

MIT License