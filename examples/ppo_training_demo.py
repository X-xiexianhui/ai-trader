"""
PPO训练完整示例
展示如何使用PPO训练交易策略，包括性能评估和可视化
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datetime import datetime

from src.models.ppo import PPOModel, PPOTrainer, TrainingMetrics, TradingEnvironment


def create_mock_data(n_steps: int = 5000) -> tuple:
    """
    创建模拟数据用于演示
    
    Args:
        n_steps: 数据点数量
        
    Returns:
        (data, transformer_states)
    """
    print("创建模拟数据...")
    
    # 创建模拟价格数据（带趋势和噪声）
    dates = pd.date_range('2023-01-01', periods=n_steps, freq='5min')
    
    # 生成价格序列
    trend = np.linspace(100, 120, n_steps)
    noise = np.random.randn(n_steps).cumsum() * 0.5
    close = trend + noise
    
    # 生成OHLC
    high = close + np.abs(np.random.randn(n_steps)) * 0.5
    low = close - np.abs(np.random.randn(n_steps)) * 0.5
    open_price = close + np.random.randn(n_steps) * 0.3
    volume = np.random.randint(1000, 10000, n_steps)
    
    data = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    # 创建模拟Transformer状态（256维）
    transformer_states = np.random.randn(n_steps, 256).astype(np.float32)
    
    # 添加一些结构（让状态与价格有关联）
    for i in range(256):
        transformer_states[:, i] += close / 100 * np.random.randn()
    
    print(f"  数据形状: {data.shape}")
    print(f"  Transformer状态形状: {transformer_states.shape}")
    print(f"  价格范围: {close.min():.2f} - {close.max():.2f}")
    
    return data, transformer_states


def main():
    """主函数"""
    print("=" * 80)
    print("PPO交易策略训练示例")
    print("=" * 80)
    
    # 1. 创建数据
    data, transformer_states = create_mock_data(n_steps=5000)
    
    # 2. 创建交易环境
    print("\n创建交易环境...")
    env = TradingEnvironment(
        data=data,
        transformer_states=transformer_states,
        initial_balance=100000.0,
        transaction_cost=0.0002,
        slippage=0.0001,
        max_position_size=1.0
    )
    print(f"  环境: {env}")
    
    # 3. 创建PPO模型
    print("\n创建PPO模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  使用设备: {device}")
    
    model = PPOModel(
        state_dim=263,  # 256 (Transformer) + 7 (持仓/风险信息)
        hidden_dim=512,
        dropout=0.1,
        lr_actor=1e-4,
        lr_critic=3e-4,
        device=device
    )
    
    # 显示模型信息
    model_info = model.get_model_info()
    print(f"  Actor参数量: {model_info['actor_parameters']:,}")
    print(f"  Critic参数量: {model_info['critic_parameters']:,}")
    print(f"  总参数量: {model_info['total_parameters']:,}")
    
    # 4. 配置训练参数
    print("\n配置训练参数...")
    config = {
        # PPO参数
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_clip': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # 训练参数
        'n_steps': 1024,  # 每次迭代收集的步数
        'batch_size': 64,
        'n_epochs': 10,
        'buffer_capacity': 10000,
        
        # 其他
        'save_interval': 5,
        'log_interval': 1,
        'eval_interval': 5
    }
    
    # 5. 创建训练器
    print("\n创建训练器...")
    trainer = PPOTrainer(
        model=model,
        env=env,
        config=config
    )
    
    # 6. 开始训练
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    
    save_dir = Path("models/ppo_demo")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        trainer.train(
            n_iterations=20,  # 训练20个迭代
            save_dir=str(save_dir)
        )
    except KeyboardInterrupt:
        print("\n训练被中断")
    
    # 7. 评估训练好的策略（包含详细性能指标）
    print("\n" + "=" * 80)
    print("评估训练好的策略")
    print("=" * 80)
    
    # 使用增强的评估方法（自动打印详细信息）
    eval_stats = trainer.evaluate(n_episodes=20, verbose=True)
    
    # 8. 可视化训练历史
    print("\n" + "=" * 80)
    print("训练历史摘要")
    print("=" * 80)
    
    history = trainer.history
    if len(history['iteration']) > 0:
        print(f"\n奖励统计:")
        print(f"  初始平均奖励: {history['mean_reward'][0]:.4f}")
        print(f"  最终平均奖励: {history['mean_reward'][-1]:.4f}")
        print(f"  最佳平均奖励: {max(history['mean_reward']):.4f}")
        
        print(f"\n损失统计:")
        print(f"  最终策略损失: {history['policy_loss'][-1]:.4f}")
        print(f"  最终价值损失: {history['value_loss'][-1]:.4f}")
        print(f"  最终熵: {history['entropy'][-1]:.4f}")
        
        print(f"\n训练指标:")
        print(f"  最终KL散度: {history['kl_divergence'][-1]:.6f}")
        print(f"  最终裁剪比例: {history['clip_fraction'][-1]:.2%}")
        print(f"  最终解释方差: {history['explained_variance'][-1]:.4f}")
    
    # 9. 生成详细性能报告
    print("\n" + "=" * 80)
    print("生成详细性能报告")
    print("=" * 80)
    
    # 收集更多评估数据
    print("\n收集50个episode的数据...")
    eval_rewards = []
    eval_lengths = []
    
    for ep in range(50):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _, _ = model.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        
        if (ep + 1) % 10 == 0:
            print(f"  完成 {ep + 1}/50 episodes")
    
    # 生成性能报告
    performance_report = TrainingMetrics.generate_performance_report(
        eval_rewards, eval_lengths
    )
    
    print("\n详细性能指标:")
    print(f"  Episode数: {performance_report['n_episodes']}")
    print(f"  平均奖励: {performance_report['mean_reward']:.4f}")
    print(f"  标准差: {performance_report['std_reward']:.4f}")
    print(f"  中位数: {performance_report['median_reward']:.4f}")
    print(f"  最小值: {performance_report['min_reward']:.4f}")
    print(f"  最大值: {performance_report['max_reward']:.4f}")
    print(f"\n风险调整收益:")
    print(f"  夏普比率: {performance_report['sharpe_ratio']:.4f}")
    print(f"  索提诺比率: {performance_report['sortino_ratio']:.4f}")
    print(f"\n交易统计:")
    print(f"  胜率: {performance_report['win_rate']:.2%}")
    print(f"  盈亏比: {performance_report['profit_factor']:.4f}")
    
    # 10. 测试推理
    print("\n" + "=" * 80)
    print("测试推理")
    print("=" * 80)
    
    state = env.reset()
    print(f"\n初始状态形状: {state.shape}")
    
    # 执行几步
    for step in range(5):
        action, log_prob, value = model.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  动作: direction={action['direction']}, "
              f"position_size={action['position_size']:.3f}, "
              f"stop_loss={action['stop_loss']:.4f}, "
              f"take_profit={action['take_profit']:.4f}")
        print(f"  奖励: {reward:.4f}")
        print(f"  账户余额: ${info['balance']:.2f}")
        print(f"  最大回撤: {info['max_drawdown']:.2%}")
        
        if done:
            print("  Episode结束")
            break
        
        state = next_state
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print(f"\n模型已保存到: {save_dir}")
    print(f"训练历史已保存到: {save_dir / 'training_history.json'}")
    print(f"训练曲线已保存到: {save_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()