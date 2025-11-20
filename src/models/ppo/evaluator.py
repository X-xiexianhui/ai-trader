"""
PPO评估器实现

实现PPO模型的评估功能，包括：
- 模型性能评估
- 交易指标计算
- 可视化和报告生成

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from .policy import ActorCritic
from .environment import TradingEnvironment

logger = logging.getLogger(__name__)


class PPOEvaluator:
    """
    PPO评估器
    
    评估训练好的PPO模型在交易环境中的性能。
    """
    
    def __init__(
        self,
        model: ActorCritic,
        env: TradingEnvironment,
        device: str = 'cpu'
    ):
        """
        初始化评估器
        
        Args:
            model: Actor-Critic模型
            env: 交易环境
            device: 设备
        """
        self.model = model.to(device)
        self.env = env
        self.device = device
        
        self.model.eval()
        
        logger.info("PPOEvaluator initialized")
    
    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估episode数量
            deterministic: 是否使用确定性策略
            render: 是否渲染环境
            
        Returns:
            评估指标字典
        """
        logger.info(f"Starting evaluation for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        episode_sharpe_ratios = []
        episode_max_drawdowns = []
        episode_win_rates = []
        
        all_actions = []
        all_rewards = []
        all_balances = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_actions = []
                episode_rewards_list = []
                episode_balances = []
                
                done = False
                while not done:
                    # 转换状态为张量
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                    
                    # 采样动作
                    action, _, _, _ = self.model(state_tensor, deterministic=deterministic)
                    
                    # 转换动作为numpy
                    action_np = {
                        'direction': action['direction'].cpu().numpy()[0],
                        'position_size': action['position_size'].cpu().numpy()[0],
                        'stop_loss': action['stop_loss'].cpu().numpy()[0],
                        'take_profit': action['take_profit'].cpu().numpy()[0]
                    }
                    
                    # 执行动作
                    next_state, reward, terminated, truncated, info = self.env.step(action_np)
                    done = terminated or truncated
                    
                    # 记录数据
                    episode_actions.append(action_np)
                    episode_rewards_list.append(reward)
                    episode_balances.append(info.get('balance', 0))
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    state = next_state
                    
                    if render:
                        self.env.render()
                
                # 计算episode指标
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_returns.append(self.env.balance / self.env.initial_balance - 1)
                
                # 计算夏普率
                if len(episode_rewards_list) > 1:
                    returns = np.diff(episode_balances) / episode_balances[:-1]
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)  # 5分钟数据
                    episode_sharpe_ratios.append(sharpe)
                
                # 计算最大回撤
                balances = np.array(episode_balances)
                running_max = np.maximum.accumulate(balances)
                drawdown = (balances - running_max) / running_max
                max_drawdown = np.min(drawdown)
                episode_max_drawdowns.append(max_drawdown)
                
                # 计算胜率
                profitable_trades = sum(1 for r in episode_rewards_list if r > 0)
                total_trades = len([r for r in episode_rewards_list if r != 0])
                win_rate = profitable_trades / total_trades if total_trades > 0 else 0
                episode_win_rates.append(win_rate)
                
                # 记录所有数据
                all_actions.extend(episode_actions)
                all_rewards.extend(episode_rewards_list)
                all_balances.extend(episode_balances)
                
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} | "
                    f"Reward: {episode_reward:.4f} | "
                    f"Return: {episode_returns[-1]:.4%} | "
                    f"Length: {episode_length}"
                )
        
        # 汇总统计
        metrics = {
            'mean_episode_reward': np.mean(episode_rewards),
            'std_episode_reward': np.std(episode_rewards),
            'min_episode_reward': np.min(episode_rewards),
            'max_episode_reward': np.max(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'mean_sharpe_ratio': np.mean(episode_sharpe_ratios) if episode_sharpe_ratios else 0,
            'mean_max_drawdown': np.mean(episode_max_drawdowns),
            'mean_win_rate': np.mean(episode_win_rates),
            'num_episodes': num_episodes
        }
        
        logger.info("Evaluation completed")
        logger.info(f"Mean Episode Reward: {metrics['mean_episode_reward']:.4f}")
        logger.info(f"Mean Return: {metrics['mean_return']:.4%}")
        logger.info(f"Mean Sharpe Ratio: {metrics['mean_sharpe_ratio']:.4f}")
        logger.info(f"Mean Max Drawdown: {metrics['mean_max_drawdown']:.4%}")
        logger.info(f"Mean Win Rate: {metrics['mean_win_rate']:.4%}")
        
        return metrics
    
    def analyze_actions(
        self,
        num_episodes: int = 10
    ) -> Dict[str, any]:
        """
        分析模型的动作分布
        
        Args:
            num_episodes: 分析的episode数量
            
        Returns:
            动作分析结果
        """
        logger.info("Analyzing action distribution")
        
        directions = []
        position_sizes = []
        stop_losses = []
        take_profits = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                done = False
                
                while not done:
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                    
                    action, _, _, _ = self.model(state_tensor, deterministic=True)
                    
                    directions.append(action['direction'].cpu().item())
                    position_sizes.append(action['position_size'].cpu().item())
                    stop_losses.append(action['stop_loss'].cpu().item())
                    take_profits.append(action['take_profit'].cpu().item())
                    
                    action_np = {
                        'direction': action['direction'].cpu().numpy()[0],
                        'position_size': action['position_size'].cpu().numpy()[0],
                        'stop_loss': action['stop_loss'].cpu().numpy()[0],
                        'take_profit': action['take_profit'].cpu().numpy()[0]
                    }
                    
                    next_state, _, terminated, truncated, _ = self.env.step(action_np)
                    done = terminated or truncated
                    state = next_state
        
        # 统计分析
        analysis = {
            'direction_distribution': {
                'close': (np.array(directions) == 0).sum() / len(directions),
                'long': (np.array(directions) == 1).sum() / len(directions),
                'short': (np.array(directions) == 2).sum() / len(directions)
            },
            'position_size_stats': {
                'mean': np.mean(position_sizes),
                'std': np.std(position_sizes),
                'min': np.min(position_sizes),
                'max': np.max(position_sizes)
            },
            'stop_loss_stats': {
                'mean': np.mean(stop_losses),
                'std': np.std(stop_losses),
                'min': np.min(stop_losses),
                'max': np.max(stop_losses)
            },
            'take_profit_stats': {
                'mean': np.mean(take_profits),
                'std': np.std(take_profits),
                'min': np.min(take_profits),
                'max': np.max(take_profits)
            }
        }
        
        logger.info("Action analysis completed")
        return analysis
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        action_analysis: Dict[str, any],
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            action_analysis: 动作分析结果
            save_path: 保存路径
            
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("PPO Model Evaluation Report")
        report.append("=" * 80)
        report.append(f"\nGenerated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n\n" + "=" * 80)
        report.append("Performance Metrics")
        report.append("=" * 80)
        report.append(f"\nNumber of Episodes: {metrics['num_episodes']}")
        report.append(f"Mean Episode Reward: {metrics['mean_episode_reward']:.4f} ± {metrics['std_episode_reward']:.4f}")
        report.append(f"Mean Episode Length: {metrics['mean_episode_length']:.2f}")
        report.append(f"Mean Return: {metrics['mean_return']:.4%} ± {metrics['std_return']:.4%}")
        report.append(f"Mean Sharpe Ratio: {metrics['mean_sharpe_ratio']:.4f}")
        report.append(f"Mean Max Drawdown: {metrics['mean_max_drawdown']:.4%}")
        report.append(f"Mean Win Rate: {metrics['mean_win_rate']:.4%}")
        
        report.append("\n\n" + "=" * 80)
        report.append("Action Distribution")
        report.append("=" * 80)
        dir_dist = action_analysis['direction_distribution']
        report.append(f"\nDirection Distribution:")
        report.append(f"  Close: {dir_dist['close']:.2%}")
        report.append(f"  Long:  {dir_dist['long']:.2%}")
        report.append(f"  Short: {dir_dist['short']:.2%}")
        
        pos_stats = action_analysis['position_size_stats']
        report.append(f"\nPosition Size Statistics:")
        report.append(f"  Mean: {pos_stats['mean']:.4f}")
        report.append(f"  Std:  {pos_stats['std']:.4f}")
        report.append(f"  Min:  {pos_stats['min']:.4f}")
        report.append(f"  Max:  {pos_stats['max']:.4f}")
        
        sl_stats = action_analysis['stop_loss_stats']
        report.append(f"\nStop Loss Statistics:")
        report.append(f"  Mean: {sl_stats['mean']:.4f}")
        report.append(f"  Std:  {sl_stats['std']:.4f}")
        report.append(f"  Min:  {sl_stats['min']:.4f}")
        report.append(f"  Max:  {sl_stats['max']:.4f}")
        
        tp_stats = action_analysis['take_profit_stats']
        report.append(f"\nTake Profit Statistics:")
        report.append(f"  Mean: {tp_stats['mean']:.4f}")
        report.append(f"  Std:  {tp_stats['std']:.4f}")
        report.append(f"  Min:  {tp_stats['min']:.4f}")
        report.append(f"  Max:  {tp_stats['max']:.4f}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {save_path}")
        
        return report_text
    
    def plot_performance(
        self,
        metrics_history: List[Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        绘制性能曲线
        
        Args:
            metrics_history: 指标历史记录
            save_path: 保存路径
        """
        if not metrics_history:
            logger.warning("No metrics history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PPO Training Performance', fontsize=16)
        
        # 提取数据
        timesteps = [m['timesteps'] for m in metrics_history]
        rewards = [m['mean_episode_reward'] for m in metrics_history]
        policy_losses = [m['policy_loss'] for m in metrics_history]
        value_losses = [m['value_loss'] for m in metrics_history]
        entropy_losses = [m['entropy_loss'] for m in metrics_history]
        
        # 绘制奖励曲线
        axes[0, 0].plot(timesteps, rewards)
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Mean Episode Reward')
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].grid(True)
        
        # 绘制策略损失
        axes[0, 1].plot(timesteps, policy_losses)
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].grid(True)
        
        # 绘制价值损失
        axes[1, 0].plot(timesteps, value_losses)
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].grid(True)
        
        # 绘制熵损失
        axes[1, 1].plot(timesteps, entropy_losses)
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Entropy Loss')
        axes[1, 1].set_title('Entropy Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        plt.close()


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== PPO评估器示例 ===")
    
    # 这里需要实际的模型和环境实例
    # model = ActorCritic(...)
    # env = TradingEnvironment(...)
    # evaluator = PPOEvaluator(model, env)
    
    # 评估模型
    # metrics = evaluator.evaluate(num_episodes=10)
    
    # 分析动作
    # action_analysis = evaluator.analyze_actions(num_episodes=10)
    
    # 生成报告
    # report = evaluator.generate_report(metrics, action_analysis)
    # print(report)
    
    print("\n示例完成!")