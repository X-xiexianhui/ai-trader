"""
PPO训练器
实现完整的PPO训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from .model import PPOModel, ExperienceBuffer
from .metrics import TrainingMetrics


class PPOTrainer:
    """
    PPO训练器
    
    实现PPO算法的完整训练流程，包括：
    - 策略损失、价值损失、熵正则化
    - 经验收集和更新
    - 模型保存和加载
    - 训练监控
    """
    
    def __init__(
        self,
        model: PPOModel,
        env,
        config: Optional[Dict] = None
    ):
        """
        初始化训练器
        
        Args:
            model: PPO模型
            env: 交易环境
            config: 训练配置
        """
        self.model = model
        self.env = env
        
        # 默认配置
        self.config = {
            # PPO参数
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_clip': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            
            # 训练参数
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'buffer_capacity': 10000,
            
            # 其他
            'save_interval': 10,
            'log_interval': 1,
            'eval_interval': 5
        }
        
        if config:
            self.config.update(config)
        
        # 训练状态
        self.iteration = 0
        self.total_steps = 0
        self.best_reward = -np.inf
        
        # 训练历史
        self.history = {
            'iteration': [],
            'mean_reward': [],
            'mean_episode_length': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': []
        }
    
    def compute_ppo_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算PPO损失
        
        Args:
            batch: 批次数据
            
        Returns:
            (total_loss, loss_dict)
        """
        states = batch['states'].to(self.model.device)
        actions = {k: v.to(self.model.device) for k, v in batch['actions'].items()}
        old_log_probs = batch['old_log_probs'].to(self.model.device)
        advantages = batch['advantages'].to(self.model.device)
        returns = batch['returns'].to(self.model.device)
        old_values = batch['old_values'].to(self.model.device)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 评估动作
        new_log_probs, entropy = self.model.actor.evaluate_actions(states, actions)
        new_values = self.model.critic(states).squeeze(-1)
        
        # 1. 策略损失（PPO-Clip）
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config['clip_epsilon'],
            1.0 + self.config['clip_epsilon']
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 2. 价值损失（带裁剪）
        if self.config['value_clip'] > 0:
            value_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.config['value_clip'],
                self.config['value_clip']
            )
            value_loss1 = F.mse_loss(new_values, returns)
            value_loss2 = F.mse_loss(value_clipped, returns)
            value_loss = torch.max(value_loss1, value_loss2)
        else:
            value_loss = F.mse_loss(new_values, returns)
        
        # 3. 熵损失（鼓励探索）
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (
            policy_loss +
            self.config['value_coef'] * value_loss +
            self.config['entropy_coef'] * entropy_loss
        )
        
        # 计算额外指标
        with torch.no_grad():
            kl_div = (old_log_probs - new_log_probs).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > self.config['clip_epsilon']).float().mean().item()
            var_y = returns.var()
            explained_var = 1 - (returns - new_values).var() / (var_y + 1e-8)
            explained_var = explained_var.item()
        
        loss_dict = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': total_loss.item(),
            'kl_divergence': kl_div,
            'clip_fraction': clip_fraction,
            'explained_variance': explained_var
        }
        
        return total_loss, loss_dict
    
    def collect_experience(
        self,
        n_steps: int
    ) -> Tuple[ExperienceBuffer, Dict[str, float]]:
        """
        收集经验
        
        Args:
            n_steps: 收集步数
            
        Returns:
            (buffer, episode_stats)
        """
        buffer = ExperienceBuffer(capacity=self.config['buffer_capacity'])
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        state = self.env.reset()
        
        for step in range(n_steps):
            action, log_prob, value = self.model.select_action(state, deterministic=False)
            next_state, reward, done, info = self.env.step(action)
            
            buffer.add(state, action, reward, done, value, log_prob)
            
            current_episode_reward += reward
            current_episode_length += 1
            self.total_steps += 1
            
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0
                current_episode_length = 0
                state = self.env.reset()
            else:
                state = next_state
        
        # 计算最后一个状态的价值
        _, _, next_value = self.model.select_action(state, deterministic=False)
        
        # 计算GAE
        buffer.compute_gae(
            next_value=next_value,
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda']
        )
        
        episode_stats = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'n_episodes': len(episode_rewards)
        }
        
        return buffer, episode_stats
    
    def update_policy(self, buffer: ExperienceBuffer) -> Dict[str, float]:
        """
        更新策略
        
        Args:
            buffer: 经验缓冲区
            
        Returns:
            更新统计信息
        """
        update_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': []
        }
        
        for epoch in range(self.config['n_epochs']):
            batches = buffer.get_batches(
                batch_size=self.config['batch_size'],
                shuffle=True
            )
            
            for batch in batches:
                total_loss, loss_dict = self.compute_ppo_loss(batch)
                
                self.model.actor_optimizer.zero_grad()
                self.model.critic_optimizer.zero_grad()
                total_loss.backward()
                
                nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    self.config['max_grad_norm']
                )
                nn.utils.clip_grad_norm_(
                    self.model.critic.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.model.actor_optimizer.step()
                self.model.critic_optimizer.step()
                
                for key, value in loss_dict.items():
                    update_stats[key].append(value)
        
        avg_stats = {key: np.mean(values) for key, values in update_stats.items()}
        return avg_stats
    
    def train(
        self,
        n_iterations: int,
        save_dir: Optional[str] = None
    ):
        """
        训练主循环
        
        Args:
            n_iterations: 训练迭代次数
            save_dir: 模型保存目录
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"开始PPO训练...")
        print(f"配置: {json.dumps(self.config, indent=2)}")
        print(f"设备: {self.model.device}")
        print("-" * 80)
        
        for iteration in range(n_iterations):
            self.iteration = iteration
            
            # 1. 收集经验
            buffer, episode_stats = self.collect_experience(self.config['n_steps'])
            
            # 2. 更新策略
            update_stats = self.update_policy(buffer)
            
            # 3. 记录历史
            self.history['iteration'].append(iteration)
            self.history['mean_reward'].append(episode_stats['mean_reward'])
            self.history['mean_episode_length'].append(episode_stats['mean_length'])
            self.history['policy_loss'].append(update_stats['policy_loss'])
            self.history['value_loss'].append(update_stats['value_loss'])
            self.history['entropy'].append(update_stats['entropy'])
            self.history['kl_divergence'].append(update_stats['kl_divergence'])
            self.history['clip_fraction'].append(update_stats['clip_fraction'])
            self.history['explained_variance'].append(update_stats['explained_variance'])
            
            # 4. 日志输出
            if iteration % self.config['log_interval'] == 0:
                print(f"\nIteration {iteration}/{n_iterations}")
                print(f"  总步数: {self.total_steps}")
                print(f"  Episode统计:")
                print(f"    平均奖励: {episode_stats['mean_reward']:.4f} ± {episode_stats['std_reward']:.4f}")
                print(f"    平均长度: {episode_stats['mean_length']:.1f}")
                print(f"    Episode数: {episode_stats['n_episodes']}")
                print(f"  更新统计:")
                print(f"    策略损失: {update_stats['policy_loss']:.4f}")
                print(f"    价值损失: {update_stats['value_loss']:.4f}")
                print(f"    熵: {update_stats['entropy']:.4f}")
                print(f"    KL散度: {update_stats['kl_divergence']:.6f}")
                print(f"    裁剪比例: {update_stats['clip_fraction']:.2%}")
                print(f"    解释方差: {update_stats['explained_variance']:.4f}")
            
            # 5. 保存模型
            if save_dir and iteration % self.config['save_interval'] == 0:
                model_path = save_dir / f"ppo_iter_{iteration}.pt"
                self.model.save(str(model_path))
                
                if episode_stats['mean_reward'] > self.best_reward:
                    self.best_reward = episode_stats['mean_reward']
                    best_path = save_dir / "ppo_best.pt"
                    self.model.save(str(best_path))
                    print(f"  ✓ 保存最佳模型 (奖励: {self.best_reward:.4f})")
                
                history_path = save_dir / "training_history.json"
                with open(history_path, 'w') as f:
                    json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳平均奖励: {self.best_reward:.4f}")
        print(f"总训练步数: {self.total_steps}")
        
        if save_dir:
            plot_path = save_dir / "training_curves.png"
            TrainingMetrics.plot_training_curves(self.history, str(plot_path))
    
    def evaluate(
        self,
        n_episodes: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        评估策略
        
        Args:
            n_episodes: 评估episode数
            verbose: 是否打印详细信息
            
        Returns:
            评估统计信息
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.model.select_action(state, deterministic=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }
        
        performance_report = TrainingMetrics.generate_performance_report(
            episode_rewards, episode_lengths
        )
        eval_stats.update(performance_report)
        
        if verbose:
            print("\n" + "=" * 80)
            print("评估结果:")
            print(f"  Episode数: {n_episodes}")
            print(f"  平均奖励: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
            print(f"  奖励范围: [{eval_stats['min_reward']:.4f}, {eval_stats['max_reward']:.4f}]")
            print(f"  平均长度: {eval_stats['mean_length']:.1f}")
            print(f"  夏普比率: {eval_stats['sharpe_ratio']:.4f}")
            print(f"  索提诺比率: {eval_stats['sortino_ratio']:.4f}")
            print(f"  胜率: {eval_stats['win_rate']:.2%}")
            print(f"  盈亏比: {eval_stats['profit_factor']:.4f}")
            print("=" * 80)
        
        return eval_stats


if __name__ == "__main__":
    print("PPO Trainer模块已加载")
    print("请使用完整的训练脚本进行训练")