"""
PPO训练器实现

实现PPO算法的完整训练流程，包括：
- 数据收集和经验回放
- PPO损失函数计算
- 策略和价值网络更新
- 训练监控和日志记录

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import logging
from tqdm import tqdm
import json

from .policy import ActorCritic
from .buffer import RolloutBuffer, MiniBatchSampler, EpisodeBuffer
from .environment import TradingEnvironment

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO训练器
    
    实现完整的PPO训练流程，包括数据收集、策略更新和评估。
    """
    
    def __init__(
        self,
        env: TradingEnvironment,
        model: ActorCritic,
        config: Dict,
        save_dir: str = 'models/ppo',
        device: str = 'cpu'
    ):
        """
        初始化PPO训练器
        
        Args:
            env: 交易环境
            model: Actor-Critic模型
            config: 训练配置
            save_dir: 模型保存目录
            device: 设备
        """
        self.env = env
        self.model = model.to(device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # 训练超参数
        self.n_steps = config.get('n_steps', 2048)
        self.n_epochs = config.get('n_epochs', 10)
        self.mini_batch_size = config.get('mini_batch_size', 256)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.clip_range_vf = config.get('clip_range_vf', None)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.target_kl = config.get('target_kl', 0.01)
        
        # 优化器
        self.policy_lr = config.get('policy_lr', 1e-4)
        self.value_lr = config.get('value_lr', 3e-4)
        
        self.optimizer = optim.Adam([
            {'params': self.model.policy.parameters(), 'lr': self.policy_lr},
            {'params': self.model.value.parameters(), 'lr': self.value_lr}
        ])
        
        # 学习率调度器
        self.use_lr_scheduler = config.get('use_lr_scheduler', True)
        if self.use_lr_scheduler:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('lr_step_size', 100),
                gamma=config.get('lr_gamma', 0.95)
            )
        
        # 缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=self.n_steps,
            state_dim=env.observation_space.shape[0],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device
        )
        
        self.sampler = MiniBatchSampler(
            batch_size=self.n_steps,
            mini_batch_size=self.mini_batch_size,
            shuffle=True
        )
        
        self.episode_buffer = EpisodeBuffer(max_episodes=100)
        
        # 训练统计
        self.total_timesteps = 0
        self.num_updates = 0
        self.training_history = []
        
        logger.info(f"PPOTrainer initialized with config: {config}")
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        收集一个rollout的经验
        
        Returns:
            收集统计信息
        """
        self.model.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_states = []
        episode_actions = []
        episode_rewards_list = []
        episode_infos = []
        
        with torch.no_grad():
            for step in range(self.n_steps):
                # 转换状态为张量
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # 采样动作
                action, log_prob, _, value = self.model(state_tensor)
                
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
                
                # 存储经验
                self.buffer.add(
                    state=state,
                    action=action_np,
                    reward=reward,
                    value=value.cpu().item(),
                    log_prob=log_prob.cpu().item(),
                    done=done
                )
                
                # 记录episode数据
                episode_states.append(state)
                episode_actions.append(action_np)
                episode_rewards_list.append(reward)
                episode_infos.append(info)
                
                episode_reward += reward
                episode_length += 1
                self.total_timesteps += 1
                
                # 处理episode结束
                if done:
                    # 完成轨迹
                    self.buffer.finish_path(last_value=0.0)
                    
                    # 记录episode
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    self.episode_buffer.add_episode(
                        episode_states, episode_actions,
                        episode_rewards_list, episode_infos
                    )
                    
                    # 重置
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                    episode_states = []
                    episode_actions = []
                    episode_rewards_list = []
                    episode_infos = []
                else:
                    state = next_state
            
            # 处理未完成的轨迹
            if episode_length > 0:
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                last_value = self.model.get_value(state_tensor).cpu().item()
                self.buffer.finish_path(last_value=last_value)
        
        # 统计信息
        stats = {
            'mean_episode_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'num_episodes': len(episode_rewards)
        }
        
        return stats
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算PPO损失
        
        Args:
            batch: 小批量数据
            
        Returns:
            total_loss: 总损失
            loss_info: 损失信息字典
        """
        # 评估动作
        log_probs, entropy, values = self.model.evaluate(
            batch['states'],
            batch['actions']
        )
        
        values = values.squeeze(-1)
        
        # 策略损失（PPO clip）
        ratio = torch.exp(log_probs - batch['old_log_probs'])
        surr1 = ratio * batch['advantages']
        surr2 = torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range
        ) * batch['advantages']
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值损失
        if self.clip_range_vf is not None:
            # Clipped value loss
            values_pred_clipped = batch['values'] + torch.clamp(
                values - batch['values'],
                -self.clip_range_vf,
                self.clip_range_vf
            )
            value_loss1 = (values - batch['returns']).pow(2)
            value_loss2 = (values_pred_clipped - batch['returns']).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * (values - batch['returns']).pow(2).mean()
        
        # 熵损失
        entropy_loss = -entropy.mean()
        
        # 总损失
        total_loss = (
            policy_loss +
            self.vf_coef * value_loss +
            self.ent_coef * entropy_loss
        )
        
        # 损失信息
        loss_info = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'approx_kl': ((ratio - 1) - torch.log(ratio)).mean().item(),
            'clip_fraction': ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
        }
        
        return total_loss, loss_info
    
    def update_policy(self) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Returns:
            更新统计信息
        """
        self.model.train()
        
        # 获取经验数据
        data = self.buffer.get()
        
        # 训练统计
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        approx_kls = []
        clip_fractions = []
        
        # 多轮更新
        for epoch in range(self.n_epochs):
            # 采样小批量
            mini_batches = self.sampler.sample(data)
            
            for batch in mini_batches:
                # 计算损失
                loss, loss_info = self.compute_loss(batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.optimizer.step()
                
                # 记录统计
                policy_losses.append(loss_info['policy_loss'])
                value_losses.append(loss_info['value_loss'])
                entropy_losses.append(loss_info['entropy_loss'])
                total_losses.append(loss_info['total_loss'])
                approx_kls.append(loss_info['approx_kl'])
                clip_fractions.append(loss_info['clip_fraction'])
            
            # 早停（如果KL散度过大）
            mean_kl = np.mean(approx_kls[-len(mini_batches):])
            if self.target_kl is not None and mean_kl > 1.5 * self.target_kl:
                logger.warning(f"Early stopping at epoch {epoch} due to reaching max kl: {mean_kl:.4f}")
                break
        
        # 更新学习率
        if self.use_lr_scheduler:
            self.scheduler.step()
        
        self.num_updates += 1
        
        # 统计信息
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def train(
        self,
        total_timesteps: int,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 50
    ):
        """
        训练PPO模型
        
        Args:
            total_timesteps: 总训练步数
            log_interval: 日志记录间隔
            save_interval: 模型保存间隔
            eval_interval: 评估间隔
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        num_updates = total_timesteps // self.n_steps
        
        with tqdm(total=num_updates, desc="Training") as pbar:
            for update in range(num_updates):
                # 收集经验
                rollout_stats = self.collect_rollouts()
                
                # 更新策略
                update_stats = self.update_policy()
                
                # 重置缓冲区
                self.buffer.reset()
                
                # 合并统计信息
                stats = {**rollout_stats, **update_stats}
                stats['timesteps'] = self.total_timesteps
                stats['update'] = self.num_updates
                
                # 记录历史
                self.training_history.append(stats)
                
                # 日志记录
                if (update + 1) % log_interval == 0:
                    logger.info(
                        f"Update {update + 1}/{num_updates} | "
                        f"Timesteps: {self.total_timesteps} | "
                        f"Mean Reward: {stats['mean_episode_reward']:.4f} | "
                        f"Policy Loss: {stats['policy_loss']:.4f} | "
                        f"Value Loss: {stats['value_loss']:.4f}"
                    )
                
                # 保存模型
                if (update + 1) % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_{update + 1}.pt")
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    'reward': f"{stats['mean_episode_reward']:.2f}",
                    'loss': f"{stats['total_loss']:.4f}"
                })
        
        # 保存最终模型
        self.save_checkpoint("final_model.pt")
        self.save_training_history()
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """
        保存检查点
        
        Args:
            filename: 文件名
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'config': self.config
        }
        
        if self.use_lr_scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filename: str):
        """
        加载检查点
        
        Args:
            filename: 文件名
        """
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_timesteps = checkpoint['total_timesteps']
        self.num_updates = checkpoint['num_updates']
        
        if self.use_lr_scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {load_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== PPO训练器示例 ===")
    
    # 创建模拟数据
    print("\n创建模拟环境和模型...")
    
    # 这里需要实际的环境和模型实例
    # env = TradingEnvironment(...)
    # model = ActorCritic(...)
    
    # 训练配置
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
    
    print("\n训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n示例完成!")