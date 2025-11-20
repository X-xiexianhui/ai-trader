"""
PPO经验缓冲区实现

实现用于存储和管理PPO训练经验的缓冲区，包括：
- 经验存储和检索
- GAE优势函数计算
- 批量数据生成

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """
    PPO经验缓冲区
    
    存储一个rollout周期内的所有经验，用于PPO训练。
    支持GAE（Generalized Advantage Estimation）计算。
    """
    
    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = 'cpu'
    ):
        """
        初始化经验缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            state_dim: 状态维度
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            device: 设备
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # 初始化存储
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = {
            'direction': np.zeros(buffer_size, dtype=np.int64),
            'position_size': np.zeros((buffer_size, 1), dtype=np.float32),
            'stop_loss': np.zeros((buffer_size, 1), dtype=np.float32),
            'take_profit': np.zeros((buffer_size, 1), dtype=np.float32)
        }
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # GAE计算结果
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        
        logger.info(f"RolloutBuffer initialized: size={buffer_size}, "
                   f"state_dim={state_dim}, gamma={gamma}, gae_lambda={gae_lambda}")
    
    def add(
        self,
        state: np.ndarray,
        action: Dict[str, np.ndarray],
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """
        添加一条经验
        
        Args:
            state: 状态
            action: 动作字典
            reward: 奖励
            value: 状态价值
            log_prob: 对数概率
            done: 是否结束
        """
        assert self.ptr < self.max_size, "Buffer overflow"
        
        self.states[self.ptr] = state
        self.actions['direction'][self.ptr] = action['direction']
        self.actions['position_size'][self.ptr] = action['position_size']
        self.actions['stop_loss'][self.ptr] = action['stop_loss']
        self.actions['take_profit'][self.ptr] = action['take_profit']
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        完成一条轨迹，计算GAE优势和回报
        
        Args:
            last_value: 最后一个状态的价值估计（用于bootstrap）
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # 计算GAE优势
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(
            deltas, self.gamma * self.gae_lambda
        )
        
        # 计算回报
        self.returns[path_slice] = self._discount_cumsum(
            rewards, self.gamma
        )[:-1]
        
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """
        计算折扣累积和
        
        Args:
            x: 输入数组
            discount: 折扣因子
            
        Returns:
            折扣累积和
        """
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            cumsum[t] = x[t] + discount * cumsum[t + 1]
        return cumsum
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        获取所有经验数据
        
        Returns:
            包含所有经验的字典
        """
        assert self.ptr == self.max_size, "Buffer not full"
        
        # 标准化优势
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        # 转换为张量
        data = {
            'states': torch.as_tensor(self.states, dtype=torch.float32).to(self.device),
            'actions': {
                'direction': torch.as_tensor(
                    self.actions['direction'], dtype=torch.long
                ).to(self.device),
                'position_size': torch.as_tensor(
                    self.actions['position_size'], dtype=torch.float32
                ).to(self.device),
                'stop_loss': torch.as_tensor(
                    self.actions['stop_loss'], dtype=torch.float32
                ).to(self.device),
                'take_profit': torch.as_tensor(
                    self.actions['take_profit'], dtype=torch.float32
                ).to(self.device)
            },
            'old_log_probs': torch.as_tensor(
                self.log_probs, dtype=torch.float32
            ).to(self.device),
            'advantages': torch.as_tensor(
                self.advantages, dtype=torch.float32
            ).to(self.device),
            'returns': torch.as_tensor(
                self.returns, dtype=torch.float32
            ).to(self.device),
            'values': torch.as_tensor(
                self.values, dtype=torch.float32
            ).to(self.device)
        }
        
        return data
    
    def reset(self):
        """重置缓冲区"""
        self.ptr = 0
        self.path_start_idx = 0
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.ptr == self.max_size
    
    def size(self) -> int:
        """返回当前缓冲区大小"""
        return self.ptr


class MiniBatchSampler:
    """
    小批量采样器
    
    用于从经验缓冲区中采样小批量数据进行训练。
    """
    
    def __init__(
        self,
        batch_size: int,
        mini_batch_size: int,
        shuffle: bool = True
    ):
        """
        初始化采样器
        
        Args:
            batch_size: 总批量大小
            mini_batch_size: 小批量大小
            shuffle: 是否打乱数据
        """
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.shuffle = shuffle
        
        assert batch_size % mini_batch_size == 0, \
            "batch_size must be divisible by mini_batch_size"
        
        self.num_mini_batches = batch_size // mini_batch_size
        
        logger.info(f"MiniBatchSampler initialized: batch_size={batch_size}, "
                   f"mini_batch_size={mini_batch_size}, "
                   f"num_mini_batches={self.num_mini_batches}")
    
    def sample(
        self,
        data: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        采样小批量数据
        
        Args:
            data: 完整数据字典
            
        Returns:
            小批量数据列表
        """
        indices = np.arange(self.batch_size)
        if self.shuffle:
            np.random.shuffle(indices)
        
        mini_batches = []
        for i in range(self.num_mini_batches):
            start = i * self.mini_batch_size
            end = start + self.mini_batch_size
            batch_indices = indices[start:end]
            
            mini_batch = {
                'states': data['states'][batch_indices],
                'actions': {
                    'direction': data['actions']['direction'][batch_indices],
                    'position_size': data['actions']['position_size'][batch_indices],
                    'stop_loss': data['actions']['stop_loss'][batch_indices],
                    'take_profit': data['actions']['take_profit'][batch_indices]
                },
                'old_log_probs': data['old_log_probs'][batch_indices],
                'advantages': data['advantages'][batch_indices],
                'returns': data['returns'][batch_indices],
                'values': data['values'][batch_indices]
            }
            
            mini_batches.append(mini_batch)
        
        return mini_batches


class EpisodeBuffer:
    """
    Episode缓冲区
    
    用于存储完整的episode数据，便于评估和分析。
    """
    
    def __init__(self, max_episodes: int = 100):
        """
        初始化Episode缓冲区
        
        Args:
            max_episodes: 最大存储episode数量
        """
        self.max_episodes = max_episodes
        self.episodes = []
        
        logger.info(f"EpisodeBuffer initialized: max_episodes={max_episodes}")
    
    def add_episode(
        self,
        states: List[np.ndarray],
        actions: List[Dict[str, np.ndarray]],
        rewards: List[float],
        infos: List[Dict]
    ):
        """
        添加一个episode
        
        Args:
            states: 状态列表
            actions: 动作列表
            rewards: 奖励列表
            infos: 信息列表
        """
        episode = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'infos': infos,
            'total_reward': sum(rewards),
            'length': len(rewards)
        }
        
        self.episodes.append(episode)
        
        # 保持最大数量限制
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        if not self.episodes:
            return {}
        
        total_rewards = [ep['total_reward'] for ep in self.episodes]
        lengths = [ep['length'] for ep in self.episodes]
        
        stats = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.episodes)
        }
        
        return stats
    
    def clear(self):
        """清空缓冲区"""
        self.episodes = []


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== PPO经验缓冲区示例 ===")
    
    # 创建缓冲区
    buffer = RolloutBuffer(
        buffer_size=2048,
        state_dim=263,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    print(f"\n缓冲区大小: {buffer.buffer_size}")
    
    # 模拟添加经验
    print("\n添加经验...")
    for i in range(100):
        state = np.random.randn(263)
        action = {
            'direction': np.random.randint(0, 3),
            'position_size': np.random.rand(1),
            'stop_loss': np.random.uniform(0.001, 0.05, 1),
            'take_profit': np.random.uniform(0.002, 0.10, 1)
        }
        reward = np.random.randn()
        value = np.random.randn()
        log_prob = np.random.randn()
        done = (i % 20 == 19)
        
        buffer.add(state, action, reward, value, log_prob, done)
        
        if done:
            buffer.finish_path(last_value=0.0)
    
    print(f"当前缓冲区大小: {buffer.size()}")
    
    # 创建采样器
    print("\n创建小批量采样器...")
    sampler = MiniBatchSampler(
        batch_size=2048,
        mini_batch_size=256,
        shuffle=True
    )
    
    print(f"小批量数量: {sampler.num_mini_batches}")
    
    # 创建Episode缓冲区
    print("\n创建Episode缓冲区...")
    episode_buffer = EpisodeBuffer(max_episodes=10)
    
    # 添加模拟episode
    for i in range(5):
        states = [np.random.randn(263) for _ in range(20)]
        actions = [{
            'direction': np.random.randint(0, 3),
            'position_size': np.random.rand(1),
            'stop_loss': np.random.uniform(0.001, 0.05, 1),
            'take_profit': np.random.uniform(0.002, 0.10, 1)
        } for _ in range(20)]
        rewards = [np.random.randn() for _ in range(20)]
        infos = [{} for _ in range(20)]
        
        episode_buffer.add_episode(states, actions, rewards, infos)
    
    # 获取统计信息
    stats = episode_buffer.get_statistics()
    print("\nEpisode统计:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n示例完成!")