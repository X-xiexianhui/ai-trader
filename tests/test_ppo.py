"""
PPO模块单元测试

测试PPO强化学习模块的各个组件。

Author: AI Trader Team
Date: 2025-11-20
"""

import unittest
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ppo import (
    TradingEnvironment,
    PolicyNetwork,
    ValueNetwork,
    ActorCritic,
    RolloutBuffer,
    MiniBatchSampler,
    EpisodeBuffer
)


class TestTradingEnvironment(unittest.TestCase):
    """测试交易环境"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟数据
        self.num_steps = 100
        self.state_dim = 256
        
        self.states = np.random.randn(self.num_steps, self.state_dim).astype(np.float32)
        self.prices = np.cumsum(np.random.randn(self.num_steps) * 0.01) + 100
        
        self.env = TradingEnvironment(
            states=self.states,
            prices=self.prices,
            initial_balance=100000.0,
            max_position_size=0.5,
            commission_rate=0.0003,
            slippage=0.0001
        )
    
    def test_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.env.initial_balance, 100000.0)
        self.assertEqual(self.env.balance, 100000.0)
        self.assertEqual(self.env.current_step, 0)
        self.assertIsNone(self.env.position)
    
    def test_observation_space(self):
        """测试观察空间"""
        obs_space = self.env.observation_space
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertEqual(obs_space.shape[0], self.state_dim + 7)
    
    def test_action_space(self):
        """测试动作空间"""
        action_space = self.env.action_space
        self.assertIsInstance(action_space, gym.spaces.Dict)
        self.assertIn('direction', action_space.spaces)
        self.assertIn('position_size', action_space.spaces)
        self.assertIn('stop_loss', action_space.spaces)
        self.assertIn('take_profit', action_space.spaces)
    
    def test_reset(self):
        """测试环境重置"""
        obs, info = self.env.reset()
        
        self.assertEqual(obs.shape[0], self.state_dim + 7)
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, self.env.initial_balance)
        self.assertIsNone(self.env.position)
    
    def test_step_close_position(self):
        """测试平仓动作"""
        self.env.reset()
        
        action = {
            'direction': 0,  # 平仓
            'position_size': np.array([0.3]),
            'stop_loss': np.array([0.02]),
            'take_profit': np.array([0.05])
        }
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(obs.shape[0], self.state_dim + 7)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
    
    def test_step_open_long(self):
        """测试开多仓"""
        self.env.reset()
        
        action = {
            'direction': 1,  # 做多
            'position_size': np.array([0.3]),
            'stop_loss': np.array([0.02]),
            'take_profit': np.array([0.05])
        }
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position['direction'], 'long')
    
    def test_step_open_short(self):
        """测试开空仓"""
        self.env.reset()
        
        action = {
            'direction': 2,  # 做空
            'position_size': np.array([0.3]),
            'stop_loss': np.array([0.02]),
            'take_profit': np.array([0.05])
        }
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertIsNotNone(self.env.position)
        self.assertEqual(self.env.position['direction'], 'short')
    
    def test_episode_termination(self):
        """测试episode终止条件"""
        self.env.reset()
        
        # 执行到最后一步
        for _ in range(self.num_steps - 1):
            action = {
                'direction': 0,
                'position_size': np.array([0.3]),
                'stop_loss': np.array([0.02]),
                'take_profit': np.array([0.05])
            }
            obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertTrue(terminated or truncated)


class TestPolicyNetwork(unittest.TestCase):
    """测试策略网络"""
    
    def setUp(self):
        """设置测试"""
        self.state_dim = 263
        self.hidden_dim = 512
        self.batch_size = 4
        
        self.policy = PolicyNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_initialization(self):
        """测试网络初始化"""
        self.assertEqual(self.policy.state_dim, self.state_dim)
        self.assertIsNotNone(self.policy.shared)
        self.assertIsNotNone(self.policy.direction_head)
    
    def test_forward(self):
        """测试前向传播"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        direction_dist, position_dist, stop_loss_logit, take_profit_logit = self.policy(state)
        
        self.assertIsNotNone(direction_dist)
        self.assertIsNotNone(position_dist)
        self.assertEqual(stop_loss_logit.shape, (self.batch_size, 1))
        self.assertEqual(take_profit_logit.shape, (self.batch_size, 1))
    
    def test_sample_action(self):
        """测试动作采样"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        action, log_prob, entropy = self.policy.sample_action(state)
        
        self.assertIn('direction', action)
        self.assertIn('position_size', action)
        self.assertIn('stop_loss', action)
        self.assertIn('take_profit', action)
        self.assertEqual(log_prob.shape[0], self.batch_size)
        self.assertEqual(entropy.shape[0], self.batch_size)
    
    def test_deterministic_action(self):
        """测试确定性动作"""
        state = torch.randn(1, self.state_dim)
        
        action1, _, _ = self.policy.sample_action(state, deterministic=True)
        action2, _, _ = self.policy.sample_action(state, deterministic=True)
        
        # 确定性策略应该产生相同的动作
        self.assertEqual(action1['direction'].item(), action2['direction'].item())


class TestValueNetwork(unittest.TestCase):
    """测试价值网络"""
    
    def setUp(self):
        """设置测试"""
        self.state_dim = 263
        self.hidden_dim = 512
        self.batch_size = 4
        
        self.value_net = ValueNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        )
    
    def test_forward(self):
        """测试前向传播"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        value = self.value_net(state)
        
        self.assertEqual(value.shape, (self.batch_size, 1))


class TestActorCritic(unittest.TestCase):
    """测试Actor-Critic模型"""
    
    def setUp(self):
        """设置测试"""
        self.state_dim = 263
        self.hidden_dim = 512
        self.batch_size = 4
        
        self.model = ActorCritic(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            share_features=False
        )
    
    def test_forward(self):
        """测试前向传播"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        action, log_prob, entropy, value = self.model(state)
        
        self.assertIn('direction', action)
        self.assertEqual(log_prob.shape[0], self.batch_size)
        self.assertEqual(entropy.shape[0], self.batch_size)
        self.assertEqual(value.shape, (self.batch_size, 1))
    
    def test_evaluate(self):
        """测试动作评估"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        # 先采样动作
        action, _, _, _ = self.model(state)
        
        # 评估动作
        log_prob, entropy, value = self.model.evaluate(state, action)
        
        self.assertEqual(log_prob.shape[0], self.batch_size)
        self.assertEqual(entropy.shape[0], self.batch_size)
        self.assertEqual(value.shape, (self.batch_size, 1))
    
    def test_get_value(self):
        """测试获取状态价值"""
        state = torch.randn(self.batch_size, self.state_dim)
        
        value = self.model.get_value(state)
        
        self.assertEqual(value.shape, (self.batch_size, 1))
    
    def test_parameter_count(self):
        """测试参数计数"""
        param_count = self.model.count_parameters()
        
        self.assertGreater(param_count, 0)


class TestRolloutBuffer(unittest.TestCase):
    """测试经验缓冲区"""
    
    def setUp(self):
        """设置测试"""
        self.buffer_size = 100
        self.state_dim = 263
        
        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size,
            state_dim=self.state_dim,
            gamma=0.99,
            gae_lambda=0.95
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.buffer.buffer_size, self.buffer_size)
        self.assertEqual(self.buffer.state_dim, self.state_dim)
        self.assertEqual(self.buffer.ptr, 0)
    
    def test_add(self):
        """测试添加经验"""
        state = np.random.randn(self.state_dim)
        action = {
            'direction': 1,
            'position_size': np.array([0.3]),
            'stop_loss': np.array([0.02]),
            'take_profit': np.array([0.05])
        }
        
        self.buffer.add(state, action, 0.5, 0.1, -0.2, False)
        
        self.assertEqual(self.buffer.ptr, 1)
    
    def test_finish_path(self):
        """测试完成轨迹"""
        # 添加一些经验
        for i in range(20):
            state = np.random.randn(self.state_dim)
            action = {
                'direction': np.random.randint(0, 3),
                'position_size': np.random.rand(1),
                'stop_loss': np.random.uniform(0.001, 0.05, 1),
                'take_profit': np.random.uniform(0.002, 0.10, 1)
            }
            done = (i == 19)
            
            self.buffer.add(state, action, np.random.randn(), np.random.randn(), 
                          np.random.randn(), done)
        
        self.buffer.finish_path(last_value=0.0)
        
        # 检查优势和回报是否计算
        self.assertIsNotNone(self.buffer.advantages)
        self.assertIsNotNone(self.buffer.returns)
    
    def test_reset(self):
        """测试重置"""
        self.buffer.add(
            np.random.randn(self.state_dim),
            {'direction': 1, 'position_size': np.array([0.3]),
             'stop_loss': np.array([0.02]), 'take_profit': np.array([0.05])},
            0.5, 0.1, -0.2, False
        )
        
        self.buffer.reset()
        
        self.assertEqual(self.buffer.ptr, 0)
        self.assertEqual(self.buffer.path_start_idx, 0)


class TestMiniBatchSampler(unittest.TestCase):
    """测试小批量采样器"""
    
    def setUp(self):
        """设置测试"""
        self.batch_size = 256
        self.mini_batch_size = 64
        
        self.sampler = MiniBatchSampler(
            batch_size=self.batch_size,
            mini_batch_size=self.mini_batch_size,
            shuffle=True
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.sampler.batch_size, self.batch_size)
        self.assertEqual(self.sampler.mini_batch_size, self.mini_batch_size)
        self.assertEqual(self.sampler.num_mini_batches, 4)
    
    def test_sample(self):
        """测试采样"""
        # 创建模拟数据
        data = {
            'states': torch.randn(self.batch_size, 263),
            'actions': {
                'direction': torch.randint(0, 3, (self.batch_size,)),
                'position_size': torch.rand(self.batch_size, 1),
                'stop_loss': torch.rand(self.batch_size, 1),
                'take_profit': torch.rand(self.batch_size, 1)
            },
            'old_log_probs': torch.randn(self.batch_size),
            'advantages': torch.randn(self.batch_size),
            'returns': torch.randn(self.batch_size),
            'values': torch.randn(self.batch_size)
        }
        
        mini_batches = self.sampler.sample(data)
        
        self.assertEqual(len(mini_batches), self.sampler.num_mini_batches)
        
        for batch in mini_batches:
            self.assertEqual(batch['states'].shape[0], self.mini_batch_size)


class TestEpisodeBuffer(unittest.TestCase):
    """测试Episode缓冲区"""
    
    def setUp(self):
        """设置测试"""
        self.buffer = EpisodeBuffer(max_episodes=10)
    
    def test_add_episode(self):
        """测试添加episode"""
        states = [np.random.randn(263) for _ in range(20)]
        actions = [{
            'direction': np.random.randint(0, 3),
            'position_size': np.random.rand(1),
            'stop_loss': np.random.uniform(0.001, 0.05, 1),
            'take_profit': np.random.uniform(0.002, 0.10, 1)
        } for _ in range(20)]
        rewards = [np.random.randn() for _ in range(20)]
        infos = [{} for _ in range(20)]
        
        self.buffer.add_episode(states, actions, rewards, infos)
        
        self.assertEqual(len(self.buffer.episodes), 1)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        # 添加多个episodes
        for _ in range(5):
            states = [np.random.randn(263) for _ in range(20)]
            actions = [{
                'direction': np.random.randint(0, 3),
                'position_size': np.random.rand(1),
                'stop_loss': np.random.uniform(0.001, 0.05, 1),
                'take_profit': np.random.uniform(0.002, 0.10, 1)
            } for _ in range(20)]
            rewards = [np.random.randn() for _ in range(20)]
            infos = [{} for _ in range(20)]
            
            self.buffer.add_episode(states, actions, rewards, infos)
        
        stats = self.buffer.get_statistics()
        
        self.assertIn('mean_reward', stats)
        self.assertIn('std_reward', stats)
        self.assertIn('num_episodes', stats)
        self.assertEqual(stats['num_episodes'], 5)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestTradingEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestPolicyNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestValueNetwork))
    suite.addTests(loader.loadTestsFromTestCase(TestActorCritic))
    suite.addTests(loader.loadTestsFromTestCase(TestRolloutBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestMiniBatchSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestEpisodeBuffer))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)