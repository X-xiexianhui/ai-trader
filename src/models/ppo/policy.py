"""
PPO策略网络和价值网络实现

实现PPO的Actor-Critic架构，包括:
- 策略网络（Actor）：输出动作分布
- 价值网络（Critic）：估计状态价值
- 混合动作空间支持（离散+连续）

Author: AI Trader Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    策略网络（Actor）
    
    输出混合动作空间的动作分布：
    - 离散动作：direction (0=平仓, 1=做多, 2=做空)
    - 连续动作：position_size, stop_loss, take_profit
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.state_dim = state_dim
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 离散动作头（direction）
        self.direction_head = nn.Linear(hidden_dim // 2, 3)
        
        # 连续动作头（position_size）
        self.position_mean = nn.Linear(hidden_dim // 2, 1)
        self.position_log_std = nn.Parameter(torch.zeros(1))
        
        # 连续动作头（stop_loss）
        self.stop_loss_head = nn.Linear(hidden_dim // 2, 1)
        
        # 连续动作头（take_profit）
        self.take_profit_head = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
        
        logger.info(f"PolicyNetwork initialized: state_dim={state_dim}, "
                   f"hidden_dim={hidden_dim}")
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[Categorical, Normal, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch, state_dim)
            
        Returns:
            direction_dist: 方向动作分布
            position_dist: 仓位大小分布
            stop_loss_logit: 止损logit
            take_profit_logit: 止盈logit
        """
        # 共享特征
        features = self.shared(state)
        
        # 离散动作分布（direction）
        direction_logits = self.direction_head(features)
        direction_dist = Categorical(logits=direction_logits)
        
        # 连续动作分布（position_size）
        position_mean = torch.sigmoid(self.position_mean(features))  # [0, 1]
        position_std = torch.exp(self.position_log_std).expand_as(position_mean)
        position_dist = Normal(position_mean, position_std)
        
        # 止损和止盈（使用sigmoid映射到指定范围）
        stop_loss_logit = self.stop_loss_head(features)
        take_profit_logit = self.take_profit_head(features)
        
        return direction_dist, position_dist, stop_loss_logit, take_profit_logit
    
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作字典
            log_prob: 对数概率
            entropy: 熵
        """
        direction_dist, position_dist, stop_loss_logit, take_profit_logit = self.forward(state)
        
        # 采样离散动作
        if deterministic:
            direction = direction_dist.probs.argmax(dim=-1)
        else:
            direction = direction_dist.sample()
        
        # 采样连续动作
        if deterministic:
            position_size = position_dist.mean
        else:
            position_size = position_dist.sample()
        position_size = torch.clamp(position_size, 0.0, 1.0)
        
        # 计算止损止盈
        stop_loss = torch.sigmoid(stop_loss_logit) * (0.05 - 0.001) + 0.001
        take_profit = torch.sigmoid(take_profit_logit) * (0.10 - 0.002) + 0.002
        
        # 组合动作
        action = {
            'direction': direction,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        # 计算对数概率
        direction_log_prob = direction_dist.log_prob(direction)
        position_log_prob = position_dist.log_prob(position_size).sum(dim=-1)
        log_prob = direction_log_prob + position_log_prob
        
        # 计算熵
        entropy = direction_dist.entropy() + position_dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy
    
    def evaluate_action(
        self,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作
        
        Args:
            state: 状态张量
            action: 动作字典
            
        Returns:
            log_prob: 对数概率
            entropy: 熵
        """
        direction_dist, position_dist, _, _ = self.forward(state)
        
        # 计算对数概率
        direction_log_prob = direction_dist.log_prob(action['direction'])
        position_log_prob = position_dist.log_prob(action['position_size']).sum(dim=-1)
        log_prob = direction_log_prob + position_log_prob
        
        # 计算熵
        entropy = direction_dist.entropy() + position_dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    价值网络（Critic）
    
    估计状态价值函数V(s)。
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
        logger.info(f"ValueNetwork initialized: state_dim={state_dim}, "
                   f"hidden_dim={hidden_dim}")
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 (batch, state_dim)
            
        Returns:
            value: 状态价值 (batch, 1)
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Actor-Critic模型
    
    组合策略网络和价值网络。
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        share_features: bool = False
    ):
        """
        初始化Actor-Critic
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            share_features: 是否共享特征提取层
        """
        super().__init__()
        
        self.share_features = share_features
        
        if share_features:
            # 共享特征提取层
            self.shared_features = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            # 策略头
            self.policy = PolicyNetwork(hidden_dim, hidden_dim, dropout)
            
            # 价值头
            self.value = ValueNetwork(hidden_dim, hidden_dim, dropout)
        else:
            # 独立网络
            self.policy = PolicyNetwork(state_dim, hidden_dim, dropout)
            self.value = ValueNetwork(state_dim, hidden_dim, dropout)
        
        logger.info(f"ActorCritic initialized: share_features={share_features}")
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 动作字典
            log_prob: 对数概率
            entropy: 熵
            value: 状态价值
        """
        if self.share_features:
            features = self.shared_features(state)
            action, log_prob, entropy = self.policy.sample_action(features, deterministic)
            value = self.value(features)
        else:
            action, log_prob, entropy = self.policy.sample_action(state, deterministic)
            value = self.value(state)
        
        return action, log_prob, entropy, value
    
    def evaluate(
        self,
        state: torch.Tensor,
        action: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估状态-动作对
        
        Args:
            state: 状态张量
            action: 动作字典
            
        Returns:
            log_prob: 对数概率
            entropy: 熵
            value: 状态价值
        """
        if self.share_features:
            features = self.shared_features(state)
            log_prob, entropy = self.policy.evaluate_action(features, action)
            value = self.value(features)
        else:
            log_prob, entropy = self.policy.evaluate_action(state, action)
            value = self.value(state)
        
        return log_prob, entropy, value
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取状态价值
        
        Args:
            state: 状态张量
            
        Returns:
            value: 状态价值
        """
        if self.share_features:
            features = self.shared_features(state)
            return self.value(features)
        else:
            return self.value(state)
    
    def count_parameters(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== PPO策略网络示例 ===")
    
    # 创建模型
    model = ActorCritic(
        state_dim=263,
        hidden_dim=512,
        share_features=False
    )
    
    print(f"\n模型参数量: {model.count_parameters():,}")
    
    # 创建模拟状态
    batch_size = 4
    state = torch.randn(batch_size, 263)
    
    # 采样动作
    print("\n采样动作...")
    action, log_prob, entropy, value = model(state, deterministic=False)
    
    print(f"Direction: {action['direction']}")
    print(f"Position size: {action['position_size'].squeeze()}")
    print(f"Stop loss: {action['stop_loss'].squeeze()}")
    print(f"Take profit: {action['take_profit'].squeeze()}")
    print(f"Log prob: {log_prob}")
    print(f"Entropy: {entropy}")
    print(f"Value: {value.squeeze()}")
    
    # 评估动作
    print("\n评估动作...")
    log_prob_eval, entropy_eval, value_eval = model.evaluate(state, action)
    print(f"Log prob (eval): {log_prob_eval}")
    print(f"Entropy (eval): {entropy_eval}")
    print(f"Value (eval): {value_eval.squeeze()}")
    
    print("\n示例完成!")