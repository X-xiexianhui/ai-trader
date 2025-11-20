"""
PPO交易环境实现

实现符合Gymnasium接口的交易环境，包括:
- 状态空间定义
- 动作空间定义
- 奖励函数
- 交易执行逻辑
- 风险管理

Author: AI Trader Team
Date: 2025-11-20
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    交易环境
    
    实现强化学习交易环境，支持做多、做空和平仓操作。
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        state_vectors: np.ndarray,
        prices: np.ndarray,
        initial_balance: float = 100000.0,
        max_position: float = 1.0,
        transaction_cost: float = 0.0002,
        slippage: float = 0.0001,
        max_leverage: float = 1.0,
        reward_scaling: float = 1.0,
        risk_penalty_weight: float = 0.5,
        stability_reward_weight: float = 0.2
    ):
        """
        初始化交易环境
        
        Args:
            state_vectors: 状态向量序列 (T, state_dim)
            prices: 价格序列 (T,)
            initial_balance: 初始资金
            max_position: 最大仓位比例
            transaction_cost: 交易成本（手续费率）
            slippage: 滑点
            max_leverage: 最大杠杆
            reward_scaling: 奖励缩放因子
            risk_penalty_weight: 风险惩罚权重
            stability_reward_weight: 稳定性奖励权重
        """
        super().__init__()
        
        self.state_vectors = state_vectors
        self.prices = prices
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_leverage = max_leverage
        self.reward_scaling = reward_scaling
        self.risk_penalty_weight = risk_penalty_weight
        self.stability_reward_weight = stability_reward_weight
        
        # 环境参数
        self.T = len(state_vectors)
        self.state_dim = state_vectors.shape[1]
        
        # 定义动作空间
        # 离散动作：direction (0=平仓, 1=做多, 2=做空)
        # 连续动作：position_size (0-1), stop_loss (0.001-0.05), take_profit (0.002-0.10)
        self.action_space = spaces.Dict({
            'direction': spaces.Discrete(3),
            'position_size': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'stop_loss': spaces.Box(low=0.001, high=0.05, shape=(1,), dtype=np.float32),
            'take_profit': spaces.Box(low=0.002, high=0.10, shape=(1,), dtype=np.float32)
        })
        
        # 定义观察空间
        # 状态向量 + 持仓信息 + 风险参数
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim + 7,),  # state_dim + 7个额外维度
            dtype=np.float32
        )
        
        # 初始化状态
        self.reset()
        
        logger.info(f"TradingEnvironment initialized: T={self.T}, "
                   f"state_dim={self.state_dim}, initial_balance={initial_balance}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观察
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置时间步
        self.current_step = 0
        
        # 重置账户状态
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        
        # 重置持仓信息
        self.position = 0.0  # 当前仓位 (-1到1，负数表示空头)
        self.entry_price = 0.0  # 入场价格
        self.holding_time = 0  # 持仓时长
        self.unrealized_pnl = 0.0  # 浮盈浮亏
        
        # 重置风险参数
        self.max_drawdown = 0.0  # 最大回撤
        self.peak_equity = self.initial_balance  # 峰值权益
        
        # 重置交易历史
        self.trade_history = []
        self.equity_history = [self.initial_balance]
        self.return_history = []
        
        # 重置止损止盈
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行一步
        
        Args:
            action: 动作字典
            
        Returns:
            observation: 新观察
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 解析动作
        direction = int(action['direction'])
        position_size = float(action['position_size'][0])
        stop_loss = float(action['stop_loss'][0])
        take_profit = float(action['take_profit'][0])
        
        # 获取当前价格
        current_price = self.prices[self.current_step]
        
        # 检查止损止盈
        if self.position != 0:
            if self._check_stop_loss(current_price) or self._check_take_profit(current_price):
                direction = 0  # 强制平仓
        
        # 执行交易
        realized_pnl = self._execute_trade(
            direction, position_size, stop_loss, take_profit, current_price
        )
        
        # 更新持仓时长
        if self.position != 0:
            self.holding_time += 1
        else:
            self.holding_time = 0
        
        # 计算浮盈浮亏
        if self.position != 0:
            self.unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        else:
            self.unrealized_pnl = 0.0
        
        # 更新权益
        self.equity = self.balance + self.unrealized_pnl
        self.equity_history.append(self.equity)
        
        # 更新最大回撤
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 计算奖励
        reward = self._calculate_reward(realized_pnl, current_drawdown)
        
        # 更新时间步
        self.current_step += 1
        
        # 检查是否结束
        terminated = self.current_step >= self.T - 1
        truncated = self.equity <= 0  # 爆仓
        
        # 获取新观察
        observation = self._get_observation()
        info = self._get_info()
        info['realized_pnl'] = realized_pnl
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trade(
        self,
        direction: int,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        current_price: float
    ) -> float:
        """
        执行交易
        
        Args:
            direction: 方向 (0=平仓, 1=做多, 2=做空)
            position_size: 仓位大小
            stop_loss: 止损百分比
            take_profit: 止盈百分比
            current_price: 当前价格
            
        Returns:
            realized_pnl: 已实现盈亏
        """
        realized_pnl = 0.0
        
        # 平仓
        if direction == 0 and self.position != 0:
            realized_pnl = self._close_position(current_price)
        
        # 开多仓
        elif direction == 1:
            if self.position < 0:  # 先平空仓
                realized_pnl = self._close_position(current_price)
            if self.position == 0:  # 开多仓
                self._open_position(1, position_size, stop_loss, take_profit, current_price)
        
        # 开空仓
        elif direction == 2:
            if self.position > 0:  # 先平多仓
                realized_pnl = self._close_position(current_price)
            if self.position == 0:  # 开空仓
                self._open_position(-1, position_size, stop_loss, take_profit, current_price)
        
        return realized_pnl
    
    def _open_position(
        self,
        direction: int,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        current_price: float
    ):
        """
        开仓
        
        Args:
            direction: 方向 (1=多, -1=空)
            position_size: 仓位大小
            stop_loss: 止损百分比
            take_profit: 止盈百分比
            current_price: 当前价格
        """
        # 限制仓位大小
        position_size = np.clip(position_size, 0.0, self.max_position)
        
        # 计算实际仓位
        self.position = direction * position_size
        
        # 考虑滑点
        if direction > 0:
            self.entry_price = current_price * (1 + self.slippage)
        else:
            self.entry_price = current_price * (1 - self.slippage)
        
        # 扣除手续费
        trade_value = abs(self.position) * self.entry_price * self.balance
        fee = trade_value * self.transaction_cost
        self.balance -= fee
        
        # 设置止损止盈
        if direction > 0:
            self.stop_loss_price = self.entry_price * (1 - stop_loss)
            self.take_profit_price = self.entry_price * (1 + take_profit)
        else:
            self.stop_loss_price = self.entry_price * (1 + stop_loss)
            self.take_profit_price = self.entry_price * (1 - take_profit)
        
        # 记录交易
        self.trade_history.append({
            'step': self.current_step,
            'action': 'open',
            'direction': 'long' if direction > 0 else 'short',
            'position_size': position_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss_price,
            'take_profit': self.take_profit_price
        })
    
    def _close_position(self, current_price: float) -> float:
        """
        平仓
        
        Args:
            current_price: 当前价格
            
        Returns:
            realized_pnl: 已实现盈亏
        """
        if self.position == 0:
            return 0.0
        
        # 考虑滑点
        if self.position > 0:
            exit_price = current_price * (1 - self.slippage)
        else:
            exit_price = current_price * (1 + self.slippage)
        
        # 计算盈亏
        if self.position > 0:
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - exit_price) / self.entry_price
        
        trade_value = abs(self.position) * self.entry_price * self.balance
        realized_pnl = trade_value * pnl_pct
        
        # 扣除手续费
        fee = abs(self.position) * exit_price * self.balance * self.transaction_cost
        realized_pnl -= fee
        
        # 更新余额
        self.balance += realized_pnl
        
        # 记录交易
        self.trade_history.append({
            'step': self.current_step,
            'action': 'close',
            'direction': 'long' if self.position > 0 else 'short',
            'exit_price': exit_price,
            'pnl': realized_pnl,
            'holding_time': self.holding_time
        })
        
        # 重置持仓
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        
        return realized_pnl
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        计算浮盈浮亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            unrealized_pnl: 浮盈浮亏
        """
        if self.position == 0:
            return 0.0
        
        if self.position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        trade_value = abs(self.position) * self.entry_price * self.balance
        return trade_value * pnl_pct
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """检查是否触发止损"""
        if self.stop_loss_price is None:
            return False
        
        if self.position > 0:
            return current_price <= self.stop_loss_price
        elif self.position < 0:
            return current_price >= self.stop_loss_price
        
        return False
    
    def _check_take_profit(self, current_price: float) -> bool:
        """检查是否触发止盈"""
        if self.take_profit_price is None:
            return False
        
        if self.position > 0:
            return current_price >= self.take_profit_price
        elif self.position < 0:
            return current_price <= self.take_profit_price
        
        return False
    
    def _calculate_reward(self, realized_pnl: float, current_drawdown: float) -> float:
        """
        计算奖励
        
        Args:
            realized_pnl: 已实现盈亏
            current_drawdown: 当前回撤
            
        Returns:
            reward: 奖励值
        """
        # 1. 盈利奖励（主导项）
        profit_reward = realized_pnl / self.initial_balance
        
        # 2. 风险控制惩罚
        risk_penalty = 0.0
        
        # 回撤惩罚
        if current_drawdown > 0.1:  # 回撤超过10%
            risk_penalty += (current_drawdown - 0.1) * 5.0
        
        # 持仓时间惩罚
        if self.holding_time > 100:  # 持仓超过100根K线
            risk_penalty += (self.holding_time - 100) * 0.001
        
        # 3. 稳定性奖励
        stability_reward = 0.0
        
        # 计算滚动夏普率（如果有足够的历史数据）
        if len(self.return_history) >= 20:
            recent_returns = self.return_history[-20:]
            if np.std(recent_returns) > 0:
                sharpe = np.mean(recent_returns) / np.std(recent_returns)
                stability_reward = sharpe * 0.1
        
        # 总奖励
        reward = (
            profit_reward -
            self.risk_penalty_weight * risk_penalty +
            self.stability_reward_weight * stability_reward
        )
        
        # 缩放奖励
        reward *= self.reward_scaling
        
        # 记录收益率
        if len(self.equity_history) >= 2:
            ret = (self.equity_history[-1] - self.equity_history[-2]) / self.equity_history[-2]
            self.return_history.append(ret)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察
        
        Returns:
            observation: 观察向量
        """
        # 状态向量
        state_vector = self.state_vectors[self.current_step]
        
        # 持仓信息（归一化）
        position_info = np.array([
            self.position,  # 已在[-1, 1]
            (self.entry_price - self.prices[self.current_step]) / self.prices[self.current_step] if self.entry_price > 0 else 0.0,
            np.log(1 + self.holding_time) / np.log(100) if self.holding_time > 0 else 0.0,
            np.tanh(self.unrealized_pnl / self.balance) if self.balance > 0 else 0.0
        ], dtype=np.float32)
        
        # 风险参数（归一化）
        risk_info = np.array([
            np.log(self.balance / self.initial_balance),
            abs(self.position),  # 当前杠杆
            self.max_drawdown
        ], dtype=np.float32)
        
        # 拼接所有信息
        observation = np.concatenate([state_vector, position_info, risk_info])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'max_drawdown': self.max_drawdown,
            'num_trades': len(self.trade_history)
        }
    
    def render(self, mode='human'):
        """渲染环境（可选实现）"""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Equity: {self.equity:.2f}, "
                  f"Position: {self.position:.2f}, "
                  f"PnL: {self.unrealized_pnl:.2f}")


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== 交易环境示例 ===")
    
    # 创建模拟数据
    T = 1000
    state_dim = 256
    state_vectors = np.random.randn(T, state_dim)
    prices = 100 + np.cumsum(np.random.randn(T) * 0.5)
    
    # 创建环境
    env = TradingEnvironment(
        state_vectors=state_vectors,
        prices=prices,
        initial_balance=100000.0
    )
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观察形状: {obs.shape}")
    print(f"初始信息: {info}")
    
    # 执行几步
    print("\n执行随机动作...")
    for i in range(5):
        action = {
            'direction': env.action_space['direction'].sample(),
            'position_size': env.action_space['position_size'].sample(),
            'stop_loss': env.action_space['stop_loss'].sample(),
            'take_profit': env.action_space['take_profit'].sample()
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, equity={info['equity']:.2f}")
        
        if terminated or truncated:
            break
    
    print("\n示例完成!")