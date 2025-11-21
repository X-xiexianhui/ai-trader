"""
PPO模型实现
整合Actor-Critic网络、经验缓冲区、动作采样和交易环境
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List, Any
from dataclasses import dataclass
from collections import deque


# ==================== 动作空间相关类 ====================

@dataclass
class Action:
    """交易动作数据类"""
    direction: int  # 0=平仓/空仓, 1=做多, 2=做空
    position_size: float  # 仓位比例 [0, 1]
    stop_loss: float  # 止损百分比 [0.001, 0.05]
    take_profit: float  # 止盈百分比 [0.002, 0.10]
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'direction': self.direction,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    def to_array(self) -> np.ndarray:
        """转换为数组"""
        return np.array([
            self.direction,
            self.position_size,
            self.stop_loss,
            self.take_profit
        ], dtype=np.float32)


class ActionSpace:
    """交易环境动作空间"""
    
    def __init__(self):
        """初始化动作空间"""
        self.n_directions = 3
        self.direction_names = ['平仓/空仓', '做多', '做空']
        self.position_size_range = (0.0, 1.0)
        self.stop_loss_range = (0.001, 0.05)
        self.take_profit_range = (0.002, 0.10)
        self.discrete_dim = 1
        self.continuous_dim = 3
        self.total_dim = self.discrete_dim + self.continuous_dim
    
    def sample(self) -> Action:
        """随机采样一个动作"""
        direction = np.random.randint(0, self.n_directions)
        position_size = np.random.uniform(*self.position_size_range)
        stop_loss = np.random.uniform(*self.stop_loss_range)
        take_profit = np.random.uniform(*self.take_profit_range)
        
        return Action(
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    def create_action(
        self,
        direction: int,
        position_size: float,
        stop_loss: float,
        take_profit: float
    ) -> Action:
        """创建动作并进行约束检查"""
        direction = int(np.clip(direction, 0, self.n_directions - 1))
        position_size = float(np.clip(position_size, *self.position_size_range))
        stop_loss = float(np.clip(stop_loss, *self.stop_loss_range))
        take_profit = float(np.clip(take_profit, *self.take_profit_range))
        
        return Action(
            direction=direction,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )


# ==================== 状态空间相关类 ====================

class StateSpace:
    """交易环境状态空间"""
    
    def __init__(self):
        """初始化状态空间"""
        self.transformer_dim = 256
        self.position_dim = 4
        self.risk_dim = 3
        self.total_dim = self.transformer_dim + self.position_dim + self.risk_dim
        self.initial_balance = 100000.0
        self.max_holding_time = 100
        self.max_leverage = 10.0
    
    def create_state(
        self,
        transformer_state: np.ndarray,
        position_size: float,
        entry_price: float,
        current_price: float,
        holding_time: int,
        unrealized_pnl: float,
        account_balance: float,
        leverage: float,
        max_drawdown: float
    ) -> np.ndarray:
        """创建完整的状态向量"""
        assert transformer_state.shape[-1] == self.transformer_dim
        
        position_info = self._normalize_position_info(
            position_size, entry_price, current_price, holding_time, unrealized_pnl, account_balance
        )
        
        risk_params = self._normalize_risk_params(
            account_balance, leverage, max_drawdown
        )
        
        state = np.concatenate([
            transformer_state.flatten(),
            position_info,
            risk_params
        ])
        
        assert state.shape[0] == self.total_dim
        return state.astype(np.float32)
    
    def _normalize_position_info(
        self,
        position_size: float,
        entry_price: float,
        current_price: float,
        holding_time: int,
        unrealized_pnl: float,
        account_balance: float
    ) -> np.ndarray:
        """归一化持仓信息"""
        norm_position_size = np.clip(position_size, -1.0, 1.0)
        
        if abs(position_size) < 1e-6:
            norm_entry_price = 0.0
        else:
            norm_entry_price = (entry_price - current_price) / (current_price + 1e-8)
            norm_entry_price = np.clip(norm_entry_price, -1.0, 1.0)
        
        if holding_time <= 0:
            norm_holding_time = 0.0
        else:
            norm_holding_time = np.log(1 + holding_time) / np.log(1 + self.max_holding_time)
            norm_holding_time = np.clip(norm_holding_time, 0.0, 1.0)
        
        if abs(position_size) < 1e-6:
            norm_unrealized_pnl = 0.0
        else:
            pnl_ratio = unrealized_pnl / (account_balance + 1e-8)
            norm_unrealized_pnl = np.tanh(pnl_ratio)
        
        return np.array([
            norm_position_size,
            norm_entry_price,
            norm_holding_time,
            norm_unrealized_pnl
        ], dtype=np.float32)
    
    def _normalize_risk_params(
        self,
        account_balance: float,
        leverage: float,
        max_drawdown: float
    ) -> np.ndarray:
        """归一化风险参数"""
        balance_ratio = account_balance / self.initial_balance
        norm_balance = np.log(balance_ratio + 1e-8)
        norm_balance = np.clip(norm_balance, -3.0, 3.0) / 3.0
        
        norm_leverage = leverage / self.max_leverage
        norm_leverage = np.clip(norm_leverage, 0.0, 1.0)
        
        norm_max_drawdown = np.clip(max_drawdown, 0.0, 1.0)
        
        return np.array([
            norm_balance,
            norm_leverage,
            norm_max_drawdown
        ], dtype=np.float32)
    
    def create_initial_state(self, transformer_state: np.ndarray) -> np.ndarray:
        """创建初始状态（空仓状态）"""
        return self.create_state(
            transformer_state=transformer_state,
            position_size=0.0,
            entry_price=0.0,
            current_price=1.0,
            holding_time=0,
            unrealized_pnl=0.0,
            account_balance=self.initial_balance,
            leverage=1.0,
            max_drawdown=0.0
        )
    
    @property
    def dim(self) -> int:
        """返回状态空间的维度"""
        return self.total_dim


# ==================== 奖励函数相关类 ====================

class RewardFunction:
    """交易环境奖励函数"""
    
    def __init__(
        self,
        profit_weight: float = 1.0,
        risk_weight: float = 0.5,
        stability_weight: float = 0.2,
        drawdown_threshold: float = 0.10,
        drawdown_penalty: float = 5.0,
        max_holding_time: int = 100,
        time_penalty: float = 0.01,
        max_leverage: float = 10.0,
        leverage_penalty: float = 1.0,
        sharpe_weight: float = 0.1,
        streak_bonus: float = 0.05,
        streak_penalty: float = 0.1
    ):
        """初始化奖励函数"""
        self.profit_weight = profit_weight
        self.risk_weight = risk_weight
        self.stability_weight = stability_weight
        self.drawdown_threshold = drawdown_threshold
        self.drawdown_penalty = drawdown_penalty
        self.max_holding_time = max_holding_time
        self.time_penalty = time_penalty
        self.max_leverage = max_leverage
        self.leverage_penalty = leverage_penalty
        self.sharpe_weight = sharpe_weight
        self.streak_bonus = streak_bonus
        self.streak_penalty = streak_penalty
        
        self.returns_history = deque(maxlen=100)
        self.correct_streak = 0
        self.wrong_streak = 0
    
    def calculate_reward(
        self,
        realized_pnl: float,
        account_balance: float,
        drawdown: float,
        holding_time: int,
        leverage: float,
        is_position_closed: bool = False,
        price_direction_correct: Optional[bool] = None
    ) -> Dict[str, float]:
        """计算总奖励"""
        profit_reward = self._calculate_profit_reward(
            realized_pnl, account_balance, is_position_closed
        )
        
        risk_reward = self._calculate_risk_reward(
            drawdown, holding_time, leverage
        )
        
        stability_reward = self._calculate_stability_reward(
            realized_pnl, account_balance, is_position_closed, price_direction_correct
        )
        
        total_reward = (
            self.profit_weight * profit_reward +
            self.risk_weight * risk_reward +
            self.stability_weight * stability_reward
        )
        
        return {
            'total': total_reward,
            'profit': profit_reward,
            'risk': risk_reward,
            'stability': stability_reward
        }
    
    def _calculate_profit_reward(
        self,
        realized_pnl: float,
        account_balance: float,
        is_position_closed: bool
    ) -> float:
        """计算盈利奖励"""
        if not is_position_closed:
            return 0.0
        
        profit_ratio = realized_pnl / (account_balance + 1e-8)
        profit_reward = np.tanh(profit_ratio * 10)
        self.returns_history.append(profit_ratio)
        
        return float(profit_reward)
    
    def _calculate_risk_reward(
        self,
        drawdown: float,
        holding_time: int,
        leverage: float
    ) -> float:
        """计算风险控制奖励"""
        risk_reward = 0.0
        
        if drawdown > self.drawdown_threshold:
            excess_drawdown = drawdown - self.drawdown_threshold
            risk_reward -= excess_drawdown * self.drawdown_penalty
        
        if holding_time > self.max_holding_time:
            excess_time = holding_time - self.max_holding_time
            risk_reward -= (excess_time / self.max_holding_time) * self.time_penalty
        
        if leverage > self.max_leverage:
            excess_leverage = leverage - self.max_leverage
            risk_reward -= (excess_leverage / self.max_leverage) * self.leverage_penalty
        
        return float(risk_reward)
    
    def _calculate_stability_reward(
        self,
        realized_pnl: float,
        account_balance: float,
        is_position_closed: bool,
        price_direction_correct: Optional[bool]
    ) -> float:
        """计算稳定性奖励"""
        stability_reward = 0.0
        
        if is_position_closed and len(self.returns_history) >= 10:
            sharpe_ratio = self._calculate_sharpe_ratio()
            stability_reward += sharpe_ratio * self.sharpe_weight
        
        if price_direction_correct is not None:
            streak_reward = self._calculate_streak_reward(price_direction_correct)
            stability_reward += streak_reward
        
        return float(stability_reward)
    
    def _calculate_sharpe_ratio(self) -> float:
        """计算滚动窗口夏普率"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
        
        sharpe = (mean_return / std_return) * np.sqrt(252)
        sharpe = np.clip(sharpe, -5.0, 5.0)
        
        return float(sharpe)
    
    def _calculate_streak_reward(self, is_correct: bool) -> float:
        """计算连续正确/错误的奖励/惩罚"""
        if is_correct:
            self.correct_streak += 1
            self.wrong_streak = 0
            streak_reward = min(self.correct_streak, 5) * self.streak_bonus
        else:
            self.wrong_streak += 1
            self.correct_streak = 0
            streak_reward = -min(self.wrong_streak, 5) * self.streak_penalty
        
        return float(streak_reward)
    
    def reset(self):
        """重置奖励函数的内部状态"""
        self.returns_history.clear()
        self.correct_streak = 0
        self.wrong_streak = 0


# ==================== 交易环境相关类 ====================

@dataclass
class Position:
    """持仓信息"""
    direction: int  # 0=空仓, 1=多头, 2=空头
    size: float  # 仓位大小
    entry_price: float  # 入场价格
    entry_time: int  # 入场时间步
    stop_loss: float  # 止损价格
    take_profit: float  # 止盈价格
    unrealized_pnl: float = 0.0  # 未实现盈亏
    
    @property
    def is_long(self) -> bool:
        return self.direction == 1
    
    @property
    def is_short(self) -> bool:
        return self.direction == 2
    
    @property
    def is_empty(self) -> bool:
        return self.direction == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'direction': self.direction,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'unrealized_pnl': self.unrealized_pnl
        }


class TradingEnvironment:
    """交易环境 - 符合Gym接口规范"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        transformer_states: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.0002,
        slippage: float = 0.0001,
        max_position_size: float = 1.0,
        reward_config: Optional[Dict] = None
    ):
        """初始化交易环境"""
        self.data = data.reset_index(drop=True)
        self.transformer_states = transformer_states
        self.n_steps = len(data)
        
        assert len(transformer_states) == self.n_steps
        
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        self.state_space = StateSpace()
        self.action_space = ActionSpace()
        self.reward_function = RewardFunction(**(reward_config or {}))
        
        self.current_step = 0
        self.account_balance = initial_balance
        self.position: Optional[Position] = None
        self.max_balance = initial_balance
        self.max_drawdown = 0.0
        
        self.trade_history = []
        self.balance_history = [initial_balance]
        self.done = False
    
    def reset(self) -> np.ndarray:
        """重置环境到初始状态"""
        self.current_step = 0
        self.account_balance = self.initial_balance
        self.position = None
        self.max_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        self.trade_history = []
        self.balance_history = [self.initial_balance]
        self.done = False
        
        self.reward_function.reset()
        
        transformer_state = self.transformer_states[self.current_step]
        state = self.state_space.create_initial_state(transformer_state)
        
        return state
    
    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一个动作"""
        if self.done:
            raise RuntimeError("环境已结束，请先调用reset()")
        
        current_price = self._get_current_price()
        realized_pnl, is_position_closed = self._execute_action(action, current_price)
        
        self.account_balance += realized_pnl
        self.balance_history.append(self.account_balance)
        
        self.max_balance = max(self.max_balance, self.account_balance)
        current_drawdown = (self.max_balance - self.account_balance) / self.max_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        if self.position and not self.position.is_empty:
            self._check_stop_loss_take_profit(current_price)
        
        self.current_step += 1
        
        self.done = (
            self.current_step >= self.n_steps - 1 or
            self.account_balance <= 0
        )
        
        holding_time = 0
        leverage = 1.0
        if self.position and not self.position.is_empty:
            holding_time = self.current_step - self.position.entry_time
            leverage = abs(self.position.size) / (self.account_balance / self.initial_balance)
        
        price_direction_correct = None
        if is_position_closed and len(self.trade_history) > 0:
            last_trade = self.trade_history[-1]
            price_direction_correct = last_trade['pnl'] > 0
        
        reward_dict = self.reward_function.calculate_reward(
            realized_pnl=realized_pnl,
            account_balance=self.account_balance,
            drawdown=current_drawdown,
            holding_time=holding_time,
            leverage=leverage,
            is_position_closed=is_position_closed,
            price_direction_correct=price_direction_correct
        )
        
        reward = reward_dict['total']
        next_state = self._get_current_state()
        
        info = {
            'step': self.current_step,
            'balance': self.account_balance,
            'max_drawdown': self.max_drawdown,
            'realized_pnl': realized_pnl,
            'is_position_closed': is_position_closed,
            'reward_breakdown': reward_dict,
            'position': self.position.to_dict() if self.position else None
        }
        
        return next_state, reward, self.done, info
    
    def _execute_action(self, action: Action, current_price: float) -> Tuple[float, bool]:
        """执行交易动作"""
        realized_pnl = 0.0
        is_position_closed = False
        
        if action.direction == 0:
            if self.position and not self.position.is_empty:
                realized_pnl = self._close_position(current_price)
                is_position_closed = True
        
        elif action.direction in [1, 2]:
            if self.position and not self.position.is_empty:
                if self.position.direction != action.direction:
                    realized_pnl = self._close_position(current_price)
                    is_position_closed = True
                    self._open_position(action, current_price)
                else:
                    self._adjust_position(action, current_price)
            else:
                self._open_position(action, current_price)
        
        return realized_pnl, is_position_closed
    
    def _open_position(self, action: Action, current_price: float):
        """开仓"""
        position_size = min(action.position_size, self.max_position_size)
        
        if action.direction == 1:
            stop_loss_price = current_price * (1 - action.stop_loss)
            take_profit_price = current_price * (1 + action.take_profit)
        else:
            stop_loss_price = current_price * (1 + action.stop_loss)
            take_profit_price = current_price * (1 - action.take_profit)
        
        entry_price = current_price * (1 + self.slippage if action.direction == 1 else 1 - self.slippage)
        
        cost = self.account_balance * position_size * self.transaction_cost
        self.account_balance -= cost
        
        self.position = Position(
            direction=action.direction,
            size=position_size,
            entry_price=entry_price,
            entry_time=self.current_step,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price
        )
    
    def _close_position(self, current_price: float) -> float:
        """平仓"""
        if not self.position or self.position.is_empty:
            return 0.0
        
        exit_price = current_price * (1 - self.slippage if self.position.is_long else 1 + self.slippage)
        
        if self.position.is_long:
            pnl_ratio = (exit_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_ratio = (self.position.entry_price - exit_price) / self.position.entry_price
        
        realized_pnl = self.account_balance * self.position.size * pnl_ratio
        
        cost = self.account_balance * self.position.size * self.transaction_cost
        realized_pnl -= cost
        
        self.trade_history.append({
            'entry_time': self.position.entry_time,
            'exit_time': self.current_step,
            'direction': 'long' if self.position.is_long else 'short',
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'size': self.position.size,
            'pnl': realized_pnl,
            'pnl_ratio': pnl_ratio,
            'holding_time': self.current_step - self.position.entry_time
        })
        
        self.position = None
        
        return realized_pnl
    
    def _adjust_position(self, action: Action, current_price: float):
        """调整持仓大小"""
        if not self.position or self.position.is_empty:
            return
        
        new_size = min(action.position_size, self.max_position_size)
        
        if new_size < self.position.size:
            size_diff = self.position.size - new_size
            if self.position.is_long:
                pnl_ratio = (current_price - self.position.entry_price) / self.position.entry_price
            else:
                pnl_ratio = (self.position.entry_price - current_price) / self.position.entry_price
            
            partial_pnl = self.account_balance * size_diff * pnl_ratio
            self.account_balance += partial_pnl
        
        self.position.size = new_size
        
        if self.position.is_long:
            self.position.stop_loss = current_price * (1 - action.stop_loss)
            self.position.take_profit = current_price * (1 + action.take_profit)
        else:
            self.position.stop_loss = current_price * (1 + action.stop_loss)
            self.position.take_profit = current_price * (1 - action.take_profit)
    
    def _check_stop_loss_take_profit(self, current_price: float):
        """检查止损止盈"""
        if not self.position or self.position.is_empty:
            return
        
        triggered = False
        
        if self.position.is_long:
            if current_price <= self.position.stop_loss:
                triggered = True
            elif current_price >= self.position.take_profit:
                triggered = True
        else:
            if current_price >= self.position.stop_loss:
                triggered = True
            elif current_price <= self.position.take_profit:
                triggered = True
        
        if triggered:
            self._close_position(current_price)
    
    def _get_current_price(self) -> float:
        """获取当前价格（收盘价）"""
        return float(self.data.loc[self.current_step, 'Close'])
    
    def _get_current_state(self) -> np.ndarray:
        """获取当前状态向量"""
        transformer_state = self.transformer_states[self.current_step]
        current_price = self._get_current_price()
        
        if self.position and not self.position.is_empty:
            position_size = self.position.size if self.position.is_long else -self.position.size
            entry_price = self.position.entry_price
            holding_time = self.current_step - self.position.entry_time
            
            if self.position.is_long:
                pnl_ratio = (current_price - entry_price) / entry_price
            else:
                pnl_ratio = (entry_price - current_price) / entry_price
            unrealized_pnl = self.account_balance * self.position.size * pnl_ratio
        else:
            position_size = 0.0
            entry_price = 0.0
            holding_time = 0
            unrealized_pnl = 0.0
        
        leverage = abs(position_size) / (self.account_balance / self.initial_balance) if position_size != 0 else 1.0
        
        state = self.state_space.create_state(
            transformer_state=transformer_state,
            position_size=position_size,
            entry_price=entry_price,
            current_price=current_price,
            holding_time=holding_time,
            unrealized_pnl=unrealized_pnl,
            account_balance=self.account_balance,
            leverage=leverage,
            max_drawdown=self.max_drawdown
        )
        
        return state
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """获取交易统计信息"""
        if len(self.trade_history) == 0:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'max_pnl': 0.0,
                'min_pnl': 0.0
            }
        
        trades_df = pd.DataFrame(self.trade_history)
        
        return {
            'n_trades': len(trades_df),
            'win_rate': (trades_df['pnl'] > 0).mean(),
            'avg_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum(),
            'max_pnl': trades_df['pnl'].max(),
            'min_pnl': trades_df['pnl'].min(),
            'avg_holding_time': trades_df['holding_time'].mean(),
            'long_trades': (trades_df['direction'] == 'long').sum(),
            'short_trades': (trades_df['direction'] == 'short').sum()
        }


# ==================== PPO网络相关类 ====================

class ActorNetwork(nn.Module):
    """
    PPO策略网络（Actor）
    
    输出混合动作空间：
    - 离散动作：direction ∈ {0, 1, 2}（平仓/做多/做空）
    - 连续动作：position_size, stop_loss, take_profit
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super(ActorNetwork, self).__init__()
        
        # 共享层
        self.shared_layers = nn.Sequential(
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
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.shared_layers(state)
        
        direction_logits = self.direction_head(features)
        
        position_mean = torch.tanh(self.position_mean(features))
        position_mean = (position_mean + 1) / 2
        position_std = torch.exp(self.position_log_std).expand_as(position_mean)
        
        stop_loss = torch.sigmoid(self.stop_loss_head(features))
        stop_loss = 0.001 + stop_loss * (0.05 - 0.001)
        
        take_profit = torch.sigmoid(self.take_profit_head(features))
        take_profit = 0.002 + take_profit * (0.10 - 0.002)
        
        return {
            'direction_logits': direction_logits,
            'position_mean': position_mean,
            'position_std': position_std,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """采样动作"""
        action_params = self.forward(state)
        
        # 离散动作采样
        direction_dist = Categorical(logits=action_params['direction_logits'])
        if deterministic:
            direction = torch.argmax(action_params['direction_logits'], dim=-1)
        else:
            direction = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction)
        direction_entropy = direction_dist.entropy()
        
        # 连续动作采样（position_size）
        position_dist = Normal(
            action_params['position_mean'],
            action_params['position_std']
        )
        if deterministic:
            position_size = action_params['position_mean']
        else:
            position_size = position_dist.sample()
        position_size = torch.clamp(position_size, 0.0, 1.0)
        position_log_prob = position_dist.log_prob(position_size).sum(dim=-1)
        position_entropy = position_dist.entropy().sum(dim=-1)
        
        # 连续动作（stop_loss和take_profit）
        stop_loss = action_params['stop_loss'].squeeze(-1)
        take_profit = action_params['take_profit'].squeeze(-1)
        
        actions = {
            'direction': direction,
            'position_size': position_size.squeeze(-1),
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        total_log_prob = direction_log_prob + position_log_prob
        total_entropy = direction_entropy + position_entropy
        
        return actions, total_log_prob, total_entropy
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率和熵"""
        action_params = self.forward(state)
        
        direction_dist = Categorical(logits=action_params['direction_logits'])
        direction_log_prob = direction_dist.log_prob(actions['direction'])
        direction_entropy = direction_dist.entropy()
        
        position_dist = Normal(
            action_params['position_mean'],
            action_params['position_std']
        )
        position_log_prob = position_dist.log_prob(
            actions['position_size'].unsqueeze(-1)
        ).sum(dim=-1)
        position_entropy = position_dist.entropy().sum(dim=-1)
        
        total_log_prob = direction_log_prob + position_log_prob
        total_entropy = direction_entropy + position_entropy
        
        return total_log_prob, total_entropy


class CriticNetwork(nn.Module):
    """
    PPO价值网络（Critic）
    估计状态价值函数 V(s)
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(state)


class ExperienceBuffer:
    """
    经验缓冲区
    存储PPO训练所需的所有经验数据
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: Dict[str, float],
        reward: float,
        done: bool,
        value: float,
        log_prob: float
    ):
        """添加一条经验"""
        if self.size >= self.capacity:
            raise RuntimeError(f"缓冲区已满（容量：{self.capacity}）")
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.size += 1
    
    def compute_gae(
        self,
        next_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """计算广义优势估计（GAE）"""
        advantages = np.zeros(self.size, dtype=np.float32)
        returns = np.zeros(self.size, dtype=np.float32)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(self.size)):
            if self.dones[t]:
                next_val = 0
                gae = 0
            
            delta = self.rewards[t] + gamma * next_val - self.values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
            next_val = self.values[t]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """获取训练批次"""
        if len(self.advantages) == 0:
            raise RuntimeError("请先调用compute_gae()计算优势函数")
        
        indices = np.arange(self.size)
        if shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]
            batch = self._get_batch_by_indices(batch_indices)
            batches.append(batch)
        
        return batches
    
    def _get_batch_by_indices(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """根据索引提取批次数据"""
        states = torch.FloatTensor(np.array([self.states[i] for i in indices]))
        
        actions = {
            'direction': torch.LongTensor([self.actions[i]['direction'] for i in indices]),
            'position_size': torch.FloatTensor([self.actions[i]['position_size'] for i in indices]),
            'stop_loss': torch.FloatTensor([self.actions[i]['stop_loss'] for i in indices]),
            'take_profit': torch.FloatTensor([self.actions[i]['take_profit'] for i in indices])
        }
        
        old_log_probs = torch.FloatTensor([self.log_probs[i] for i in indices])
        advantages = torch.FloatTensor([self.advantages[i] for i in indices])
        returns = torch.FloatTensor([self.returns[i] for i in indices])
        old_values = torch.FloatTensor([self.values[i] for i in indices])
        
        return {
            'states': states,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'advantages': advantages,
            'returns': returns,
            'old_values': old_values
        }
    
    def __len__(self) -> int:
        return self.size


class PPOModel:
    """
    PPO模型
    整合Actor-Critic网络和经验缓冲区，提供统一接口
    """
    
    def __init__(
        self,
        state_dim: int = 263,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化PPO模型
        
        Args:
            state_dim: 状态维度（默认263 = 256 Transformer + 7 持仓/风险信息）
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            device: 计算设备
        """
        self.device = device
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 创建网络
        self.actor = ActorNetwork(state_dim, hidden_dim, dropout).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dim, dropout).to(device)
        
        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=lr_actor,
            weight_decay=1e-5
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=lr_critic,
            weight_decay=1e-5
        )
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[Dict[str, float], float, float]:
        """
        选择动作
        
        Args:
            state: 状态数组
            deterministic: 是否使用确定性策略
            
        Returns:
            (actions, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 获取动作
            actions, log_prob, _ = self.actor.get_action(state_tensor, deterministic)
            
            # 获取价值
            value = self.critic(state_tensor)
            
            # 转换为numpy
            actions_np = {
                k: v.cpu().numpy()[0] if isinstance(v, torch.Tensor) else v
                for k, v in actions.items()
            }
            log_prob_np = log_prob.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0, 0]
            
            return actions_np, log_prob_np, value_np
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'hidden_dim': self.hidden_dim
            }
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    def get_model_info(self) -> Dict[str, int]:
        """获取模型信息"""
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        
        return {
            'actor_parameters': actor_params,
            'critic_parameters': critic_params,
            'total_parameters': actor_params + critic_params,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim
        }