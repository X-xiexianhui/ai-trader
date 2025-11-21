"""
训练指标和性能评估模块
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingMetrics:
    """
    训练指标计算和可视化
    """
    
    @staticmethod
    def compute_sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252 * 78  # 5分钟K线，每天78根
    ) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年的周期数
            
        Returns:
            夏普比率
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # 年化
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * np.sqrt(periods_per_year)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_std
        return sharpe
    
    @staticmethod
    def compute_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252 * 78
    ) -> float:
        """
        计算索提诺比率（只考虑下行风险）
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            periods_per_year: 每年的周期数
            
        Returns:
            索提诺比率
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        # 年化
        annualized_return = mean_return * periods_per_year
        annualized_downside_std = downside_std * np.sqrt(periods_per_year)
        
        sortino = (annualized_return - risk_free_rate) / annualized_downside_std
        return sortino
    
    @staticmethod
    def compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """
        计算最大回撤
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            最大回撤（百分比）
        """
        if len(equity_curve) == 0:
            return 0.0
        
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = np.min(drawdown)
        
        return abs(max_dd)
    
    @staticmethod
    def compute_calmar_ratio(
        returns: np.ndarray,
        equity_curve: np.ndarray,
        periods_per_year: int = 252 * 78
    ) -> float:
        """
        计算卡玛比率（年化收益/最大回撤）
        
        Args:
            returns: 收益率序列
            equity_curve: 权益曲线
            periods_per_year: 每年的周期数
            
        Returns:
            卡玛比率
        """
        if len(returns) == 0:
            return 0.0
        
        annualized_return = np.mean(returns) * periods_per_year
        max_dd = TrainingMetrics.compute_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf
        
        calmar = annualized_return / max_dd
        return calmar
    
    @staticmethod
    def compute_win_rate(returns: np.ndarray) -> float:
        """
        计算胜率
        
        Args:
            returns: 收益率序列
            
        Returns:
            胜率（0-1）
        """
        if len(returns) == 0:
            return 0.0
        
        win_rate = np.sum(returns > 0) / len(returns)
        return win_rate
    
    @staticmethod
    def compute_profit_factor(returns: np.ndarray) -> float:
        """
        计算盈亏比
        
        Args:
            returns: 收益率序列
            
        Returns:
            盈亏比
        """
        if len(returns) == 0:
            return 0.0
        
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_profit = np.sum(profits)
        total_loss = abs(np.sum(losses))
        
        if total_loss == 0:
            return np.inf if total_profit > 0 else 0.0
        
        profit_factor = total_profit / total_loss
        return profit_factor
    
    @staticmethod
    def plot_training_curves(
        history: Dict[str, List],
        save_path: Optional[str] = None
    ):
        """
        绘制训练曲线
        
        Args:
            history: 训练历史
            save_path: 保存路径
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('PPO训练曲线', fontsize=16)
        
        iterations = history['iteration']
        
        # 1. 平均奖励
        ax = axes[0, 0]
        ax.plot(iterations, history['mean_reward'], label='平均奖励')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('平均奖励')
        ax.set_title('Episode平均奖励')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Episode长度
        ax = axes[0, 1]
        ax.plot(iterations, history['mean_episode_length'], label='平均长度', color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('步数')
        ax.set_title('Episode平均长度')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. 策略损失
        ax = axes[1, 0]
        ax.plot(iterations, history['policy_loss'], label='策略损失', color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('损失')
        ax.set_title('策略损失')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. 价值损失
        ax = axes[1, 1]
        ax.plot(iterations, history['value_loss'], label='价值损失', color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('损失')
        ax.set_title('价值损失')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 5. 熵和KL散度
        ax = axes[2, 0]
        ax.plot(iterations, history['entropy'], label='熵', color='purple')
        ax2 = ax.twinx()
        ax2.plot(iterations, history['kl_divergence'], label='KL散度', color='brown', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('熵', color='purple')
        ax2.set_ylabel('KL散度', color='brown')
        ax.set_title('熵和KL散度')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 6. 裁剪比例和解释方差
        ax = axes[2, 1]
        ax.plot(iterations, history['clip_fraction'], label='裁剪比例', color='cyan')
        ax2 = ax.twinx()
        ax2.plot(iterations, history['explained_variance'], label='解释方差', color='magenta', linestyle='--')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('裁剪比例', color='cyan')
        ax2.set_ylabel('解释方差', color='magenta')
        ax.set_title('裁剪比例和解释方差')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def generate_performance_report(
        episode_rewards: List[float],
        episode_lengths: List[int],
        equity_curve: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        生成性能报告
        
        Args:
            episode_rewards: Episode奖励列表
            episode_lengths: Episode长度列表
            equity_curve: 权益曲线（可选）
            
        Returns:
            性能指标字典
        """
        if len(episode_rewards) == 0:
            return {}
        
        rewards = np.array(episode_rewards)
        
        report = {
            # 基础统计
            'n_episodes': len(episode_rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            
            # Episode长度
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            
            # 风险调整收益
            'sharpe_ratio': TrainingMetrics.compute_sharpe_ratio(rewards),
            'sortino_ratio': TrainingMetrics.compute_sortino_ratio(rewards),
            
            # 交易统计
            'win_rate': TrainingMetrics.compute_win_rate(rewards),
            'profit_factor': TrainingMetrics.compute_profit_factor(rewards)
        }
        
        # 如果提供了权益曲线
        if equity_curve is not None:
            report['max_drawdown'] = TrainingMetrics.compute_max_drawdown(equity_curve)
            report['calmar_ratio'] = TrainingMetrics.compute_calmar_ratio(
                rewards, equity_curve
            )
        
        return report


if __name__ == "__main__":
    # 测试代码
    print("测试TrainingMetrics...")
    
    # 生成模拟数据
    np.random.seed(42)
    returns = np.random.randn(1000) * 0.01 + 0.0005
    equity_curve = 100000 * (1 + returns).cumprod()
    
    # 计算指标
    sharpe = TrainingMetrics.compute_sharpe_ratio(returns)
    sortino = TrainingMetrics.compute_sortino_ratio(returns)
    max_dd = TrainingMetrics.compute_max_drawdown(equity_curve)
    calmar = TrainingMetrics.compute_calmar_ratio(returns, equity_curve)
    win_rate = TrainingMetrics.compute_win_rate(returns)
    profit_factor = TrainingMetrics.compute_profit_factor(returns)
    
    print(f"\n性能指标:")
    print(f"  夏普比率: {sharpe:.4f}")
    print(f"  索提诺比率: {sortino:.4f}")
    print(f"  最大回撤: {max_dd:.2%}")
    print(f"  卡玛比率: {calmar:.4f}")
    print(f"  胜率: {win_rate:.2%}")
    print(f"  盈亏比: {profit_factor:.4f}")
    
    # 生成性能报告
    episode_rewards = list(returns[:100])
    episode_lengths = [10] * 100
    
    report = TrainingMetrics.generate_performance_report(
        episode_rewards, episode_lengths, equity_curve[:100]
    )
    
    print(f"\n性能报告:")
    for key, value in report.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n✓ TrainingMetrics测试通过!")