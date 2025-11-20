"""
GPU加速的回测计算

使用PyTorch实现GPU加速的性能指标计算和并行回测
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.gpu_utils import get_gpu_manager

logger = logging.getLogger(__name__)


class GPUBacktestCalculator:
    """GPU加速的回测计算器"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化GPU回测计算器
        
        Args:
            device: 设备类型
        """
        self.gpu_manager = get_gpu_manager()
        
        if device is None or device == 'auto':
            self.device = self.gpu_manager.get_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"GPU回测计算器初始化，使用设备: {self.device}")
    
    def to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """转换为GPU张量"""
        return torch.tensor(data, dtype=torch.float32, device=self.device)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """转换为numpy数组"""
        return tensor.cpu().numpy()
    
    def calculate_returns(self, prices: torch.Tensor) -> torch.Tensor:
        """
        计算收益率（GPU加速）
        
        Args:
            prices: 价格序列
            
        Returns:
            torch.Tensor: 收益率序列
        """
        returns = (prices[1:] - prices[:-1]) / prices[:-1]
        # 添加第一个值为0
        returns = torch.cat([torch.zeros(1, device=self.device), returns])
        return returns
    
    def calculate_cumulative_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """
        计算累积收益率（GPU加速）
        
        Args:
            returns: 收益率序列
            
        Returns:
            torch.Tensor: 累积收益率
        """
        return torch.cumprod(1 + returns, dim=0) - 1
    
    def calculate_drawdown(self, equity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算回撤（GPU加速）
        
        Args:
            equity: 权益曲线
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (回撤序列, 最大回撤)
        """
        # 计算累积最大值
        cummax = torch.cummax(equity, dim=0)[0]
        
        # 计算回撤
        drawdown = (equity - cummax) / cummax
        
        # 最大回撤
        max_drawdown = torch.min(drawdown)
        
        return drawdown, max_drawdown
    
    def calculate_sharpe_ratio(self, returns: torch.Tensor, 
                              risk_free_rate: float = 0.02,
                              periods_per_year: int = 252 * 78) -> torch.Tensor:
        """
        计算夏普比率（GPU加速）
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            periods_per_year: 每年周期数
            
        Returns:
            torch.Tensor: 夏普比率
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = torch.mean(excess_returns)
        std_excess = torch.std(excess_returns)
        
        if std_excess == 0:
            return torch.tensor(0.0, device=self.device)
        
        sharpe = mean_excess / std_excess * torch.sqrt(torch.tensor(periods_per_year, device=self.device))
        return sharpe
    
    def calculate_sortino_ratio(self, returns: torch.Tensor,
                               risk_free_rate: float = 0.02,
                               periods_per_year: int = 252 * 78) -> torch.Tensor:
        """
        计算索提诺比率（GPU加速）
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            periods_per_year: 每年周期数
            
        Returns:
            torch.Tensor: 索提诺比率
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        mean_excess = torch.mean(excess_returns)
        
        # 只考虑负收益的标准差
        downside_returns = torch.where(excess_returns < 0, excess_returns, torch.zeros_like(excess_returns))
        downside_std = torch.std(downside_returns)
        
        if downside_std == 0:
            return torch.tensor(0.0, device=self.device)
        
        sortino = mean_excess / downside_std * torch.sqrt(torch.tensor(periods_per_year, device=self.device))
        return sortino
    
    def calculate_var(self, returns: torch.Tensor, confidence: float = 0.95) -> torch.Tensor:
        """
        计算VaR（GPU加速）
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            
        Returns:
            torch.Tensor: VaR值
        """
        return torch.quantile(returns, 1 - confidence)
    
    def calculate_cvar(self, returns: torch.Tensor, confidence: float = 0.95) -> torch.Tensor:
        """
        计算CVaR（GPU加速）
        
        Args:
            returns: 收益率序列
            confidence: 置信水平
            
        Returns:
            torch.Tensor: CVaR值
        """
        var = self.calculate_var(returns, confidence)
        cvar = torch.mean(returns[returns <= var])
        return cvar
    
    def batch_calculate_metrics(self, returns_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        批量计算指标（GPU加速）
        
        Args:
            returns_batch: 批量收益率 [batch_size, seq_len]
            
        Returns:
            Dict: 批量指标
        """
        batch_size = returns_batch.shape[0]
        
        metrics = {
            'total_return': torch.zeros(batch_size, device=self.device),
            'sharpe_ratio': torch.zeros(batch_size, device=self.device),
            'sortino_ratio': torch.zeros(batch_size, device=self.device),
            'max_drawdown': torch.zeros(batch_size, device=self.device),
            'var_95': torch.zeros(batch_size, device=self.device),
            'cvar_95': torch.zeros(batch_size, device=self.device),
        }
        
        for i in range(batch_size):
            returns = returns_batch[i]
            
            # 总收益
            metrics['total_return'][i] = torch.prod(1 + returns) - 1
            
            # 夏普比率
            metrics['sharpe_ratio'][i] = self.calculate_sharpe_ratio(returns)
            
            # 索提诺比率
            metrics['sortino_ratio'][i] = self.calculate_sortino_ratio(returns)
            
            # 最大回撤
            equity = torch.cumprod(1 + returns, dim=0)
            _, max_dd = self.calculate_drawdown(equity)
            metrics['max_drawdown'][i] = max_dd
            
            # VaR和CVaR
            metrics['var_95'][i] = self.calculate_var(returns, 0.95)
            metrics['cvar_95'][i] = self.calculate_cvar(returns, 0.95)
        
        return metrics
    
    def parallel_backtest(self, strategies: List[Dict], 
                         data: pd.DataFrame,
                         n_workers: int = 4) -> List[Dict]:
        """
        并行回测多个策略
        
        Args:
            strategies: 策略配置列表
            data: 市场数据
            n_workers: 并行工作线程数
            
        Returns:
            List[Dict]: 回测结果列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._run_single_backtest, strategy, data): strategy
                for strategy in strategies
            }
            
            for future in as_completed(futures):
                strategy = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"策略 {strategy.get('name', 'unknown')} 回测完成")
                except Exception as e:
                    logger.error(f"策略 {strategy.get('name', 'unknown')} 回测失败: {e}")
        
        return results
    
    def _run_single_backtest(self, strategy: Dict, data: pd.DataFrame) -> Dict:
        """
        运行单个策略回测
        
        Args:
            strategy: 策略配置
            data: 市场数据
            
        Returns:
            Dict: 回测结果
        """
        # 这里应该实际运行回测
        # 暂时返回模拟结果
        returns = np.random.randn(len(data)) * 0.01
        returns_tensor = self.to_tensor(returns)
        
        result = {
            'strategy_name': strategy.get('name', 'unknown'),
            'total_return': self.to_numpy(torch.prod(1 + returns_tensor) - 1).item(),
            'sharpe_ratio': self.to_numpy(self.calculate_sharpe_ratio(returns_tensor)).item(),
            'max_drawdown': self.to_numpy(self.calculate_drawdown(torch.cumprod(1 + returns_tensor, dim=0))[1]).item(),
        }
        
        return result
    
    def optimize_parameters(self, param_grid: Dict, 
                          data: pd.DataFrame,
                          metric: str = 'sharpe_ratio') -> Dict:
        """
        参数优化（GPU加速网格搜索）
        
        Args:
            param_grid: 参数网格
            data: 市场数据
            metric: 优化目标指标
            
        Returns:
            Dict: 最优参数
        """
        # 生成所有参数组合
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        logger.info(f"开始参数优化，共 {len(combinations)} 个组合")
        
        # 并行测试所有组合
        strategies = [{'name': f'strategy_{i}', **params} 
                     for i, params in enumerate(combinations)]
        
        results = self.parallel_backtest(strategies, data)
        
        # 找到最优参数
        best_result = max(results, key=lambda x: x.get(metric, -float('inf')))
        best_idx = results.index(best_result)
        best_params = combinations[best_idx]
        
        logger.info(f"最优参数: {best_params}")
        logger.info(f"最优{metric}: {best_result[metric]:.4f}")
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'all_results': results
        }


def calculate_metrics_gpu(returns: np.ndarray, device: Optional[str] = None) -> Dict:
    """
    使用GPU计算指标的便捷函数
    
    Args:
        returns: 收益率数组
        device: 设备类型
        
    Returns:
        Dict: 指标字典
    """
    calculator = GPUBacktestCalculator(device)
    returns_tensor = calculator.to_tensor(returns)
    
    equity = torch.cumprod(1 + returns_tensor, dim=0)
    drawdown, max_dd = calculator.calculate_drawdown(equity)
    
    metrics = {
        'total_return': calculator.to_numpy(torch.prod(1 + returns_tensor) - 1).item(),
        'sharpe_ratio': calculator.to_numpy(calculator.calculate_sharpe_ratio(returns_tensor)).item(),
        'sortino_ratio': calculator.to_numpy(calculator.calculate_sortino_ratio(returns_tensor)).item(),
        'max_drawdown': calculator.to_numpy(max_dd).item(),
        'var_95': calculator.to_numpy(calculator.calculate_var(returns_tensor, 0.95)).item(),
        'cvar_95': calculator.to_numpy(calculator.calculate_cvar(returns_tensor, 0.95)).item(),
    }
    
    return metrics


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    n = 10000
    returns = np.random.randn(n) * 0.01 + 0.0001
    
    print("测试GPU回测计算...")
    import time
    
    # GPU计算
    start = time.time()
    metrics_gpu = calculate_metrics_gpu(returns)
    gpu_time = time.time() - start
    
    print(f"\nGPU计算时间: {gpu_time:.4f}秒")
    print("\nGPU计算结果:")
    for key, value in metrics_gpu.items():
        print(f"  {key}: {value:.6f}")
    
    # 测试批量计算
    print("\n测试批量计算...")
    calculator = GPUBacktestCalculator()
    
    batch_size = 100
    returns_batch = calculator.to_tensor(np.random.randn(batch_size, 1000) * 0.01)
    
    start = time.time()
    batch_metrics = calculator.batch_calculate_metrics(returns_batch)
    batch_time = time.time() - start
    
    print(f"批量计算时间 ({batch_size}个策略): {batch_time:.4f}秒")
    print(f"平均每个策略: {batch_time/batch_size*1000:.2f}ms")
    
    # 测试参数优化
    print("\n测试参数优化...")
    param_grid = {
        'period': [10, 20, 30],
        'threshold': [0.01, 0.02, 0.03]
    }
    
    dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
    data = pd.DataFrame({
        'close': np.random.randn(1000).cumsum() + 100
    }, index=dates)
    
    optimization_result = calculator.optimize_parameters(param_grid, data)
    print(f"\n最优参数: {optimization_result['best_params']}")