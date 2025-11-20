"""
Walk-Forward验证框架

实现滚动窗口验证，评估模型的时间稳定性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward验证器
    
    使用滚动窗口方法验证模型的时间稳定性
    """
    
    def __init__(
        self,
        train_window: int = 24,  # 训练窗口（月）
        val_window: int = 6,  # 验证窗口（月）
        test_window: int = 6,  # 测试窗口（月）
        step_size: int = 6,  # 步长（月）
        min_train_size: Optional[int] = None  # 最小训练集大小
    ):
        """
        初始化Walk-Forward验证器
        
        Args:
            train_window: 训练窗口大小（月）
            val_window: 验证窗口大小（月）
            test_window: 测试窗口大小（月）
            step_size: 滚动步长（月）
            min_train_size: 最小训练集大小（数据点数）
        """
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        
        self.splits = []
        self.results = []
        
        logger.info(f"Walk-Forward验证器初始化: "
                   f"训练={train_window}月, 验证={val_window}月, "
                   f"测试={test_window}月, 步长={step_size}月")
    
    def create_splits(
        self,
        data: pd.DataFrame,
        date_column: Optional[str] = None
    ) -> List[Dict]:
        """
        创建时间分割
        
        Args:
            data: 数据DataFrame
            date_column: 日期列名，如果为None则使用索引
            
        Returns:
            List[Dict]: 分割列表，每个包含train/val/test的索引
        """
        # 获取日期索引
        if date_column:
            dates = pd.to_datetime(data[date_column])
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("数据索引必须是DatetimeIndex或指定date_column")
            dates = data.index
        
        # 按月分组
        data_with_dates = data.copy()
        data_with_dates['_date'] = dates
        data_with_dates['_year_month'] = dates.to_period('M')
        
        # 获取所有月份
        all_months = sorted(data_with_dates['_year_month'].unique())
        
        self.splits = []
        
        # 滚动窗口
        start_idx = 0
        while start_idx + self.train_window + self.val_window + self.test_window <= len(all_months):
            # 训练集月份
            train_months = all_months[start_idx:start_idx + self.train_window]
            # 验证集月份
            val_months = all_months[
                start_idx + self.train_window:
                start_idx + self.train_window + self.val_window
            ]
            # 测试集月份
            test_months = all_months[
                start_idx + self.train_window + self.val_window:
                start_idx + self.train_window + self.val_window + self.test_window
            ]
            
            # 获取对应的数据索引
            train_mask = data_with_dates['_year_month'].isin(train_months)
            val_mask = data_with_dates['_year_month'].isin(val_months)
            test_mask = data_with_dates['_year_month'].isin(test_months)
            
            train_indices = data_with_dates[train_mask].index.tolist()
            val_indices = data_with_dates[val_mask].index.tolist()
            test_indices = data_with_dates[test_mask].index.tolist()
            
            # 检查最小训练集大小
            if self.min_train_size and len(train_indices) < self.min_train_size:
                logger.warning(f"训练集大小{len(train_indices)}小于最小要求{self.min_train_size}，跳过")
                start_idx += self.step_size
                continue
            
            split = {
                'fold': len(self.splits),
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'train_months': [str(m) for m in train_months],
                'val_months': [str(m) for m in val_months],
                'test_months': [str(m) for m in test_months],
                'train_start': dates[train_indices[0]],
                'train_end': dates[train_indices[-1]],
                'val_start': dates[val_indices[0]],
                'val_end': dates[val_indices[-1]],
                'test_start': dates[test_indices[0]],
                'test_end': dates[test_indices[-1]]
            }
            
            self.splits.append(split)
            
            # 移动窗口
            start_idx += self.step_size
        
        logger.info(f"创建了{len(self.splits)}个Walk-Forward分割")
        return self.splits
    
    def validate(
        self,
        data: pd.DataFrame,
        train_func: Callable,
        eval_func: Callable,
        date_column: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        执行Walk-Forward验证
        
        Args:
            data: 完整数据
            train_func: 训练函数，接收(train_data, val_data, **kwargs)，返回模型
            eval_func: 评估函数，接收(model, test_data, **kwargs)，返回指标字典
            date_column: 日期列名
            **kwargs: 传递给train_func和eval_func的额外参数
            
        Returns:
            List[Dict]: 每个fold的评估结果
        """
        if not self.splits:
            self.create_splits(data, date_column)
        
        self.results = []
        
        for split in self.splits:
            fold = split['fold']
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold + 1}/{len(self.splits)}")
            logger.info(f"训练: {split['train_start'].date()} 到 {split['train_end'].date()}")
            logger.info(f"验证: {split['val_start'].date()} 到 {split['val_end'].date()}")
            logger.info(f"测试: {split['test_start'].date()} 到 {split['test_end'].date()}")
            logger.info(f"{'='*60}")
            
            # 准备数据
            train_data = data.loc[split['train_indices']]
            val_data = data.loc[split['val_indices']]
            test_data = data.loc[split['test_indices']]
            
            try:
                # 训练模型
                logger.info("训练模型...")
                model = train_func(train_data, val_data, **kwargs)
                
                # 在验证集上评估
                logger.info("验证集评估...")
                val_metrics = eval_func(model, val_data, **kwargs)
                
                # 在测试集上评估
                logger.info("测试集评估...")
                test_metrics = eval_func(model, test_data, **kwargs)
                
                # 记录结果
                result = {
                    'fold': fold,
                    'train_start': split['train_start'],
                    'train_end': split['train_end'],
                    'val_start': split['val_start'],
                    'val_end': split['val_end'],
                    'test_start': split['test_start'],
                    'test_end': split['test_end'],
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'test_size': len(test_data),
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics,
                    'model': model  # 保存模型引用
                }
                
                self.results.append(result)
                
                logger.info(f"Fold {fold + 1} 完成")
                logger.info(f"验证集指标: {val_metrics}")
                logger.info(f"测试集指标: {test_metrics}")
                
            except Exception as e:
                logger.error(f"Fold {fold + 1} 失败: {str(e)}")
                continue
        
        logger.info(f"\nWalk-Forward验证完成，成功{len(self.results)}/{len(self.splits)}个fold")
        return self.results
    
    def get_summary_statistics(self) -> Dict:
        """
        获取汇总统计
        
        Returns:
            Dict: 汇总统计字典
        """
        if not self.results:
            return {}
        
        # 提取所有指标
        val_metrics_list = [r['val_metrics'] for r in self.results]
        test_metrics_list = [r['test_metrics'] for r in self.results]
        
        # 计算统计量
        summary = {
            'num_folds': len(self.results),
            'validation': self._compute_statistics(val_metrics_list),
            'test': self._compute_statistics(test_metrics_list)
        }
        
        # 计算稳定性指标
        summary['stability'] = self._compute_stability(test_metrics_list)
        
        return summary
    
    def _compute_statistics(self, metrics_list: List[Dict]) -> Dict:
        """计算指标统计量"""
        if not metrics_list:
            return {}
        
        # 获取所有指标名称
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        stats = {}
        for name in metric_names:
            values = [m.get(name, np.nan) for m in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                }
        
        return stats
    
    def _compute_stability(self, metrics_list: List[Dict]) -> Dict:
        """
        计算稳定性指标
        
        Returns:
            Dict: 稳定性指标
        """
        if len(metrics_list) < 2:
            return {}
        
        stability = {}
        
        # 获取所有指标名称
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        for name in metric_names:
            values = [m.get(name, np.nan) for m in metrics_list]
            values = [v for v in values if not np.isnan(v)]
            
            if len(values) >= 2:
                # 变异系数（CV）
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                
                # 稳定性得分（1 - CV，越接近1越稳定）
                stability_score = max(0, 1 - abs(cv))
                
                stability[name] = {
                    'cv': cv,
                    'stability_score': stability_score,
                    'is_stable': cv < 0.3  # CV < 0.3认为是稳定的
                }
        
        return stability
    
    def print_summary(self):
        """打印汇总报告"""
        summary = self.get_summary_statistics()
        
        if not summary:
            print("没有可用的结果")
            return
        
        print("\n" + "="*80)
        print("Walk-Forward验证汇总报告")
        print("="*80)
        
        print(f"\n总fold数: {summary['num_folds']}")
        
        # 验证集统计
        print("\n验证集性能:")
        print("-" * 80)
        self._print_statistics(summary['validation'])
        
        # 测试集统计
        print("\n测试集性能:")
        print("-" * 80)
        self._print_statistics(summary['test'])
        
        # 稳定性分析
        print("\n稳定性分析:")
        print("-" * 80)
        for metric_name, stability in summary['stability'].items():
            status = "✓ 稳定" if stability['is_stable'] else "✗ 不稳定"
            print(f"  {metric_name}:")
            print(f"    变异系数(CV): {stability['cv']:.4f}")
            print(f"    稳定性得分: {stability['stability_score']:.4f}")
            print(f"    状态: {status}")
        
        print("="*80 + "\n")
    
    def _print_statistics(self, stats: Dict):
        """打印统计信息"""
        if not stats:
            print("  无数据")
            return
        
        for metric_name, values in stats.items():
            print(f"  {metric_name}:")
            print(f"    均值: {values['mean']:.4f}")
            print(f"    标准差: {values['std']:.4f}")
            print(f"    最小值: {values['min']:.4f}")
            print(f"    最大值: {values['max']:.4f}")
            print(f"    中位数: {values['median']:.4f}")
            print(f"    变异系数: {values['cv']:.4f}")
    
    def save_results(self, filepath: str):
        """
        保存结果到文件
        
        Args:
            filepath: 保存路径
        """
        import json
        
        # 准备可序列化的结果
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'fold': result['fold'],
                'train_start': result['train_start'].isoformat(),
                'train_end': result['train_end'].isoformat(),
                'val_start': result['val_start'].isoformat(),
                'val_end': result['val_end'].isoformat(),
                'test_start': result['test_start'].isoformat(),
                'test_end': result['test_end'].isoformat(),
                'train_size': result['train_size'],
                'val_size': result['val_size'],
                'test_size': result['test_size'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics']
            }
            serializable_results.append(serializable_result)
        
        # 保存
        with open(filepath, 'w') as f:
            json.dump({
                'config': {
                    'train_window': self.train_window,
                    'val_window': self.val_window,
                    'test_window': self.test_window,
                    'step_size': self.step_size
                },
                'results': serializable_results,
                'summary': self.get_summary_statistics()
            }, f, indent=2, default=str)
        
        logger.info(f"结果已保存到: {filepath}")


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'value': np.random.randn(len(dates)).cumsum(),
        'feature1': np.random.randn(len(dates)),
        'feature2': np.random.randn(len(dates))
    }, index=dates)
    
    # 创建验证器
    validator = WalkForwardValidator(
        train_window=12,
        val_window=3,
        test_window=3,
        step_size=3
    )
    
    # 创建分割
    splits = validator.create_splits(data)
    print(f"\n创建了{len(splits)}个分割")
    
    # 定义简单的训练和评估函数
    def simple_train(train_data, val_data):
        # 简单返回训练数据的均值作为"模型"
        return {'mean': train_data['value'].mean()}
    
    def simple_eval(model, test_data):
        # 简单计算预测误差
        predictions = np.full(len(test_data), model['mean'])
        mse = np.mean((test_data['value'].values - predictions) ** 2)
        return {'mse': mse, 'rmse': np.sqrt(mse)}
    
    # 执行验证
    results = validator.validate(data, simple_train, simple_eval)
    
    # 打印汇总
    validator.print_summary()