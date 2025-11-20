"""
交叉验证

实现时间序列交叉验证和Walk-Forward验证
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable, Dict
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class TimeSeriesSplit:
    """时间序列交叉验证分割器"""
    
    def __init__(self,
                 n_splits: int = 5,
                 test_size: Optional[int] = None,
                 gap: int = 0):
        """
        初始化时间序列分割器
        
        Args:
            n_splits: 分割数量
            test_size: 测试集大小
            gap: 训练集和测试集之间的间隔
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
        logger.info(f"时间序列分割器初始化: n_splits={n_splits}, gap={gap}")
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        分割数据
        
        Args:
            X: 数据数组
            
        Returns:
            List[Tuple]: (训练索引, 测试索引)列表
        """
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # 测试集结束位置
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            # 测试集开始位置
            test_start = test_end - test_size
            # 训练集结束位置（考虑gap）
            train_end = test_start - self.gap
            
            if train_end <= 0:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            
            logger.debug(f"Split {i+1}: train={len(train_indices)}, test={len(test_indices)}")
        
        return splits
    
    def get_n_splits(self) -> int:
        """获取分割数量"""
        return self.n_splits


class WalkForwardValidation:
    """Walk-Forward验证"""
    
    def __init__(self,
                 train_period: int,
                 test_period: int,
                 step_size: Optional[int] = None,
                 gap: int = 0):
        """
        初始化Walk-Forward验证
        
        Args:
            train_period: 训练周期长度
            test_period: 测试周期长度
            step_size: 步长（默认等于test_period）
            gap: 训练和测试之间的间隔
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size or test_period
        self.gap = gap
        
        logger.info(f"Walk-Forward验证初始化: train={train_period}, test={test_period}")
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        分割数据
        
        Args:
            X: 数据数组
            
        Returns:
            List[Tuple]: (训练索引, 测试索引)列表
        """
        n_samples = len(X)
        splits = []
        
        start = 0
        while start + self.train_period + self.gap + self.test_period <= n_samples:
            # 训练集
            train_start = start
            train_end = start + self.train_period
            
            # 测试集
            test_start = train_end + self.gap
            test_end = test_start + self.test_period
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            
            # 移动窗口
            start += self.step_size
        
        logger.info(f"生成了{len(splits)}个Walk-Forward分割")
        return splits
    
    def get_n_splits(self, X: np.ndarray) -> int:
        """获取分割数量"""
        return len(self.split(X))


class ExpandingWindowSplit:
    """扩展窗口分割"""
    
    def __init__(self,
                 initial_train_size: int,
                 test_size: int,
                 step_size: Optional[int] = None,
                 gap: int = 0):
        """
        初始化扩展窗口分割
        
        Args:
            initial_train_size: 初始训练集大小
            test_size: 测试集大小
            step_size: 步长
            gap: 间隔
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.gap = gap
        
        logger.info(f"扩展窗口分割初始化: initial_train={initial_train_size}")
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """分割数据"""
        n_samples = len(X)
        splits = []
        
        train_end = self.initial_train_size
        
        while train_end + self.gap + self.test_size <= n_samples:
            # 训练集（从开始到train_end）
            train_indices = np.arange(0, train_end)
            
            # 测试集
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
            
            # 扩展训练集
            train_end += self.step_size
        
        logger.info(f"生成了{len(splits)}个扩展窗口分割")
        return splits


class CrossValidator:
    """交叉验证器"""
    
    def __init__(self,
                 splitter,
                 scoring_fn: Callable,
                 verbose: bool = True):
        """
        初始化交叉验证器
        
        Args:
            splitter: 分割器
            scoring_fn: 评分函数
            verbose: 是否显示详细信息
        """
        self.splitter = splitter
        self.scoring_fn = scoring_fn
        self.verbose = verbose
        
        logger.info("交叉验证器初始化")
    
    def validate(self,
                 model_fn: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 **fit_params) -> Dict:
        """
        执行交叉验证
        
        Args:
            model_fn: 模型创建函数
            X: 特征
            y: 标签
            **fit_params: 训练参数
            
        Returns:
            Dict: 验证结果
        """
        splits = self.splitter.split(X)
        scores = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splits):
            if self.verbose:
                logger.info(f"Fold {fold + 1}/{len(splits)}")
            
            # 分割数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 创建并训练模型
            model = model_fn()
            model.fit(X_train, y_train, **fit_params)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评分
            score = self.scoring_fn(y_test, y_pred)
            scores.append(score)
            
            fold_results.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'score': score
            })
            
            if self.verbose:
                logger.info(f"  Score: {score:.6f}")
        
        results = {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_results': fold_results
        }
        
        logger.info(f"交叉验证完成: {results['mean_score']:.6f} ± {results['std_score']:.6f}")
        
        return results


class TimeBasedSplit:
    """基于时间的分割"""
    
    def __init__(self,
                 train_months: int = 24,
                 val_months: int = 6,
                 test_months: int = 6):
        """
        初始化基于时间的分割
        
        Args:
            train_months: 训练月数
            val_months: 验证月数
            test_months: 测试月数
        """
        self.train_months = train_months
        self.val_months = val_months
        self.test_months = test_months
        
        logger.info(f"时间分割初始化: {train_months}/{val_months}/{test_months}月")
    
    def split(self, dates: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        根据日期分割数据
        
        Args:
            dates: 日期索引
            
        Returns:
            Tuple: (训练索引, 验证索引, 测试索引)
        """
        # 计算分割点
        total_days = (dates[-1] - dates[0]).days
        
        train_end = dates[0] + timedelta(days=self.train_months * 30)
        val_end = train_end + timedelta(days=self.val_months * 30)
        
        # 生成索引
        train_mask = dates < train_end
        val_mask = (dates >= train_end) & (dates < val_end)
        test_mask = dates >= val_end
        
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        logger.info(f"时间分割: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return train_idx, val_idx, test_idx


class PurgedKFold:
    """净化K折交叉验证（用于时间序列）"""
    
    def __init__(self,
                 n_splits: int = 5,
                 embargo_pct: float = 0.01):
        """
        初始化净化K折
        
        Args:
            n_splits: 分割数
            embargo_pct: 禁运期百分比（防止数据泄露）
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        
        logger.info(f"净化K折初始化: n_splits={n_splits}, embargo={embargo_pct}")
    
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        分割数据
        
        Args:
            X: 特征
            y: 标签
            
        Returns:
            List[Tuple]: (训练索引, 测试索引)列表
        """
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # 计算每折的大小
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # 测试集
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = np.arange(test_start, test_end)
            
            # 训练集（排除测试集和禁运期）
            train_idx = []
            
            # 测试集之前的数据
            if test_start > embargo_size:
                train_idx.extend(range(0, test_start - embargo_size))
            
            # 测试集之后的数据
            if test_end + embargo_size < n_samples:
                train_idx.extend(range(test_end + embargo_size, n_samples))
            
            train_idx = np.array(train_idx)
            
            splits.append((train_idx, test_idx))
            
            logger.debug(f"Fold {i+1}: train={len(train_idx)}, test={len(test_idx)}")
        
        return splits


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)
    
    # 测试时间序列分割
    print("测试时间序列分割...")
    tscv = TimeSeriesSplit(n_splits=5, gap=10)
    splits = tscv.split(X)
    
    print(f"生成了{len(splits)}个分割")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    
    # 测试Walk-Forward验证
    print("\n测试Walk-Forward验证...")
    wfv = WalkForwardValidation(train_period=500, test_period=100, step_size=100)
    splits = wfv.split(X)
    
    print(f"生成了{len(splits)}个分割")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    
    # 测试扩展窗口
    print("\n测试扩展窗口分割...")
    ews = ExpandingWindowSplit(initial_train_size=500, test_size=100)
    splits = ews.split(X)
    
    print(f"生成了{len(splits)}个分割")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")
    
    # 测试净化K折
    print("\n测试净化K折...")
    pkf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    splits = pkf.split(X, y)
    
    print(f"生成了{len(splits)}个分割")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Split {i+1}: train={len(train_idx)}, test={len(test_idx)}")