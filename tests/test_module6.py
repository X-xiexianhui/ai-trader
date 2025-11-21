"""
模块6测试文件
测试评估层的所有功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 从evaluation模块导入
from src.evaluation import (
    TimeWindowGenerator,
    WalkForwardValidator,
    ValidationResultAggregator,
    AblationStudy,
    FeatureGroupAnalyzer,
    OverfittingDetector,
    MultiSeedStabilityTester,
    MarketStateIdentifier,
    StateSpecificEvaluator
)


# ============================================================================
# 测试数据生成
# ============================================================================

def generate_test_data(n_samples=1000, n_features=10):
    """生成测试数据"""
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='5min')
    
    # 生成价格数据
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)
    
    data = pd.DataFrame({
        'open': prices + np.random.randn(n_samples) * 0.1,
        'high': prices + np.abs(np.random.randn(n_samples) * 0.2),
        'low': prices - np.abs(np.random.randn(n_samples) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    }, index=dates)
    
    # 生成特征
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_samples)
    
    return data


def generate_training_history(n_epochs=50):
    """生成训练历史"""
    train_loss = 1.0 - np.linspace(0, 0.7, n_epochs) + np.random.randn(n_epochs) * 0.05
    val_loss = 1.0 - np.linspace(0, 0.5, n_epochs) + np.random.randn(n_epochs) * 0.08
    
    train_history = {
        'loss': train_loss.tolist(),
        'accuracy': (0.5 + np.linspace(0, 0.3, n_epochs) + np.random.randn(n_epochs) * 0.02).tolist()
    }
    
    val_history = {
        'loss': val_loss.tolist(),
        'accuracy': (0.5 + np.linspace(0, 0.2, n_epochs) + np.random.randn(n_epochs) * 0.03).tolist()
    }
    
    return train_history, val_history


# ============================================================================
# 测试6.1: Walk-forward验证
# ============================================================================

class TestWalkForward:
    """测试Walk-forward验证"""
    
    def test_time_window_generator(self):
        """测试时间窗口生成器"""
        print("\n" + "="*60)
        print("测试6.1.2: 时间窗口滚动生成器")
        print("="*60)
        
        data = generate_test_data(n_samples=2000)
        
        generator = TimeWindowGenerator(
            train_months=6,
            val_months=2,
            test_months=2,
            step_months=2
        )
        
        folds = generator.generate_folds(data)
        
        print(f"生成了 {len(folds)} 个fold")
        assert len(folds) > 0, "应该生成至少一个fold"
        
        # 验证fold结构
        for i, fold in enumerate(folds):
            print(f"\nFold {i+1}:")
            print(f"  训练集: {fold['train_start']} 到 {fold['train_end']}")
            print(f"  验证集: {fold['val_start']} 到 {fold['val_end']}")
            print(f"  测试集: {fold['test_start']} 到 {fold['test_end']}")
            
            assert fold['train_start'] < fold['train_end']
            assert fold['val_start'] < fold['val_end']
            assert fold['test_start'] < fold['test_end']
            assert fold['train_end'] <= fold['val_start']
            assert fold['val_end'] <= fold['test_start']
        
        print("\n✓ 时间窗口生成器测试通过")
    
    def test_validation_result_aggregator(self):
        """测试验证结果汇总"""
        print("\n" + "="*60)
        print("测试6.1.3: 多折验证结果汇总")
        print("="*60)
        
        # 模拟多折结果
        results = []
        for i in range(5):
            results.append({
                'fold': i,
                'train_metrics': {'loss': 0.3 + np.random.rand() * 0.1},
                'val_metrics': {'loss': 0.4 + np.random.rand() * 0.1},
                'test_metrics': {'loss': 0.45 + np.random.rand() * 0.1}
            })
        
        aggregator = ValidationResultAggregator(results)
        
        # 汇总指标
        summary = aggregator.aggregate_metrics()
        print("\n汇总指标:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        assert 'test_loss_mean' in summary
        assert 'test_loss_std' in summary
        
        # 稳定性分析
        stability = aggregator.analyze_stability()
        print(f"\n稳定性分析:")
        print(f"  稳定: {stability['is_stable']}")
        print(f"  平均CV: {stability['avg_cv']:.4f}")
        
        # 泛化能力评估
        generalization = aggregator.evaluate_generalization()
        print(f"\n泛化能力:")
        print(f"  良好: {generalization['is_generalizable']}")
        
        print("\n✓ 验证结果汇总测试通过")


# ============================================================================
# 测试6.2: 特征重要性分析
# ============================================================================

class TestAblationStudy:
    """测试消融实验"""
    
    def test_ablation_study(self):
        """测试消融实验框架"""
        print("\n" + "="*60)
        print("测试6.2.1: 消融实验框架")
        print("="*60)
        
        # 生成测试数据
        data = generate_test_data(n_samples=500, n_features=10)
        feature_cols = [f'feature_{i}' for i in range(10)]
        
        X = data[feature_cols]
        y = pd.Series(np.random.randn(len(data)), index=data.index)
        
        # 定义特征组
        feature_groups = {
            'group_1': feature_cols[:3],
            'group_2': feature_cols[3:6],
            'group_3': feature_cols[6:]
        }
        
        # 简单的训练和评估函数
        def train_func(X_train, y_train, **kwargs):
            return {'coef': np.random.randn(X_train.shape[1])}
        
        def eval_func(model, X_val, y_val, **kwargs):
            pred = np.random.randn(len(y_val))
            mse = np.mean((pred - y_val) ** 2)
            return {'mse': mse, 'mae': np.mean(np.abs(pred - y_val))}
        
        # 运行消融实验
        ablation = AblationStudy(feature_groups)
        
        split_idx = int(len(X) * 0.8)
        results = ablation.run_ablation(
            X.iloc[:split_idx],
            y.iloc[:split_idx],
            X.iloc[split_idx:],
            y.iloc[split_idx:],
            train_func,
            eval_func
        )
        
        print(f"\n完成 {len(results)} 个实验")
        assert len(results) > 0
        
        # 计算贡献度
        contributions = ablation.calculate_contributions()
        print("\n特征组贡献度:")
        print(contributions.to_string())
        
        print("\n✓ 消融实验框架测试通过")
    
    def test_feature_group_analyzer(self):
        """测试特征组贡献度分析"""
        print("\n" + "="*60)
        print("测试6.2.2: 特征组贡献度分析")
        print("="*60)
        
        # 模拟消融实验结果
        results = [
            {
                'experiment': 'baseline',
                'removed_group': None,
                'metrics': {'mse': 0.5, 'mae': 0.4}
            },
            {
                'experiment': 'remove_group_1',
                'removed_group': 'group_1',
                'removed_features': ['f1', 'f2'],
                'metrics': {'mse': 0.6, 'mae': 0.45}
            },
            {
                'experiment': 'remove_group_2',
                'removed_group': 'group_2',
                'removed_features': ['f3', 'f4'],
                'metrics': {'mse': 0.55, 'mae': 0.42}
            }
        ]
        
        analyzer = FeatureGroupAnalyzer(results)
        
        # 绝对贡献度
        abs_contrib = analyzer.analyze_absolute_contribution()
        print("\n绝对贡献度:")
        print(abs_contrib.to_string())
        
        # 相对贡献度
        rel_contrib = analyzer.analyze_relative_contribution()
        print("\n相对贡献度:")
        print(rel_contrib.to_string())
        
        # 排名
        ranked = analyzer.rank_feature_groups(metric='mse')
        print("\n特征组排名:")
        print(ranked.to_string())
        
        # 识别冗余特征
        redundant = analyzer.identify_redundant_features(threshold=0.01)
        print(f"\n冗余特征组: {redundant}")
        
        print("\n✓ 特征组贡献度分析测试通过")


# ============================================================================
# 测试6.3: 过拟合检测
# ============================================================================

class TestOverfittingDetection:
    """测试过拟合检测"""
    
    def test_overfitting_detector(self):
        """测试过拟合检测器"""
        print("\n" + "="*60)
        print("测试6.3.1: 过拟合检测器")
        print("="*60)
        
        train_history, val_history = generate_training_history(n_epochs=50)
        
        detector = OverfittingDetector(
            gap_threshold=0.20,
            variance_threshold=0.15,
            stagnation_patience=5
        )
        
        result = detector.detect(train_history, val_history)
        
        print(f"\n检测结果:")
        print(f"  是否过拟合: {result['has_overfitting']}")
        print(f"  严重程度: {result['severity']}")
        print(f"  信号数量: {len(result['signals'])}")
        
        if result['signals']:
            print(f"\n检测到的信号:")
            for signal in result['signals']:
                print(f"  - {signal['type']}: {signal.get('severity', 'unknown')}")
        
        if result['recommendations']:
            print(f"\n修复建议:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        
        print("\n✓ 过拟合检测器测试通过")
    
    def test_multi_seed_stability(self):
        """测试多Seed稳定性测试"""
        print("\n" + "="*60)
        print("测试6.3.2: 多Seed稳定性测试")
        print("="*60)
        
        # 简单的训练和评估函数
        def train_func(seed, **kwargs):
            np.random.seed(seed)
            return {'weights': np.random.randn(10)}
        
        def eval_func(model, **kwargs):
            return {
                'accuracy': 0.8 + np.random.randn() * 0.05,
                'loss': 0.3 + np.random.randn() * 0.02
            }
        
        tester = MultiSeedStabilityTester(n_seeds=5, stability_threshold=0.3)
        
        result = tester.run_stability_test(train_func, eval_func)
        
        print(f"\n稳定性测试结果:")
        print(f"  测试种子数: {result['n_seeds']}")
        print(f"  是否稳定: {result['is_stable']}")
        print(f"  平均CV: {result['avg_cv']:.4f}")
        
        print(f"\n各指标统计:")
        for metric, stats in result['statistics'].items():
            print(f"  {metric}:")
            print(f"    均值: {stats['mean']:.4f}")
            print(f"    标准差: {stats['std']:.4f}")
            print(f"    CV: {stats['cv']:.4f}")
        
        print("\n✓ 多Seed稳定性测试通过")


# ============================================================================
# 测试6.4: 市场状态泛化
# ============================================================================

class TestMarketState:
    """测试市场状态识别"""
    
    def test_market_state_identifier(self):
        """测试市场状态识别器"""
        print("\n" + "="*60)
        print("测试6.4.1: 市场状态识别器")
        print("="*60)
        
        data = generate_test_data(n_samples=1000)
        
        identifier = MarketStateIdentifier(
            trend_window=20,
            volatility_window=20,
            trend_threshold=0.02,
            volatility_threshold=0.015
        )
        
        states = identifier.identify_states(data, price_col='close')
        
        print(f"\n识别了 {len(states)} 个时间点的市场状态")
        
        state_counts = states.value_counts()
        print(f"\n市场状态分布:")
        for state, count in state_counts.items():
            pct = count / len(states) * 100
            print(f"  {state}: {count} ({pct:.1f}%)")
        
        assert len(states) == len(data)
        assert len(state_counts) > 0
        
        print("\n✓ 市场状态识别器测试通过")
    
    def test_state_specific_evaluator(self):
        """测试分状态性能评估"""
        print("\n" + "="*60)
        print("测试6.4.2: 分状态性能评估")
        print("="*60)
        
        # 生成测试数据
        n_samples = 500
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='5min')
        
        predictions = pd.Series(np.random.randn(n_samples), index=dates)
        actuals = pd.Series(np.random.randn(n_samples), index=dates)
        
        # 生成市场状态
        states = pd.Series(
            np.random.choice(['bull_market', 'bear_market', 'ranging'], n_samples),
            index=dates
        )
        
        evaluator = StateSpecificEvaluator()
        
        results = evaluator.evaluate_by_state(predictions, actuals, states)
        
        print(f"\n评估了 {len(results)} 个市场状态")
        
        for state, metrics in results.items():
            print(f"\n{state}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # 状态比较
        comparison = evaluator.compare_states()
        print(f"\n状态性能比较:")
        print(comparison.to_string())
        
        # 识别弱点
        weak_states = evaluator.identify_weaknesses(
            metric='direction_accuracy',
            threshold=0.5
        )
        print(f"\n弱点状态: {weak_states}")
        
        # 泛化能力评估
        generalization = evaluator.assess_generalization()
        print(f"\n泛化能力:")
        print(f"  平均CV: {generalization['overall']['avg_cv']:.4f}")
        print(f"  是否泛化良好: {generalization['overall']['is_generalizable']}")
        
        print("\n✓ 分状态性能评估测试通过")


# ============================================================================
# 主测试函数
# ============================================================================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("开始模块6测试")
    print("="*80)
    
    # 测试6.1: Walk-forward验证
    test_wf = TestWalkForward()
    test_wf.test_time_window_generator()
    test_wf.test_validation_result_aggregator()
    
    # 测试6.2: 特征重要性分析
    test_ablation = TestAblationStudy()
    test_ablation.test_ablation_study()
    test_ablation.test_feature_group_analyzer()
    
    # 测试6.3: 过拟合检测
    test_overfitting = TestOverfittingDetection()
    test_overfitting.test_overfitting_detector()
    test_overfitting.test_multi_seed_stability()
    
    # 测试6.4: 市场状态泛化
    test_market = TestMarketState()
    test_market.test_market_state_identifier()
    test_market.test_state_specific_evaluator()
    
    print("\n" + "="*80)
    print("✓ 所有模块6测试通过！")
    print("="*80)


if __name__ == '__main__':
    run_all_tests()