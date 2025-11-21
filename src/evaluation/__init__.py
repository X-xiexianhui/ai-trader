"""
模型评估模块
包含特征验证、模型评估和性能分析

模块6: 评估层 - 完整实现
- Walk-forward验证 (6.1)
- 特征重要性分析 (6.2)
- 过拟合检测 (6.3)
- 市场状态泛化 (6.4)
"""

# 特征验证 (模块1.4)
from .feature_validator import FeatureValidator

# Walk-forward验证 (模块6.1)
from .walk_forward import (
    TimeWindowGenerator,
    WalkForwardValidator,
    ValidationResultAggregator
)

# 特征重要性分析 (模块6.2)
from .ablation_study import (
    AblationStudy,
    FeatureGroupAnalyzer
)

# 过拟合检测 (模块6.3)
from .overfitting_detection import (
    OverfittingDetector,
    MultiSeedStabilityTester
)

# 市场状态泛化 (模块6.4)
from .market_state import (
    MarketStateIdentifier,
    StateSpecificEvaluator
)

__all__ = [
    # 特征验证
    'FeatureValidator',
    
    # Walk-forward验证
    'TimeWindowGenerator',
    'WalkForwardValidator',
    'ValidationResultAggregator',
    
    # 特征重要性分析
    'AblationStudy',
    'FeatureGroupAnalyzer',
    
    # 过拟合检测
    'OverfittingDetector',
    'MultiSeedStabilityTester',
    
    # 市场状态泛化
    'MarketStateIdentifier',
    'StateSpecificEvaluator',
]