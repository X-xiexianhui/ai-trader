"""
评估与验证模块

提供全面的模型评估和验证工具
"""

from .walk_forward import WalkForwardValidator
from .feature_importance import PermutationImportance, AblationStudy
from .overfitting_detector import OverfittingDetector
from .market_regime import MarketRegimeDetector, RegimeBasedEvaluator
from .evaluation_tools import (
    MultiSeedStabilityTest,
    StressTest,
    BenchmarkComparison
)
from .report_generator import (
    EvaluationReportGenerator,
    create_comprehensive_report
)

__all__ = [
    # Walk-Forward验证
    'WalkForwardValidator',
    
    # 特征重要性分析
    'PermutationImportance',
    'AblationStudy',
    
    # 过拟合检测
    'OverfittingDetector',
    
    # 市场状态分析
    'MarketRegimeDetector',
    'RegimeBasedEvaluator',
    
    # 评估工具
    'MultiSeedStabilityTest',
    'StressTest',
    'BenchmarkComparison',
    
    # 报告生成
    'EvaluationReportGenerator',
    'create_comprehensive_report',
]

__version__ = '1.0.0'