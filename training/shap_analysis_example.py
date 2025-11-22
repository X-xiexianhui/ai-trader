"""
SHAP值分析示例脚本

演示如何使用SHAPAnalyzer进行模型可解释性分析
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.evaluation.shap_analysis import SHAPAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("SHAP值分析示例")
    logger.info("=" * 80)
    
    # 1. 加载归一化后的数据
    data_path = project_root / 'data' / 'processed' / 'MES_normalized_5m.csv'
    
    if not data_path.exists():
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请先运行 training/feature_engineering.py 生成归一化数据")
        return
    
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # 2. 准备特征和目标
    # 特征列（20维手工特征）
    feature_cols = [
        # 价格与收益（3维）
        'ret_1', 'ret_5', 'ret_20',
        # 波动率（3维）
        'ATR14', 'vol_20', 'parkinson_vol',
        # 技术指标（2维）
        'EMA20', 'stoch',
        # 成交量（4维）
        'volume', 'volume_zscore', 'volume_change_1', 'OBV_slope_20',
        # K线形态（6维）
        'pos_in_range_20', 'dist_to_HH20', 'dist_to_LL20',
        'upper_shadow_ratio', 'lower_shadow_ratio', 'FVG',
        # 时间周期（2维）
        'sin_tod', 'cos_tod'
    ]
    
    X = df[feature_cols].copy()
    
    # 目标变量：未来1周期收益率
    y = df['Close'].pct_change(1).shift(-1)
    
    # 删除NaN
    valid_idx = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_idx]
    y = y[valid_idx]
    
    logger.info(f"有效样本数: {len(X)}")
    logger.info(f"特征数: {len(feature_cols)}")
    
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info(f"训练集: {len(X_train)} 样本")
    logger.info(f"测试集: {len(X_test)} 样本")
    
    # 4. 训练一个简单的随机森林模型
    logger.info("\n训练随机森林模型...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"训练集R²: {train_score:.4f}")
    logger.info(f"测试集R²: {test_score:.4f}")
    
    # 5. 创建SHAP分析器
    logger.info("\n初始化SHAP分析器...")
    shap_analyzer = SHAPAnalyzer(model, model_type='tree')
    
    # 6. 拟合解释器（使用训练集）
    shap_analyzer.fit(X_train)
    
    # 7. 计算测试集的SHAP值
    shap_analyzer.calculate_shap_values(X_test)
    
    # 8. 获取全局特征重要性
    logger.info("\n全局特征重要性分析:")
    importance_df = shap_analyzer.get_global_importance(top_n=20)
    
    # 9. 解释几个样本
    logger.info("\n样本预测解释:")
    for idx in [0, 100, 200]:
        if idx < len(X_test):
            explanation = shap_analyzer.explain_prediction(X_test, sample_idx=idx)
    
    # 10. 分析特征交互
    logger.info("\n特征交互分析:")
    interactions_df = shap_analyzer.analyze_feature_interactions(X_test, top_n=10)
    
    # 11. 生成完整报告
    output_dir = project_root / 'training' / 'output' / 'shap_analysis'
    logger.info(f"\n生成SHAP分析报告到: {output_dir}")
    
    shap_analyzer.generate_report(
        X_test,
        output_dir=output_dir,
        report_name='mes_shap_analysis',
        n_samples_to_explain=5
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("SHAP分析完成！")
    logger.info("=" * 80)
    logger.info(f"报告保存在: {output_dir}")
    logger.info("\n生成的文件:")
    logger.info("  1. mes_shap_analysis_importance.csv - 特征重要性")
    logger.info("  2. mes_shap_analysis_summary_*.png - 摘要图")
    logger.info("  3. mes_shap_analysis_waterfall_*.png - 样本瀑布图")
    logger.info("  4. mes_shap_analysis_force_*.png - 样本力图")
    logger.info("  5. mes_shap_analysis_dependence_*.png - 特征依赖图")
    logger.info("  6. mes_shap_analysis_interactions.csv - 特征交互")
    logger.info("  7. mes_shap_analysis.txt - 文本报告")


if __name__ == '__main__':
    main()