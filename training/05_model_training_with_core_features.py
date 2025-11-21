"""
使用核心特征的模型训练脚本

功能：
1. 从JSON文件加载核心特征列表
2. 加载归一化后的训练数据
3. 提取核心特征进行模型训练
4. 在特征验证时仍使用全部27个手工特征
5. 保存训练好的模型和训练报告

注意：
- 训练模型时只使用核心特征
- 特征验证时保留所有27个手工特征

使用方法：
    python training/05_model_training_with_core_features.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import yaml
import logging
from datetime import datetime
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from src.data.storage import DataStorage
from src.features.feature_calculator import FeatureCalculator
from src.utils.logger import setup_logger


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_core_features(json_path: str = "training/output/core_features.json") -> dict:
    """从JSON文件加载核心特征"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['core_features']


def calculate_future_return(df: pd.DataFrame, periods: int = 1) -> pd.Series:
    """计算未来收益率作为目标变量"""
    future_close = df['Close'].shift(-periods)
    current_close = df['Close']
    future_return = np.log(future_close / current_close)
    return future_return


def train_model(X_train, y_train, X_test, y_test, model_type='ridge'):
    """
    训练模型
    
    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_test: 测试特征
        y_test: 测试目标
        model_type: 模型类型 ('ridge', 'lasso', 'rf')
        
    Returns:
        训练好的模型和评估指标
    """
    # 创建模型
    if model_type == 'ridge':
        model = Ridge(alpha=1.0, random_state=42)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.01, random_state=42)
    elif model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估指标
    metrics = {
        'train': {
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'r2': r2_score(y_train, y_train_pred)
        },
        'test': {
            'mse': mean_squared_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
    }
    
    return model, metrics


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger(
        name="model_training",
        log_file="05_model_training.log",
        log_level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("开始使用核心特征的模型训练流程")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("\n步骤1: 加载配置文件")
    config = load_config()
    
    data_config = config['data']
    symbols = data_config['symbols']
    
    logger.info(f"品种: {symbols}")
    
    # 2. 加载核心特征列表
    logger.info("\n步骤2: 从JSON加载核心特征列表")
    
    core_features_path = "training/output/core_features.json"
    
    if not Path(core_features_path).exists():
        logger.error(f"核心特征文件不存在: {core_features_path}")
        logger.error("请先运行 04_core_feature_selection.py")
        return
    
    core_features_dict = load_core_features(core_features_path)
    
    for symbol, features in core_features_dict.items():
        logger.info(f"{symbol}: {len(features)} 个核心特征")
        logger.info(f"  特征列表: {features}")
    
    # 3. 获取所有27个手工特征名称（用于验证）
    logger.info("\n步骤3: 获取所有手工特征名称")
    feature_calculator = FeatureCalculator()
    
    # 手工计算一次以获取特征名称
    processed_storage = DataStorage(base_path='data/processed')
    sample_symbol = symbols[0]
    sample_df = processed_storage.load_parquet(f"{sample_symbol}_train_normalized")
    
    if sample_df is None:
        logger.error("无法加载样本数据")
        return
    
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    sample_ohlcv = sample_df[ohlcv_cols].head(100)
    feature_calculator.calculate_all_features(sample_ohlcv)
    
    all_feature_names = feature_calculator.get_feature_names()
    logger.info(f"所有手工特征数量: {len(all_feature_names)}")
    logger.info(f"特征列表: {all_feature_names}")
    
    # 4. 加载训练和测试数据
    logger.info("\n步骤4: 加载训练和测试数据")
    
    train_data_dict = {}
    test_data_dict = {}
    
    for symbol in symbols:
        logger.info(f"\n加载 {symbol}...")
        
        train_df = processed_storage.load_parquet(f"{symbol}_train_normalized")
        test_df = processed_storage.load_parquet(f"{symbol}_test_normalized")
        
        if train_df is not None and test_df is not None:
            train_data_dict[symbol] = train_df
            test_data_dict[symbol] = test_df
            logger.info(f"✓ {symbol} 加载成功")
            logger.info(f"  训练集: {len(train_df)} 条")
            logger.info(f"  测试集: {len(test_df)} 条")
        else:
            logger.warning(f"✗ {symbol} 加载失败")
    
    if not train_data_dict:
        logger.error("没有成功加载任何数据，退出")
        return
    
    # 5. 训练模型
    logger.info("\n步骤5: 使用核心特征训练模型")
    
    models_dict = {}
    metrics_dict = {}
    
    for symbol in train_data_dict.keys():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"训练 {symbol} 的模型")
        logger.info(f"{'=' * 80}")
        
        train_df = train_data_dict[symbol]
        test_df = test_data_dict[symbol]
        
        # 计算目标变量
        y_train = calculate_future_return(train_df, periods=1)
        y_test = calculate_future_return(test_df, periods=1)
        
        # 获取该品种的核心特征
        core_features = core_features_dict[symbol]
        logger.info(f"使用 {len(core_features)} 个核心特征进行训练")
        
        # 提取核心特征
        X_train_core = train_df[core_features].copy()
        X_test_core = test_df[core_features].copy()
        
        # 删除目标为NaN的行
        train_valid_mask = y_train.notna()
        test_valid_mask = y_test.notna()
        
        X_train_core = X_train_core[train_valid_mask]
        y_train_valid = y_train[train_valid_mask]
        
        X_test_core = X_test_core[test_valid_mask]
        y_test_valid = y_test[test_valid_mask]
        
        logger.info(f"有效训练样本: {len(X_train_core)}")
        logger.info(f"有效测试样本: {len(X_test_core)}")
        
        # 训练多个模型
        symbol_models = {}
        symbol_metrics = {}
        
        for model_type in ['ridge', 'lasso', 'rf']:
            logger.info(f"\n训练 {model_type.upper()} 模型...")
            
            try:
                model, metrics = train_model(
                    X_train_core, y_train_valid,
                    X_test_core, y_test_valid,
                    model_type=model_type
                )
                
                symbol_models[model_type] = model
                symbol_metrics[model_type] = metrics
                
                logger.info(f"✓ {model_type.upper()} 训练完成")
                logger.info(f"  训练集 R²: {metrics['train']['r2']:.4f}, RMSE: {metrics['train']['rmse']:.6f}")
                logger.info(f"  测试集 R²: {metrics['test']['r2']:.4f}, RMSE: {metrics['test']['rmse']:.6f}")
                
            except Exception as e:
                logger.error(f"✗ {model_type.upper()} 训练失败: {e}")
        
        models_dict[symbol] = symbol_models
        metrics_dict[symbol] = symbol_metrics
    
    # 6. 保存模型
    logger.info("\n步骤6: 保存训练好的模型")
    
    models_dir = Path("models/checkpoints/core_feature_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for symbol, symbol_models in models_dict.items():
        for model_type, model in symbol_models.items():
            model_path = models_dir / f"{symbol}_{model_type}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"✓ {symbol} {model_type.upper()} 模型已保存: {model_path}")
    
    # 7. 保存核心特征列表（供推理使用）
    logger.info("\n步骤7: 保存核心特征配置")
    
    feature_config = {
        'all_features': all_feature_names,
        'core_features': core_features_dict,
        'metadata': {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': '模型训练使用的核心特征配置'
        }
    }
    
    config_path = models_dir / "feature_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 特征配置已保存: {config_path}")
    
    # 8. 生成训练报告
    logger.info("\n步骤8: 生成训练报告")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("模型训练报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("训练配置:")
    report_lines.append("-" * 80)
    report_lines.append(f"所有手工特征数量: {len(all_feature_names)}")
    report_lines.append(f"品种数量: {len(symbols)}")
    report_lines.append(f"模型类型: Ridge, Lasso, Random Forest")
    report_lines.append("")
    
    report_lines.append("核心特征使用情况:")
    report_lines.append("-" * 80)
    for symbol, features in core_features_dict.items():
        report_lines.append(f"{symbol}: {len(features)}/{len(all_feature_names)} 个核心特征")
    report_lines.append("")
    
    report_lines.append("模型性能:")
    report_lines.append("-" * 80)
    
    for symbol in symbols:
        if symbol not in metrics_dict:
            continue
        
        report_lines.append(f"\n品种: {symbol}")
        report_lines.append("-" * 40)
        
        symbol_metrics = metrics_dict[symbol]
        
        for model_type, metrics in symbol_metrics.items():
            report_lines.append(f"\n{model_type.upper()} 模型:")
            report_lines.append(f"  训练集:")
            report_lines.append(f"    R²: {metrics['train']['r2']:.4f}")
            report_lines.append(f"    RMSE: {metrics['train']['rmse']:.6f}")
            report_lines.append(f"    MAE: {metrics['train']['mae']:.6f}")
            report_lines.append(f"  测试集:")
            report_lines.append(f"    R²: {metrics['test']['r2']:.4f}")
            report_lines.append(f"    RMSE: {metrics['test']['rmse']:.6f}")
            report_lines.append(f"    MAE: {metrics['test']['mae']:.6f}")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("重要说明:")
    report_lines.append("-" * 80)
    report_lines.append("1. 模型训练时使用核心特征（从JSON文件加载）")
    report_lines.append("2. 特征验证时保留所有27个手工特征")
    report_lines.append("3. 核心特征是基于特征验证结果选择的最重要特征")
    report_lines.append("4. 模型和特征配置已保存，可用于推理")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_path = "training/output/05_training_report.txt"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"\n训练报告已保存: {report_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("模型训练流程完成！")
    logger.info("=" * 80)
    logger.info("\n训练摘要:")
    logger.info(f"  - 使用核心特征训练模型")
    logger.info(f"  - 保留所有27个手工特征用于验证")
    logger.info(f"  - 训练了 {len(symbols)} 个品种的模型")
    logger.info(f"  - 每个品种训练了 3 种模型 (Ridge, Lasso, RF)")
    logger.info(f"  - 模型已保存到: {models_dir}")


if __name__ == "__main__":
    main()