"""
任务7.2.3: 评估脚本

统一的模型评估脚本，支持多种评估模式
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger, EvaluationLogger
from src.pipeline.training_pipeline import TrainingDataPipeline
from src.models.ts2vec.model import TS2Vec
from src.models.ts2vec.evaluation import TS2VecEvaluator
from src.models.transformer.model import TransformerStateModel
from src.models.transformer.evaluation import TransformerEvaluator
from src.models.ppo.model import PPOAgent
from src.models.ppo.metrics import PPOMetrics
from src.backtest.engine import BacktestEngine
from src.evaluation.walk_forward import WalkForwardValidator
from src.evaluation.overfitting_detection import OverfittingDetector
from src.evaluation.market_state import MarketStateAnalyzer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_ts2vec(config: dict, logger):
    """评估TS2Vec模型"""
    logger.info("=" * 50)
    logger.info("评估TS2Vec模型")
    logger.info("=" * 50)
    
    eval_logger = EvaluationLogger(
        log_dir=config['logging']['log_dir'],
        evaluation_name='ts2vec_evaluation'
    )
    
    try:
        # 加载模型
        model_path = Path(config['checkpoints']['ts2vec_dir']) / 'ts2vec_final.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        model = TS2Vec(
            input_dim=4,
            hidden_dim=config['ts2vec']['hidden_dim'],
            output_dim=config['ts2vec']['output_dim']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 准备测试数据
        logger.info("准备测试数据...")
        from src.data.downloader import DataDownloader
        downloader = DataDownloader()
        
        symbol = config['data']['symbols'][0]
        df = downloader.download(
            symbol=symbol,
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            interval=config['data']['interval']
        )
        
        # 创建评估器
        evaluator = TS2VecEvaluator(
            model=model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 评估embedding质量
        logger.info("评估embedding质量...")
        embedding_results = evaluator.evaluate_embedding_quality(df)
        eval_logger.log_results(embedding_results, category='embedding_quality')
        
        # 线性探测评估
        logger.info("线性探测评估...")
        probe_results = evaluator.linear_probe_evaluation(df)
        eval_logger.log_results(probe_results, category='linear_probe')
        
        # 聚类质量评估
        logger.info("聚类质量评估...")
        cluster_results = evaluator.clustering_evaluation(df)
        eval_logger.log_results(cluster_results, category='clustering')
        
        logger.info("TS2Vec评估完成!")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        raise
    finally:
        eval_logger.close()


def evaluate_transformer(config: dict, logger):
    """评估Transformer模型"""
    logger.info("=" * 50)
    logger.info("评估Transformer模型")
    logger.info("=" * 50)
    
    eval_logger = EvaluationLogger(
        log_dir=config['logging']['log_dir'],
        evaluation_name='transformer_evaluation'
    )
    
    try:
        # 加载模型
        model_path = Path(config['checkpoints']['transformer_dir']) / 'transformer_final.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        logger.info(f"加载模型: {model_path}")
        model = TransformerStateModel(
            input_dim=config['transformer']['input_dim'],
            d_model=config['transformer']['d_model'],
            nhead=config['transformer']['nhead'],
            num_layers=config['transformer']['num_layers'],
            dim_feedforward=config['transformer']['dim_feedforward'],
            dropout=config['transformer']['dropout']
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 准备测试数据
        logger.info("准备测试数据...")
        ts2vec_model_path = Path(config['checkpoints']['ts2vec_dir']) / 'ts2vec_final.pth'
        scaler_path = Path(config['checkpoints']['scaler_dir']) / 'feature_scaler.pkl'
        
        pipeline = TrainingDataPipeline(
            ts2vec_model_path=str(ts2vec_model_path),
            scaler_path=str(scaler_path),
            config=config
        )
        
        from src.data.downloader import DataDownloader
        downloader = DataDownloader()
        
        symbol = config['data']['symbols'][0]
        df = downloader.download(
            symbol=symbol,
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            interval=config['data']['interval']
        )
        
        train_data, val_data, test_data = pipeline.process(df)
        
        # 创建评估器
        evaluator = TransformerEvaluator(
            model=model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 评估监督学习指标
        logger.info("评估监督学习指标...")
        supervised_results = evaluator.evaluate_supervised_metrics(test_data)
        eval_logger.log_results(supervised_results, category='supervised_learning')
        
        # 评估状态表征质量
        logger.info("评估状态表征质量...")
        state_results = evaluator.evaluate_state_quality(test_data)
        eval_logger.log_results(state_results, category='state_representation')
        
        logger.info("Transformer评估完成!")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        raise
    finally:
        eval_logger.close()


def evaluate_ppo(config: dict, logger, mode: str = 'backtest'):
    """评估PPO模型"""
    logger.info("=" * 50)
    logger.info(f"评估PPO模型 (模式: {mode})")
    logger.info("=" * 50)
    
    eval_logger = EvaluationLogger(
        log_dir=config['logging']['log_dir'],
        evaluation_name=f'ppo_evaluation_{mode}'
    )
    
    try:
        # 加载PPO模型
        model_path = Path(config['checkpoints']['ppo_dir']) / 'ppo_final.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        logger.info(f"加载PPO模型: {model_path}")
        agent = PPOAgent(
            state_dim=config['ppo']['state_dim'],
            action_dim=config['ppo']['action_dim'],
            hidden_dim=config['ppo']['hidden_dim'],
            lr=config['ppo']['learning_rate']
        )
        agent.load(str(model_path))
        
        # 加载Transformer模型
        transformer_model_path = Path(config['checkpoints']['transformer_dir']) / 'transformer_final.pth'
        logger.info(f"加载Transformer模型: {transformer_model_path}")
        transformer_model = TransformerStateModel(
            input_dim=config['transformer']['input_dim'],
            d_model=config['transformer']['d_model'],
            nhead=config['transformer']['nhead'],
            num_layers=config['transformer']['num_layers'],
            dim_feedforward=config['transformer']['dim_feedforward'],
            dropout=config['transformer']['dropout']
        )
        transformer_model.load_state_dict(torch.load(transformer_model_path))
        transformer_model.eval()
        
        # 准备测试数据
        logger.info("准备测试数据...")
        from src.data.downloader import DataDownloader
        downloader = DataDownloader()
        
        symbol = config['data']['symbols'][0]
        df = downloader.download(
            symbol=symbol,
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            interval=config['data']['interval']
        )
        
        if mode == 'backtest':
            # 回测评估
            logger.info("运行回测...")
            engine = BacktestEngine(
                agent=agent,
                transformer_model=transformer_model,
                config=config
            )
            
            backtest_results = engine.run(df)
            eval_logger.log_results(backtest_results, category='backtest')
            
            # 计算交易性能指标
            logger.info("计算交易性能指标...")
            metrics = PPOMetrics()
            performance_results = metrics.calculate_trading_metrics(backtest_results)
            eval_logger.log_results(performance_results, category='trading_performance')
            
            # 计算风险调整收益指标
            logger.info("计算风险调整收益指标...")
            risk_results = metrics.calculate_risk_adjusted_returns(backtest_results)
            eval_logger.log_results(risk_results, category='risk_adjusted_returns')
            
        elif mode == 'walk_forward':
            # Walk-forward验证
            logger.info("运行Walk-forward验证...")
            validator = WalkForwardValidator(
                agent=agent,
                transformer_model=transformer_model,
                config=config['evaluation']['walk_forward']
            )
            
            wf_results = validator.validate(df)
            eval_logger.log_results(wf_results, category='walk_forward')
            
        elif mode == 'overfitting':
            # 过拟合检测
            logger.info("运行过拟合检测...")
            detector = OverfittingDetector(config=config)
            
            of_results = detector.detect(
                agent=agent,
                transformer_model=transformer_model,
                data=df
            )
            eval_logger.log_results(of_results, category='overfitting_detection')
            
        elif mode == 'market_state':
            # 市场状态泛化分析
            logger.info("运行市场状态分析...")
            analyzer = MarketStateAnalyzer(config=config)
            
            ms_results = analyzer.analyze(
                agent=agent,
                transformer_model=transformer_model,
                data=df
            )
            eval_logger.log_results(ms_results, category='market_state_analysis')
        
        logger.info("PPO评估完成!")
        
    except Exception as e:
        logger.error(f"评估失败: {str(e)}")
        raise
    finally:
        eval_logger.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI交易系统评估脚本')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['ts2vec', 'transformer', 'ppo', 'all'],
        default='all',
        help='要评估的模型'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'walk_forward', 'overfitting', 'market_state', 'all'],
        default='backtest',
        help='PPO评估模式'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger(
        name='evaluate',
        log_level=args.log_level,
        log_dir='logs',
        log_file='evaluate.log'
    )
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 评估模型
        if args.model in ['ts2vec', 'all']:
            evaluate_ts2vec(config, logger)
        
        if args.model in ['transformer', 'all']:
            evaluate_transformer(config, logger)
        
        if args.model in ['ppo', 'all']:
            if args.mode == 'all':
                # 运行所有PPO评估模式
                for mode in ['backtest', 'walk_forward', 'overfitting', 'market_state']:
                    evaluate_ppo(config, logger, mode=mode)
            else:
                evaluate_ppo(config, logger, mode=args.mode)
        
        logger.info("=" * 50)
        logger.info("所有评估任务完成!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"评估过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()