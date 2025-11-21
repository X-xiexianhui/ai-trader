"""
任务7.2.2: 训练脚本

统一的模型训练脚本，支持TS2Vec、Transformer和PPO的训练
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger, TrainingLogger
from src.pipeline.training_pipeline import TrainingDataPipeline
from src.models.ts2vec.model import TS2Vec
from src.models.ts2vec.training import TS2VecTrainer
from src.models.transformer.model import TransformerStateModel
from src.models.transformer.training import TransformerTrainer
from src.models.ppo.model import PPOAgent
from src.models.ppo.trainer import PPOTrainer


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_ts2vec(config: dict, logger):
    """训练TS2Vec模型"""
    logger.info("=" * 50)
    logger.info("开始训练TS2Vec模型")
    logger.info("=" * 50)
    
    # 创建训练日志记录器
    training_logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        experiment_name='ts2vec_training',
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    try:
        # 准备数据
        logger.info("准备训练数据...")
        pipeline = TrainingDataPipeline(
            ts2vec_model_path=None,  # TS2Vec训练不需要预训练模型
            scaler_path=None,
            config=config
        )
        
        # 加载原始数据
        from src.data.downloader import DataDownloader
        downloader = DataDownloader()
        
        data = {}
        for symbol in config['data']['symbols']:
            logger.info(f"下载数据: {symbol}")
            df = downloader.download(
                symbol=symbol,
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date'],
                interval=config['data']['interval']
            )
            data[symbol] = df
        
        # 处理数据（仅到特征计算和归一化）
        logger.info("处理数据...")
        processed_data = {}
        for symbol, df in data.items():
            # 清洗数据
            cleaned_df = pipeline._clean_data(df)
            # 计算特征
            features_df = pipeline._calculate_features(cleaned_df)
            # 归一化
            normalized_df, scaler = pipeline._normalize_features(features_df, fit=True)
            processed_data[symbol] = normalized_df
            
            # 保存scaler
            if symbol == config['data']['symbols'][0]:  # 使用第一个品种的scaler
                pipeline.scaler = scaler
                pipeline.save_scaler()
        
        # 创建TS2Vec模型
        logger.info("创建TS2Vec模型...")
        model = TS2Vec(
            input_dim=4,  # OHLC
            hidden_dim=config['ts2vec']['hidden_dim'],
            output_dim=config['ts2vec']['output_dim']
        )
        
        # 创建训练器
        trainer = TS2VecTrainer(
            model=model,
            config=config['ts2vec'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 记录超参数
        training_logger.log_hyperparameters(config['ts2vec'])
        
        # 训练模型
        logger.info("开始训练...")
        trainer.train(
            data=processed_data,
            training_logger=training_logger
        )
        
        # 保存模型
        model_path = Path(config['checkpoints']['ts2vec_dir']) / 'ts2vec_final.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存: {model_path}")
        
        logger.info("TS2Vec训练完成!")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    finally:
        training_logger.close()


def train_transformer(config: dict, logger):
    """训练Transformer模型"""
    logger.info("=" * 50)
    logger.info("开始训练Transformer模型")
    logger.info("=" * 50)
    
    # 创建训练日志记录器
    training_logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        experiment_name='transformer_training',
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    try:
        # 准备数据
        logger.info("准备训练数据...")
        
        # 加载TS2Vec模型
        ts2vec_model_path = Path(config['checkpoints']['ts2vec_dir']) / 'ts2vec_final.pth'
        if not ts2vec_model_path.exists():
            raise FileNotFoundError(f"TS2Vec模型不存在: {ts2vec_model_path}")
        
        # 加载scaler
        scaler_path = Path(config['checkpoints']['scaler_dir']) / 'feature_scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler不存在: {scaler_path}")
        
        # 创建数据管道
        pipeline = TrainingDataPipeline(
            ts2vec_model_path=str(ts2vec_model_path),
            scaler_path=str(scaler_path),
            config=config
        )
        
        # 下载数据
        from src.data.downloader import DataDownloader
        downloader = DataDownloader()
        
        logger.info("下载数据...")
        symbol = config['data']['symbols'][0]  # 使用第一个品种
        df = downloader.download(
            symbol=symbol,
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            interval=config['data']['interval']
        )
        
        # 处理数据
        logger.info("处理数据...")
        train_data, val_data, test_data = pipeline.process(df)
        
        logger.info(f"训练集: {len(train_data['sequences'])} 样本")
        logger.info(f"验证集: {len(val_data['sequences'])} 样本")
        logger.info(f"测试集: {len(test_data['sequences'])} 样本")
        
        # 创建Transformer模型
        logger.info("创建Transformer模型...")
        model = TransformerStateModel(
            input_dim=config['transformer']['input_dim'],
            d_model=config['transformer']['d_model'],
            nhead=config['transformer']['nhead'],
            num_layers=config['transformer']['num_layers'],
            dim_feedforward=config['transformer']['dim_feedforward'],
            dropout=config['transformer']['dropout']
        )
        
        # 创建训练器
        trainer = TransformerTrainer(
            model=model,
            config=config['transformer'],
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 记录超参数
        training_logger.log_hyperparameters(config['transformer'])
        
        # 训练模型
        logger.info("开始训练...")
        trainer.train(
            train_data=train_data,
            val_data=val_data,
            training_logger=training_logger
        )
        
        # 保存模型
        model_path = Path(config['checkpoints']['transformer_dir']) / 'transformer_final.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"模型已保存: {model_path}")
        
        logger.info("Transformer训练完成!")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    finally:
        training_logger.close()


def train_ppo(config: dict, logger):
    """训练PPO模型"""
    logger.info("=" * 50)
    logger.info("开始训练PPO模型")
    logger.info("=" * 50)
    
    # 创建训练日志记录器
    training_logger = TrainingLogger(
        log_dir=config['logging']['log_dir'],
        experiment_name='ppo_training',
        use_tensorboard=config['logging']['use_tensorboard']
    )
    
    try:
        # 加载Transformer模型
        transformer_model_path = Path(config['checkpoints']['transformer_dir']) / 'transformer_final.pth'
        if not transformer_model_path.exists():
            raise FileNotFoundError(f"Transformer模型不存在: {transformer_model_path}")
        
        logger.info("加载Transformer模型...")
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
        
        # 创建PPO智能体
        logger.info("创建PPO智能体...")
        agent = PPOAgent(
            state_dim=config['ppo']['state_dim'],
            action_dim=config['ppo']['action_dim'],
            hidden_dim=config['ppo']['hidden_dim'],
            lr=config['ppo']['learning_rate']
        )
        
        # 创建训练器
        trainer = PPOTrainer(
            agent=agent,
            transformer_model=transformer_model,
            config=config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 记录超参数
        training_logger.log_hyperparameters(config['ppo'])
        
        # 训练模型
        logger.info("开始训练...")
        trainer.train(training_logger=training_logger)
        
        # 保存模型
        model_path = Path(config['checkpoints']['ppo_dir']) / 'ppo_final.pth'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(model_path))
        logger.info(f"模型已保存: {model_path}")
        
        logger.info("PPO训练完成!")
        
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    finally:
        training_logger.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI交易系统训练脚本')
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
        help='要训练的模型'
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
        name='train',
        log_level=args.log_level,
        log_dir='logs',
        log_file='train.log'
    )
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config = load_config(args.config)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备: {device}")
        
        # 训练模型
        if args.model in ['ts2vec', 'all']:
            train_ts2vec(config, logger)
        
        if args.model in ['transformer', 'all']:
            train_transformer(config, logger)
        
        if args.model in ['ppo', 'all']:
            train_ppo(config, logger)
        
        logger.info("=" * 50)
        logger.info("所有训练任务完成!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()