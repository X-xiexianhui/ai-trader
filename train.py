#!/usr/bin/env python3
"""
AI交易系统训练脚本

一键启动完整的模型训练流程
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """训练流水线"""
    
    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        初始化训练流水线
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent
        
        # 设置日志
        setup_logger(
            log_dir=self.project_root / "logs" / "training",
            log_level=logging.INFO
        )
        
        logger.info("Training pipeline initialized")
    
    def prepare_data(self, data_path: str = None):
        """
        准备数据
        
        Args:
            data_path: OHLCV数据文件路径（parquet格式）
        """
        logger.info("=" * 60)
        logger.info("Step 1: Preparing data...")
        logger.info("=" * 60)
        
        from src.data.cleaner import DataCleaningPipeline
        import pandas as pd
        
        if data_path is None:
            # 如果没有提供数据路径，查找已有数据
            data_dir = self.project_root / "data" / "raw"
            data_files = list(data_dir.glob("*.parquet")) + list(data_dir.glob("*.csv"))
            
            if not data_files:
                logger.error("No data files found in data/raw/")
                logger.error("Please provide OHLCV data file path or place data in data/raw/")
                return False
            
            logger.info(f"Found {len(data_files)} data file(s)")
            
            for data_file in data_files:
                logger.info(f"Processing {data_file.name}...")
                
                # 加载数据
                if data_file.suffix == '.parquet':
                    data = pd.read_parquet(data_file)
                else:
                    data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                
                # 清洗数据
                cleaner = DataCleaningPipeline()
                cleaned_data = cleaner.clean(data)
                
                # 保存
                output_path = self.project_root / "data" / "processed" / f"{data_file.stem}_cleaned.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cleaned_data.to_parquet(output_path)
                logger.info(f"Saved cleaned data to {output_path}")
        else:
            # 使用提供的数据路径
            logger.info(f"Loading data from {data_path}...")
            
            if data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            
            # 清洗数据
            cleaner = DataCleaningPipeline()
            cleaned_data = cleaner.clean(data)
            
            # 保存
            output_path = self.project_root / "data" / "processed" / "cleaned_data.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_data.to_parquet(output_path)
            logger.info(f"Saved cleaned data to {output_path}")
        
        logger.info("✓ Data preparation complete")
        return True
    
    def extract_features(self):
        """提取特征"""
        logger.info("=" * 60)
        logger.info("Step 2: Extracting features...")
        logger.info("=" * 60)
        
        from src.features.pipeline import FeatureEngineeringPipeline
        import pandas as pd
        
        pipeline = FeatureEngineeringPipeline()
        
        # 处理所有数据文件
        data_dir = self.project_root / "data" / "processed"
        for data_file in data_dir.glob("*_cleaned.parquet"):
            logger.info(f"Processing {data_file.name}...")
            
            data = pd.read_parquet(data_file)
            features = pipeline.transform(data)
            
            # 保存特征
            output_path = data_file.parent / f"{data_file.stem}_features.parquet"
            features.to_parquet(output_path)
            logger.info(f"Saved features to {output_path}")
        
        logger.info("✓ Feature extraction complete")
    
    def train_ts2vec(self):
        """训练TS2Vec模型"""
        logger.info("=" * 60)
        logger.info("Step 3: Training TS2Vec model...")
        logger.info("=" * 60)
        
        from src.models.ts2vec.trainer import TS2VecTrainer
        from src.models.ts2vec.dataset import TS2VecDataset
        import pandas as pd
        
        # 加载数据
        data_dir = self.project_root / "data" / "processed"
        all_data = []
        for data_file in data_dir.glob("*_features.parquet"):
            data = pd.read_parquet(data_file)
            all_data.append(data)
        
        if not all_data:
            logger.error("No feature data found!")
            return False
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 创建数据集
        dataset = TS2VecDataset(combined_data, window_size=256)
        
        # 训练
        trainer = TS2VecTrainer(
            config_path=str(self.project_root / "configs" / "ts2vec_config.yaml")
        )
        
        trainer.train(dataset, epochs=50)
        
        # 保存模型
        model_path = self.project_root / "models" / "ts2vec" / "best_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        
        logger.info(f"✓ TS2Vec model saved to {model_path}")
        return True
    
    def train_transformer(self):
        """训练Transformer模型"""
        logger.info("=" * 60)
        logger.info("Step 4: Training Transformer model...")
        logger.info("=" * 60)
        
        from src.models.transformer.trainer import TransformerTrainer
        from src.models.transformer.dataset import TransformerDataset
        import pandas as pd
        
        # 加载数据和TS2Vec embeddings
        data_dir = self.project_root / "data" / "processed"
        all_data = []
        for data_file in data_dir.glob("*_features.parquet"):
            data = pd.read_parquet(data_file)
            all_data.append(data)
        
        if not all_data:
            logger.error("No feature data found!")
            return False
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 创建数据集
        dataset = TransformerDataset(
            data=combined_data,
            ts2vec_model_path=str(self.project_root / "models" / "ts2vec" / "best_model.pt"),
            sequence_length=64
        )
        
        # 训练
        trainer = TransformerTrainer(
            config_path=str(self.project_root / "configs" / "transformer_config.yaml")
        )
        
        trainer.train(dataset, epochs=30)
        
        # 保存模型
        model_path = self.project_root / "models" / "transformer" / "best_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        
        logger.info(f"✓ Transformer model saved to {model_path}")
        return True
    
    def train_ppo(self):
        """训练PPO策略"""
        logger.info("=" * 60)
        logger.info("Step 5: Training PPO policy...")
        logger.info("=" * 60)
        
        from src.models.ppo.trainer import PPOTrainer
        from src.models.ppo.environment import TradingEnvironment
        import pandas as pd
        
        # 加载数据
        data_dir = self.project_root / "data" / "processed"
        all_data = []
        for data_file in data_dir.glob("*_features.parquet"):
            data = pd.read_parquet(data_file)
            all_data.append(data)
        
        if not all_data:
            logger.error("No feature data found!")
            return False
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 创建环境
        env = TradingEnvironment(
            data=combined_data,
            transformer_model_path=str(self.project_root / "models" / "transformer" / "best_model.pt")
        )
        
        # 训练
        trainer = PPOTrainer(
            env=env,
            config_path=str(self.project_root / "configs" / "ppo_config.yaml")
        )
        
        trainer.train(total_timesteps=1000000)
        
        # 保存模型
        model_path = self.project_root / "models" / "ppo" / "best_model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(model_path))
        
        logger.info(f"✓ PPO policy saved to {model_path}")
        return True
    
    def run_full_pipeline(self, data_path: str = None):
        """
        运行完整训练流程
        
        Args:
            data_path: OHLCV数据文件路径
        """
        logger.info("=" * 80)
        logger.info("Starting full training pipeline...")
        logger.info("=" * 80)
        
        try:
            # 1. 准备数据
            if not self.prepare_data(data_path):
                return False
            
            # 2. 提取特征
            self.extract_features()
            
            # 3. 训练TS2Vec
            if not self.train_ts2vec():
                logger.error("TS2Vec training failed!")
                return False
            
            # 4. 训练Transformer
            if not self.train_transformer():
                logger.error("Transformer training failed!")
                return False
            
            # 5. 训练PPO
            if not self.train_ppo():
                logger.error("PPO training failed!")
                return False
            
            logger.info("=" * 80)
            logger.info("✓ Full training pipeline completed successfully!")
            logger.info("=" * 80)
            
            # 显示模型位置
            logger.info("\nTrained models saved to:")
            logger.info(f"  - TS2Vec: models/ts2vec/best_model.pt")
            logger.info(f"  - Transformer: models/transformer/best_model.pt")
            logger.info(f"  - PPO: models/ppo/best_model.pt")
            
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI Trading System Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行完整训练流程
  python train.py
  
  # 使用自定义配置
  python train.py --config configs/my_config.yaml
  
  # 只训练特定模型
  python train.py --model ts2vec
  python train.py --model transformer
  python train.py --model ppo
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["ts2vec", "transformer", "ppo", "all"],
        default="all",
        help="Which model to train"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to OHLCV data file (parquet or csv format)"
    )
    
    args = parser.parse_args()
    
    # 创建训练流水线
    pipeline = TrainingPipeline(config_path=args.config)
    
    # 根据参数执行训练
    if args.model == "all":
        success = pipeline.run_full_pipeline(data_path=args.data)
    elif args.model == "ts2vec":
        if not pipeline.prepare_data(data_path=args.data):
            sys.exit(1)
        pipeline.extract_features()
        success = pipeline.train_ts2vec()
    elif args.model == "transformer":
        success = pipeline.train_transformer()
    elif args.model == "ppo":
        success = pipeline.train_ppo()
    
    if success:
        logger.info("\n✓ Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n✗ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()