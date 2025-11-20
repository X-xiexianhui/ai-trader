#!/usr/bin/env python3
"""
AI交易系统运行脚本

加载已训练的模型并进行实时推理
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.api.inference_service import LocalInferenceService
from src.api.monitoring import SystemMonitor, MonitoringConfig, PerformanceDashboard
from src.data.downloader import YahooFinanceDownloader
from src.features.pipeline import FeatureEngineeringPipeline
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class TradingBot:
    """交易机器人"""
    
    def __init__(
        self,
        model_dir: str = "models",
        device: str = "auto",
        enable_monitoring: bool = True
    ):
        """
        初始化交易机器人
        
        Args:
            model_dir: 模型目录
            device: 设备 (auto/cuda/cpu)
            enable_monitoring: 是否启用监控
        """
        self.project_root = Path(__file__).parent
        
        # 设置日志
        setup_logger(
            log_dir=self.project_root / "logs" / "inference",
            log_level=logging.INFO
        )
        
        # 初始化推理服务
        logger.info("Loading models...")
        self.inference_service = LocalInferenceService(
            model_dir=model_dir,
            device=device
        )
        
        # 初始化监控
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            config = MonitoringConfig()
            self.monitor = SystemMonitor(config)
            self.dashboard = PerformanceDashboard(self.monitor)
        
        # 初始化特征工程
        self.feature_pipeline = FeatureEngineeringPipeline()
        
        # 初始化数据下载器
        self.downloader = YahooFinanceDownloader()
        
        logger.info("Trading bot initialized successfully")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载OHLCV数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            data: 市场数据
        """
        logger.info(f"Loading data from {data_path}...")
        
        if data_path.endswith('.parquet'):
            data = pd.read_parquet(data_path)
        else:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        logger.info(f"Data loaded: {len(data)} bars")
        return data
    
    def prepare_input(self, data: pd.DataFrame) -> tuple:
        """
        准备模型输入
        
        Args:
            data: 原始市场数据
            
        Returns:
            market_data: 市场数据 (256, 4)
            features: 手工特征 (27,)
        """
        # 提取特征
        features_df = self.feature_pipeline.transform(data)
        
        # 准备TS2Vec输入（最近256个时间步的OHLC）
        market_data = data[['Open', 'High', 'Low', 'Close']].tail(256).values
        
        # 准备手工特征（最新的27维特征）
        features = features_df.iloc[-1].values
        
        return market_data, features
    
    def predict_once(self, data: pd.DataFrame, symbol: str = "Unknown") -> dict:
        """
        执行一次预测
        
        Args:
            data: OHLCV数据
            symbol: 交易品种名称
            
        Returns:
            signal: 交易信号
        """
        try:
            
            # 准备输入
            market_data, features = self.prepare_input(data)
            
            # 执行推理
            start_time = time.time()
            signal = self.inference_service.predict(market_data, features)
            latency = time.time() - start_time
            
            # 记录监控
            if self.enable_monitoring:
                self.monitor.record_request(latency, success=True)
            
            # 添加额外信息
            signal['symbol'] = symbol
            signal['current_price'] = float(data['Close'].iloc[-1])
            signal['timestamp'] = datetime.now().isoformat()
            
            return signal
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            
            if self.enable_monitoring:
                self.monitor.record_request(0, success=False)
            
            return {
                "error": str(e),
                "symbol": symbol,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_continuous(
        self,
        data_path: str,
        symbol: str = "Unknown",
        interval_seconds: int = 300,
        max_iterations: int = None
    ):
        """
        持续运行模式（需要定期更新数据文件）
        
        Args:
            data_path: OHLCV数据文件路径
            symbol: 交易品种名称
            interval_seconds: 预测间隔（秒）
            max_iterations: 最大迭代次数（None表示无限）
        """
        logger.info("=" * 80)
        logger.info(f"Starting continuous trading mode for {symbol}")
        logger.info(f"Data source: {data_path}")
        logger.info(f"Prediction interval: {interval_seconds} seconds")
        logger.info("=" * 80)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # 重新加载数据（假设数据文件会被外部更新）
                logger.info(f"\n[Iteration {iteration}] Loading latest data...")
                data = self.load_data(data_path)
                
                # 执行预测
                logger.info(f"Generating signal...")
                signal = self.predict_once(data, symbol)
                
                # 显示信号
                if "error" not in signal:
                    logger.info(f"Signal generated:")
                    logger.info(f"  Symbol: {signal['symbol']}")
                    logger.info(f"  Current Price: ${signal['current_price']:.2f}")
                    logger.info(f"  Direction: {signal['direction'].upper()}")
                    logger.info(f"  Position Size: {signal['position_size']:.2%}")
                    logger.info(f"  Stop Loss: {signal['stop_loss']:.2%}")
                    logger.info(f"  Take Profit: {signal['take_profit']:.2%}")
                    logger.info(f"  Confidence: {signal['confidence']:.2%}")
                    logger.info(f"  Latency: {signal['latency_ms']:.2f}ms")
                else:
                    logger.error(f"Error: {signal['error']}")
                
                # 显示监控信息
                if self.enable_monitoring and iteration % 10 == 0:
                    logger.info("\n" + "=" * 80)
                    self.dashboard.print_dashboard()
                    logger.info("=" * 80)
                    
                    # 检查告警
                    alerts = self.monitor.check_all()
                    if alerts:
                        logger.warning(f"\n⚠️  {len(alerts)} alert(s) detected!")
                
                # 检查是否达到最大迭代次数
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"\nReached maximum iterations ({max_iterations})")
                    break
                
                # 等待下一次预测
                logger.info(f"\nWaiting {interval_seconds} seconds for next prediction...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("\n\nStopped by user")
        
        # 显示最终统计
        if self.enable_monitoring:
            logger.info("\n" + "=" * 80)
            logger.info("Final Statistics:")
            logger.info("=" * 80)
            self.dashboard.print_dashboard()
    
    def run_backtest(
        self,
        data_path: str,
        symbol: str = "Unknown"
    ):
        """
        回测模式
        
        Args:
            data_path: OHLCV数据文件路径
            symbol: 交易品种名称
        """
        logger.info("=" * 80)
        logger.info(f"Starting backtest mode for {symbol}")
        logger.info(f"Data source: {data_path}")
        logger.info("=" * 80)
        
        # 加载历史数据
        data = self.load_data(data_path)
        
        # 滚动预测
        signals = []
        window_size = 256
        
        for i in range(window_size, len(data)):
            if i % 100 == 0:
                logger.info(f"Processing: {i}/{len(data)} ({i/len(data)*100:.1f}%)")
            
            # 准备输入
            window_data = data.iloc[i-window_size:i]
            market_data, features = self.prepare_input(window_data)
            
            # 预测
            signal = self.inference_service.predict(market_data, features)
            signal['timestamp'] = data.index[i]
            signal['price'] = data['Close'].iloc[i]
            signals.append(signal)
        
        # 保存结果
        output_path = self.project_root / "logs" / "backtest" / f"signals_{symbol}_{start_date}_{end_date}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        signals_df = pd.DataFrame(signals)
        signals_df.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Backtest complete!")
        logger.info(f"Signals saved to: {output_path}")
        logger.info(f"Total signals: {len(signals)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AI Trading System - Run trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单次预测
  python run.py --mode once --data data/raw/ES_F.parquet --symbol ES=F
  
  # 持续运行（每5分钟预测一次，需要外部更新数据文件）
  python run.py --mode continuous --data data/raw/ES_F.parquet --symbol ES=F --interval 300
  
  # 回测模式
  python run.py --mode backtest --data data/raw/ES_F_historical.parquet --symbol ES=F
  
  # 使用CPU运行
  python run.py --device cpu --mode once --data data/raw/ES_F.parquet
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["once", "continuous", "backtest"],
        default="once",
        help="Running mode"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="ES=F",
        help="Trading symbol"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Prediction interval in seconds (for continuous mode)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum iterations (for continuous mode)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default="2023-12-31",
        help="End date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable monitoring"
    )
    
    args = parser.parse_args()
    
    # 创建交易机器人
    bot = TradingBot(
        model_dir="models",
        device=args.device,
        enable_monitoring=not args.no_monitoring
    )
    
    # 根据模式运行
    if args.mode == "once":
        signal = bot.predict_once(symbol=args.symbol)
        
        if "error" not in signal:
            print("\n" + "=" * 60)
            print("TRADING SIGNAL")
            print("=" * 60)
            print(f"Symbol:        {signal['symbol']}")
            print(f"Current Price: ${signal['current_price']:.2f}")
            print(f"Direction:     {signal['direction'].upper()}")
            print(f"Position Size: {signal['position_size']:.2%}")
            print(f"Stop Loss:     {signal['stop_loss']:.2%}")
            print(f"Take Profit:   {signal['take_profit']:.2%}")
            print(f"Confidence:    {signal['confidence']:.2%}")
            print(f"Latency:       {signal['latency_ms']:.2f}ms")
            print(f"Timestamp:     {signal['timestamp']}")
            print("=" * 60)
        else:
            print(f"\nError: {signal['error']}")
            sys.exit(1)
    
    elif args.mode == "continuous":
        bot.run_continuous(
            symbol=args.symbol,
            interval_seconds=args.interval,
            max_iterations=args.max_iterations
        )
    
    elif args.mode == "backtest":
        bot.run_backtest(
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end
        )


if __name__ == "__main__":
    main()