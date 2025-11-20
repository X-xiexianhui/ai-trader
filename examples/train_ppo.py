"""
PPO训练示例脚本

演示如何使用PPO模块训练交易策略。

Author: AI Trader Team
Date: 2025-11-20
"""

import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ppo import TradingEnvironment, ActorCritic, PPOTrainer, PPOEvaluator
from src.models.transformer import TransformerModel
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger('ppo_training', 'logs/ppo/training.log')
    logger.info("=" * 80)
    logger.info("PPO训练开始")
    logger.info("=" * 80)
    
    # 1. 加载配置
    logger.info("加载配置文件...")
    config = load_config('configs/ppo_config.yaml')
    
    # 2. 设置设备
    device = config['training']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA不可用，使用CPU")
        device = 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 3. 加载Transformer模型
    logger.info("加载Transformer模型...")
    transformer_path = config['data']['transformer_model_path']
    
    if not Path(transformer_path).exists():
        logger.error(f"Transformer模型不存在: {transformer_path}")
        logger.info("请先训练Transformer模型")
        return
    
    transformer = TransformerModel.load(transformer_path)
    transformer.to(device)
    transformer.eval()
    logger.info("Transformer模型加载成功")
    
    # 4. 准备训练数据
    logger.info("准备训练数据...")
    
    # 这里需要实际的数据加载逻辑
    # 示例：使用模拟数据
    num_steps = 10000
    state_dim = 256
    
    # 生成模拟状态向量（实际应该从Transformer获取）
    states = np.random.randn(num_steps, state_dim).astype(np.float32)
    
    # 生成模拟价格序列
    prices = np.cumsum(np.random.randn(num_steps) * 0.01) + 100
    
    logger.info(f"数据准备完成: {num_steps}步")
    
    # 5. 创建交易环境
    logger.info("创建交易环境...")
    env = TradingEnvironment(
        states=states,
        prices=prices,
        initial_balance=config['environment']['initial_balance'],
        max_position_size=config['environment']['max_position_size'],
        commission_rate=config['environment']['commission_rate'],
        slippage=config['environment']['slippage'],
        max_steps=config['environment']['max_steps']
    )
    logger.info("交易环境创建成功")
    
    # 6. 创建Actor-Critic模型
    logger.info("创建Actor-Critic模型...")
    model = ActorCritic(
        state_dim=config['model']['state_dim'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        share_features=config['model']['share_features']
    )
    
    param_count = model.count_parameters()
    logger.info(f"模型参数量: {param_count:,}")
    
    # 7. 创建训练器
    logger.info("创建PPO训练器...")
    trainer = PPOTrainer(
        env=env,
        model=model,
        config=config['training'],
        save_dir=config['save']['model_dir'],
        device=device
    )
    logger.info("训练器创建成功")
    
    # 8. 开始训练
    logger.info("=" * 80)
    logger.info("开始训练")
    logger.info("=" * 80)
    
    try:
        trainer.train(
            total_timesteps=config['training']['total_timesteps'],
            log_interval=config['training']['log_interval'],
            save_interval=config['training']['save_interval'],
            eval_interval=config['training']['eval_interval']
        )
        
        logger.info("=" * 80)
        logger.info("训练完成！")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        trainer.save_checkpoint('interrupted_model.pt')
        logger.info("模型已保存")
    
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        trainer.save_checkpoint('error_model.pt')
        logger.info("模型已保存")
        raise
    
    # 9. 评估最终模型
    logger.info("=" * 80)
    logger.info("评估最终模型")
    logger.info("=" * 80)
    
    evaluator = PPOEvaluator(
        model=trainer.model,
        env=env,
        device=device
    )
    
    # 评估性能
    metrics = evaluator.evaluate(
        num_episodes=config['evaluation']['num_episodes'],
        deterministic=config['evaluation']['deterministic']
    )
    
    # 分析动作
    action_analysis = evaluator.analyze_actions(num_episodes=10)
    
    # 生成报告
    report_path = Path(config['save']['model_dir']) / 'evaluation_report.txt'
    report = evaluator.generate_report(
        metrics=metrics,
        action_analysis=action_analysis,
        save_path=str(report_path)
    )
    
    print("\n" + report)
    
    # 绘制训练曲线
    plot_path = Path(config['save']['model_dir']) / 'training_curves.png'
    evaluator.plot_performance(
        metrics_history=trainer.training_history,
        save_path=str(plot_path)
    )
    
    logger.info("=" * 80)
    logger.info("所有任务完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()