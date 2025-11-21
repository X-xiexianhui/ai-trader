"""
任务7.2.4: 日志记录模块

实现统一的日志记录系统，支持多级别、多目标输出
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import colorlog


def setup_logger(
    name: str = 'ai_trader',
    log_level: str = 'INFO',
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_dir: 日志目录
        log_file: 日志文件名
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
        format_string: 自定义格式字符串
        
    Returns:
        配置好的logger对象
    """
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的handlers
    logger.handlers.clear()
    
    # 默认格式
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    # 控制台输出（带颜色）
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 彩色格式
        color_format = (
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d%(reset)s - %(message)s'
        )
        
        color_formatter = colorlog.ColoredFormatter(
            color_format,
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
    
    # 文件输出（带轮转）
    if file_output:
        if log_dir is None:
            log_dir = 'logs'
        
        if log_file is None:
            log_file = f'{name}.log'
        
        # 创建日志目录
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 完整日志文件路径
        full_log_path = log_path / log_file
        
        # 使用RotatingFileHandler实现日志轮转
        file_handler = RotatingFileHandler(
            full_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 文件格式（不带颜色）
        file_formatter = logging.Formatter(
            format_string,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件: {full_log_path}")
    
    # 防止日志传播到父logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = 'ai_trader') -> logging.Logger:
    """
    获取已配置的logger
    
    Args:
        name: logger名称
        
    Returns:
        logger对象
    """
    return logging.getLogger(name)


class TrainingLogger:
    """
    训练过程专用日志记录器
    
    记录训练指标、损失曲线等
    """
    
    def __init__(self,
                 log_dir: str = 'logs',
                 experiment_name: str = 'experiment',
                 use_tensorboard: bool = True):
        """
        初始化训练日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            use_tensorboard: 是否使用TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # 创建实验目录
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础logger
        self.logger = setup_logger(
            name=f'training_{experiment_name}',
            log_dir=str(self.experiment_dir),
            log_file='training.log'
        )
        
        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tensorboard_dir = self.experiment_dir / 'tensorboard'
                self.writer = SummaryWriter(str(tensorboard_dir))
                self.logger.info(f"TensorBoard日志: {tensorboard_dir}")
            except ImportError:
                self.logger.warning("未安装tensorboard，跳过TensorBoard日志")
        
        # 指标历史
        self.metrics_history = {}
    
    def log_metrics(self, metrics: dict, step: int, prefix: str = ''):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀（如'train/', 'val/'）
        """
        # 记录到文本日志
        metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {prefix}{metrics_str}")
        
        # 记录到TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{prefix}{key}', value, step)
        
        # 保存到历史
        for key, value in metrics.items():
            full_key = f'{prefix}{key}'
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append((step, value))
    
    def log_hyperparameters(self, hparams: dict):
        """
        记录超参数
        
        Args:
            hparams: 超参数字典
        """
        self.logger.info("=" * 50)
        self.logger.info("超参数:")
        for key, value in hparams.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)
        
        if self.writer is not None:
            # TensorBoard记录超参数
            self.writer.add_hparams(hparams, {})
    
    def log_model_summary(self, model, input_size):
        """
        记录模型结构摘要
        
        Args:
            model: PyTorch模型
            input_size: 输入尺寸
        """
        try:
            from torchsummary import summary
            import io
            import sys
            
            # 捕获summary输出
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            summary(model, input_size)
            
            sys.stdout = old_stdout
            model_summary = buffer.getvalue()
            
            self.logger.info("\n模型结构:\n" + model_summary)
        except ImportError:
            self.logger.warning("未安装torchsummary，跳过模型摘要")
    
    def save_metrics(self):
        """保存指标历史到文件"""
        import json
        
        metrics_file = self.experiment_dir / 'metrics_history.json'
        
        # 转换为可序列化格式
        serializable_metrics = {
            key: [(int(step), float(value)) for step, value in values]
            for key, values in self.metrics_history.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        self.logger.info(f"指标历史已保存: {metrics_file}")
    
    def close(self):
        """关闭日志记录器"""
        if self.writer is not None:
            self.writer.close()
        
        self.save_metrics()
        self.logger.info("训练日志记录器已关闭")


class EvaluationLogger:
    """
    评估过程专用日志记录器
    """
    
    def __init__(self,
                 log_dir: str = 'logs',
                 evaluation_name: str = 'evaluation'):
        """
        初始化评估日志记录器
        
        Args:
            log_dir: 日志目录
            evaluation_name: 评估名称
        """
        self.log_dir = Path(log_dir)
        self.evaluation_name = evaluation_name
        
        # 创建评估目录
        self.evaluation_dir = self.log_dir / evaluation_name
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础logger
        self.logger = setup_logger(
            name=f'evaluation_{evaluation_name}',
            log_dir=str(self.evaluation_dir),
            log_file='evaluation.log'
        )
        
        # 结果存储
        self.results = {}
    
    def log_results(self, results: dict, category: str = 'general'):
        """
        记录评估结果
        
        Args:
            results: 结果字典
            category: 结果类别
        """
        self.logger.info(f"\n{category}评估结果:")
        self.logger.info("=" * 50)
        
        for key, value in results.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 50)
        
        # 保存结果
        if category not in self.results:
            self.results[category] = {}
        self.results[category].update(results)
    
    def save_results(self):
        """保存所有结果到文件"""
        import json
        
        results_file = self.evaluation_dir / 'evaluation_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"评估结果已保存: {results_file}")
    
    def close(self):
        """关闭日志记录器"""
        self.save_results()
        self.logger.info("评估日志记录器已关闭")


# 便捷函数
def log_exception(logger: logging.Logger, exception: Exception, context: str = ''):
    """
    记录异常信息
    
    Args:
        logger: logger对象
        exception: 异常对象
        context: 上下文信息
    """
    import traceback
    
    error_msg = f"{context}\n" if context else ""
    error_msg += f"异常类型: {type(exception).__name__}\n"
    error_msg += f"异常信息: {str(exception)}\n"
    error_msg += f"堆栈跟踪:\n{traceback.format_exc()}"
    
    logger.error(error_msg)