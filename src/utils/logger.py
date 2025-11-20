"""
日志系统
Logging System

统一的日志记录系统，支持多级别日志、控制台和文件输出、日志轮转
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


class Logger:
    """日志记录器类"""
    
    def __init__(
        self,
        name: str = "ai-trader",
        level: str = "INFO",
        log_dir: Optional[str] = None,
        console: bool = True,
        file: bool = True,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5
    ):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            console: 是否输出到控制台
            file: 是否输出到文件
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # 设置日志级别
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = self._create_formatter()
        
        # 添加控制台处理器
        if console:
            console_handler = self._create_console_handler(formatter)
            self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        if file:
            if log_dir is None:
                # 默认日志目录
                root_dir = Path(__file__).parent.parent.parent
                log_dir = root_dir / "logs"
            
            file_handler = self._create_file_handler(
                log_dir, formatter, max_bytes, backup_count
            )
            self.logger.addHandler(file_handler)
        
        # 防止日志传播到父记录器
        self.logger.propagate = False
    
    def _create_formatter(self) -> logging.Formatter:
        """
        创建日志格式化器
        
        Returns:
            日志格式化器
        """
        # 日志格式
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
        )
        
        # 日期格式
        date_format = "%Y-%m-%d %H:%M:%S"
        
        return logging.Formatter(log_format, date_format)
    
    def _create_console_handler(
        self, formatter: logging.Formatter
    ) -> logging.StreamHandler:
        """
        创建控制台处理器
        
        Args:
            formatter: 日志格式化器
            
        Returns:
            控制台处理器
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # 为不同级别设置不同颜色（如果终端支持）
        if sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter(formatter))
        
        return console_handler
    
    def _create_file_handler(
        self,
        log_dir: Path,
        formatter: logging.Formatter,
        max_bytes: int,
        backup_count: int
    ) -> RotatingFileHandler:
        """
        创建文件处理器（支持日志轮转）
        
        Args:
            log_dir: 日志目录
            formatter: 日志格式化器
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            
        Returns:
            文件处理器
        """
        # 确保日志目录存在
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"{self.name}_{date_str}.log"
        
        # 创建轮转文件处理器
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        return file_handler
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """记录DEBUG级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """记录INFO级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """记录WARNING级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """记录ERROR级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """记录CRITICAL级别日志"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """记录异常信息"""
        self.logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: str) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        self.logger.setLevel(getattr(logging, level.upper()))


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def __init__(self, base_formatter: logging.Formatter):
        """
        初始化彩色格式化器
        
        Args:
            base_formatter: 基础格式化器
        """
        super().__init__()
        self.base_formatter = base_formatter
    
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录
        
        Args:
            record: 日志记录
            
        Returns:
            格式化后的日志字符串
        """
        # 获取基础格式化结果
        log_message = self.base_formatter.format(record)
        
        # 添加颜色
        level_name = record.levelname
        if level_name in self.COLORS:
            color = self.COLORS[level_name]
            reset = self.COLORS['RESET']
            # 只给级别名称添加颜色
            log_message = log_message.replace(
                level_name,
                f"{color}{level_name}{reset}"
            )
        
        return log_message


# 全局日志记录器实例
_global_logger = None


def get_logger(
    name: str = "ai-trader",
    level: str = "INFO",
    **kwargs
) -> Logger:
    """
    获取全局日志记录器实例
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        **kwargs: 其他参数
        
    Returns:
        Logger实例
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = Logger(name=name, level=level, **kwargs)
    
    return _global_logger


def reset_logger() -> None:
    """重置全局日志记录器实例"""
    global _global_logger
    _global_logger = None


if __name__ == "__main__":
    # 测试日志系统
    print("日志系统测试")
    print("-" * 50)
    
    # 创建日志记录器
    logger = Logger(name="test_logger", level="DEBUG")
    
    # 测试不同级别的日志
    logger.debug("这是一条DEBUG日志")
    logger.info("这是一条INFO日志")
    logger.warning("这是一条WARNING日志")
    logger.error("这是一条ERROR日志")
    logger.critical("这是一条CRITICAL日志")
    
    # 测试异常日志
    try:
        1 / 0
    except Exception as e:
        logger.exception("捕获到异常")
    
    print("-" * 50)
    print("✓ 日志系统测试完成")
    print(f"日志文件已保存到: logs/test_logger_{datetime.now().strftime('%Y%m%d')}.log")