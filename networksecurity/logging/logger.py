import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# 定义日志文件名
LOG_FILE_NAME = f"networksecurity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# 定义日志文件夹的路径
LOGS_DIRECTORY = os.path.join(os.getcwd(), "logs")

# 确保日志文件夹存在
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# 定义最终的、完整的日志文件路径
LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)


def setup_logger(name: str = "networksecurity", log_level: str = "INFO") -> logging.Logger:
    """
    配置并返回一个结构化的日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 格式化器 - 更详细的格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 文件处理器 - 按大小轮转 (每个文件最大10MB，保留5个备份)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 2. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # 3. 错误日志单独文件
    error_log_path = os.path.join(LOGS_DIRECTORY, "errors.log")
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


# 创建默认日志记录器
logging = setup_logger()
