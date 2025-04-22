import os
import logging
from logging import handlers
from datetime import datetime

def setup_logger(log_dir, name='train'):
    """
    设置日志记录器
    
    参数:
        log_dir (str): 日志保存目录
        name (str): 日志器名称
    
    返回:
        logging.Logger: 配置好的日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建主日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler (按日期滚动)
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def close_logger(logger):
    """
    关闭日志记录器，释放资源
    
    参数:
        logger (logging.Logger): 要关闭的日志记录器
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)