import logging
import sys

def setup_logging(level=logging.INFO):
    """Настройка логирования для всего проекта"""
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    logger.handlers.clear()
    logger.addHandler(console_handler)
    return logger

setup_logging()
