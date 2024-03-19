import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

formatter = '%(asctime)s - %(funcName)s - %(filename)s - %(levelname)s - %(message)s'

def get_logger(logger_name):
    name_app = os.path.basename(__file__)[0:-3]
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_directory = "logs"  # 设置日志目录
    log_file_path = os.path.join(log_directory, f"G3_Chat_Bot.log")
    os.makedirs(log_directory, exist_ok=True)
    logger = logging.getLogger(logger_name)
    file_handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7, delay=True, encoding="utf-8")

    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    ch_format = logging.Formatter(formatter)
    console_handler.setFormatter(ch_format)
    file_handler.setFormatter(ch_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger
