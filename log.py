import logging
from logging.handlers import RotatingFileHandler
import os

def get_logger(save_log_path):

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  

    if not os.path.exists(os.path.dirname(save_log_path)):
        os.makedirs(os.path.dirname(save_log_path))
    file_handler = RotatingFileHandler(save_log_path, maxBytes=1024*1024*10, backupCount=5)
    file_handler.setLevel(logging.DEBUG)  

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
