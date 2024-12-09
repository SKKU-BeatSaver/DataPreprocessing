# logger_config.py
from datetime import datetime
import logging
today_string = datetime.now().strftime("%Y%m%d%H%M")

# 로그 설정
log_file = f"../log/training{today_string}.log"
# 공통 로거 설정
def setup_logger(log_file=log_file):
    logger = logging.getLogger("")
    if not logger.hasHandlers():  # 동일 설정이 여러 번 추가되지 않도록 방지
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger