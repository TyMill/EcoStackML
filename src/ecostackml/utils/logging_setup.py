import logging
import os
from datetime import datetime

def setup_logger(name: str = "ecostackml", log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')

        # Log do konsoli
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # Log do pliku
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"ecostackml_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger
