import logging
from typing import Optional
from pathlib import Path

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)

    if log_file is None:
        log_file = f"{name}.log"

    log_path = log_dir / log_file

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False  # Prevent double logs

    return logger
