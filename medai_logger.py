"""
medai_logger.py

Global logging setup for MEDAI.

Each module that calls get_logger("modulename") gets its own log file at:
logs/modulename.log
"""

import os
import logging

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger for the given module name.
    
    Args:
        name (str): Module name (e.g., "retrievalqa")

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
