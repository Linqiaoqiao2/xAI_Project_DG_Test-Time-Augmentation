# utils/logger.py

import logging

def init_logger(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Initialize and return a logger that logs to stdout and optionally to file.

    Args:
        log_file (str): Optional path to save log.
        level (str): Logging level ("INFO", "DEBUG", etc.)

    Returns:
        logger (logging.Logger)
    """
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
