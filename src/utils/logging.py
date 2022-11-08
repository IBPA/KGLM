import logging as log
import os
from pathlib import Path
from typing import List


def get_log_levels() -> List[str]:
    return ['DEBUG', 'INFO', 'WARNING', 'ERROR']


def set_logging(name: str = None, log_file: str = None, log_level: str = 'DEBUG') -> log.Logger:
    """
    Configure logging. By default, log to the console.
    If log_file is specified, log that file.

    Args:
        name: Name of the logger.
        log_file: Path to save the log file.
        log_level: Log level.
    """
    if log_level.upper() not in get_log_levels():
        raise ValueError('Invalid log level: %s', log_level)

    # create logger
    logger = log.getLogger(name)
    logger.setLevel(log_level.upper())

    # create formatter
    formatter = log.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')

    if log_file:
        Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
        # create and set file handler if requested
        file_handler = log.FileHandler(log_file)
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not log_file and not logger.handlers:
        # create and set console handler
        stream_handler = log.StreamHandler()
        stream_handler.setLevel(log_level.upper())
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
