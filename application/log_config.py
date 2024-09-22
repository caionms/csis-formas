import logging

from config import settings

default_logging_level = settings.app.loglevel

FORMAT = "[%(asctime)s %(filename)s:%(lineno)s] %(levelname)s: %(message)s"


def get_logger(
    name: str, level: str = default_logging_level, propagate: bool = False
) -> logging.Logger:
    log_level = level.upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
        logger.setLevel(log_level)
        handler.setLevel(log_level)
        logger.addHandler(handler)
        logger.propagate = propagate
    return logger
