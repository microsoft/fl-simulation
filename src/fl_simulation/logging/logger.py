# Configuration for the logger.

import logging
from typing import Union

logger_name = 'fl-simulation'


def get_logger(log_level: Union[int, str] = logging.INFO) -> logging.Logger:
    """
    Args:
        log_level (Union[int, str]): The log level to use if the logger has not been set up yet.

    Returns:
        logging.Logger: A logger for this simulation system.
    """
    result = logging.getLogger(logger_name)

    if len(result.handlers) == 0:
        # The logger has not been set up yet.
        result.setLevel(log_level)
        f = logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s:%(filename)s:%(funcName)s\n%(message)s')
        h = logging.StreamHandler()
        h.setFormatter(f)
        result.addHandler(h)
    return result
