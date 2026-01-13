import sys
from typing import Optional
from datetime import datetime
from loguru import logger as _logger

from utils.constants import ROOT_PATH

_print_level = "INFO"

def define_log_level(print_level="INFO", logfile_level="DEBUG", name: Optional[str] = None):
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(ROOT_PATH / f"logs/{log_name}.txt", level=logfile_level)
    return _logger

logger = define_log_level()
train_logger = define_log_level(name="train")
evaluation_logger = define_log_level(name="evaluation")
data_logger = define_log_level(name="data")