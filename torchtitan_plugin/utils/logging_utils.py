import logging
import os

from .utils import is_main_process

logger = logging.getLogger()


def init_logger():
    logger.setLevel(logging.INFO)
    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


# log only onece: https://stackoverflow.com/questions/31953272/logging-print-message-only-once
def logging_info(string, warning=False):
    if is_main_process():
        if warning:
            logger.warning(string)
        else:
            logger.info(string)
