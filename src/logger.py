import logging
from datetime import datetime
import sys
import types
from typing import Type

logger = logging.getLogger(__name__)

stream_handle = logging.StreamHandler()
file_handle = logging.FileHandler(filename=f"logs/{datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}_gpt_run.log")

stream_handle.setLevel(logging.INFO)
file_handle.setLevel(logging.INFO)

stream_handle_format = logging.Formatter('%(asctime)s - %(message)s')
file_handle_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handle.setFormatter(stream_handle_format)
file_handle.setFormatter(file_handle_format)

logger.addHandler(stream_handle)
logger.addHandler(file_handle)

logger.setLevel(logging.INFO)


def log_uncaught_exceptions(exctype: Type[BaseException], value: BaseException, tb: types.TracebackType | None) -> None:
	logger.critical('ERROR', exc_info=(exctype, value, tb))


sys.excepthook = log_uncaught_exceptions
