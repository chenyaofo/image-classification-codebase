import os
import sys
import time
import typing
import logging
import uuid
import glob
import zipfile

from torch.utils.collect_env import get_pretty_env_info

from .distributed import is_master
from .common import get_branch_name, get_last_commit_id

T = typing.TypeVar("T")


class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(
            exc_type, exc_value, traceback))


def get_logger(name: str, output_directory: str, log_name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )
    if is_master():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if output_directory is not None:
            file_handler = logging.FileHandler(
                os.path.join(output_directory, log_name))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    logger.propagate = False
    return logger


def create_code_snapshot(name: str,
                         include_suffix: typing.List[str],
                         source_directory: str,
                         store_directory: str) -> None:
    if store_directory is None:
        return
    with zipfile.ZipFile(os.path.join(store_directory, "{}.zip".format(name)), "w") as f:
        for suffix in include_suffix:
            for file in glob.glob(os.path.join(source_directory, "**", "*{}".format(suffix)), recursive=True):
                f.write(file, os.path.join(name, file))

def get_diagnostic_info():
    diagnostic_info = f"Log Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    diagnostic_info += f"UUID: {uuid.uuid1()}\n"
    diagnostic_info += f"Argv: {' '.join(sys.argv)}\n"
    diagnostic_info += f"Git Branch: {get_branch_name()}\n"
    diagnostic_info += f"Git Commit ID: {get_last_commit_id()}\n\n"

    diagnostic_info += f"More Diagnostic Info: \n"
    diagnostic_info += "-" * 50 + "\n"
    diagnostic_info += get_pretty_env_info() + "\n"
    diagnostic_info += "-" * 50 + "\n"
    

    return diagnostic_info
