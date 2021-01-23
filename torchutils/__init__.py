import os
import sys
import logging
import argparse
import pathlib

import torch
from torch.utils.tensorboard import SummaryWriter

from .distributed import distributed_init, is_master
from .log import get_logger, LogExceptionHook, create_code_snapshot, get_diagnostic_info
from .common import DummyClass


__version__ = "v1.0.0-alpha0"

__all__ = [
    "auto_device",
    "logger",
    "summary_writer",
    "output_directory"
]

auto_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_directory", type=str, default=None)
    args, _ = parser.parse_known_args(argv)
    return args


# parse command line args
_args = get_args(sys.argv)
output_directory: pathlib.Path = None if _args.output_directory is None else pathlib.Path(_args.output_directory)

logger: logging.Logger = None
summary_writer: SummaryWriter = None


def torchutils_init(output_directory_: pathlib.Path = output_directory,
                    auto_resume: bool = False,
                    auto_create_code_snapshot: bool = True,
                    auto_save_diagnostic_info: bool = True):
    # automatically detect environment variables to initilize distributed mode
    distributed_init()

    if isinstance(output_directory_, str):
        output_directory_ = pathlib.Path(output_directory_)

    global logger
    global summary_writer
    if is_master():
        if output_directory_ is not None:
            os.makedirs(output_directory_, exist_ok=auto_resume)
        logger = get_logger("project", output_directory_, "log.txt")
        logger.debug(f"set output directory to {output_directory_}.")

        if output_directory_ is not None and auto_create_code_snapshot:
            create_code_snapshot("code", [".py"], ".", output_directory_)
            logger.debug(f"save code to {output_directory_/'code.zip'} .")

        if output_directory_ is not None and auto_save_diagnostic_info:
            with open(os.path.join(output_directory_, ".diagnostic_info"), "w") as f:
                f.write(get_diagnostic_info())
            logger.debug(f"save diagnostic info to {output_directory_/'.diagnostic_info'} .")

        sys.excepthook = LogExceptionHook(logger)
        logger.debug(f"mount 'LogExceptionHook' to log uncaught exception.")

        if output_directory_ is None:
            summary_writer = DummyClass()
        else:
            summary_writer = SummaryWriter(output_directory_)
            logger.debug(f"init tensorboard summarr writer.")
    else:
        logger = DummyClass()
        summary_writer = DummyClass()
