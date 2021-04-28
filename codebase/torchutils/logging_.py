import os
import sys
import typing
import logging
import glob
import zipfile


_logger = logging.getLogger(__name__)


class LogExceptionHook(object):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def __call__(self, exc_type, exc_value, traceback):
        self.logger.exception("Uncaught exception", exc_info=(
            exc_type, exc_value, traceback))


FORMAT = "%(asctime)s %(levelname)-8s: %(message)s"


def init_logger(rank: int, filenmae: str = None) -> logging.Logger:
    if rank != 0:
        logging.basicConfig(level=logging.CRITICAL)
    else:
        formatter = logging.Formatter(FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers = [console_handler]
        if filenmae is not None:
            file_handler = logging.FileHandler(filenmae)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"))

        logging.basicConfig(
            level=level,
            handlers=handlers
        )

        sys.excepthook = LogExceptionHook(_logger)


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
