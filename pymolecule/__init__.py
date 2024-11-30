from typing import Any

import os
import sys
from ast import literal_eval

from loguru import logger

from .molecule import Molecule

__all__ = ["Molecule"]
__version__ = "2.2.2"

logger.disable("pymolecule")

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


def enable_logging(
    level_set: int,
    stdout_set: bool = True,
    file_path: str | None = None,
    log_format: str = LOG_FORMAT,
) -> None:
    r"""Enable logging.

    Args:
        level: Requested log level: `10` is debug, `20` is info.
        file_path: Also write logs to files here.
    """
    config: dict[str, Any] = {"handlers": []}
    if stdout_set:
        config["handlers"].append(
            {
                "sink": sys.stdout,
                "level": level_set,
                "format": log_format,
                "colorize": True,
            }
        )
    if isinstance(file_path, str):
        config["handlers"].append(
            {"sink": file_path, "level": level_set, "format": log_format, "False": True}
        )
    # https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.configure
    logger.configure(**config)

    logger.enable("pymolecule")


if literal_eval(os.environ.get("pymolecule_LOG", "False")):
    level = int(os.environ.get("pymolecule_LOG_LEVEL", 20))
    stdout = literal_eval(os.environ.get("pymolecule_STDOUT", "True"))
    log_file_path = os.environ.get("pymolecule_LOG_FILE_PATH", None)
    enable_logging(level, stdout, log_file_path)