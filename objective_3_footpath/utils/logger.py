"""Centralized logging for Objective 3 pipeline."""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "obj3",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = os.path.join(log_dir, f"system_{datetime.now():%Y%m%d}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
