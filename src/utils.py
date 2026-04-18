"""
utils.py
--------
Shared logging utility for the SP500 pipeline.
Creates a named logger that writes to both stdout and reports/run.log.
All modules import `logger` from here to ensure a consistent log format.
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str = "QuantFactor") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        from . import config
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.REPORTS_DIR / "run.log", mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

logger = setup_logger()
