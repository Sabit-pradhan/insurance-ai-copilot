# src/core/logger.py

import logging
import sys


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

def get_logger(name: str):
    """
    Create and return a reusable application logger.
    """

    logger = logging.getLogger(name)

    # Log INFO and above
    logger.setLevel(logging.INFO)

    # Prevent duplicate log lines
    if logger.handlers:
        return logger

    # Send logs to terminal
    handler = logging.StreamHandler(sys.stdout)

    # Standard log format
    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger