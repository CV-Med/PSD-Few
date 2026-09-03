"""Simple file + stdout logger."""

import logging
import os
import sys


class Logger:
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        """Create a logger that writes to both a file and the console."""
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(stream_handler)

    def info(self, message: str):
        """Log an informational message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def close(self):
        """Close all attached handlers."""
        for handler in self.logger.handlers:
            handler.close()
