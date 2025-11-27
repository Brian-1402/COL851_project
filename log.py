import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name: str, base_filename: str = "app", log_dir: str = "logs") -> logging.Logger:
    """
    Configures a logger with three outputs:
    1. Console (INFO+)
    2. Timestamped file (DEBUG+) - e.g., logs/app_2023-10-27_10-30-00.log
    3. 'latest.log' file (DEBUG+) - Overwritten on every run to mirror current execution.

    Args:
        name: Name of the logger instance.
        base_filename: Prefix for the log file.
        log_dir: Directory to store log files.
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamped_log = os.path.join(log_dir, f"{base_filename}_{timestamp}.log")
    latest_log = os.path.join(log_dir, "latest.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Timestamped File Handler (Append mode)
    ts_handler = logging.FileHandler(timestamped_log, mode='a', encoding='utf-8')
    ts_handler.setLevel(logging.DEBUG)
    ts_handler.setFormatter(formatter)
    logger.addHandler(ts_handler)

    # 2. Latest File Handler (Write mode - overwrites previous latest)
    latest_handler = logging.FileHandler(latest_log, mode='w', encoding='utf-8')
    latest_handler.setLevel(logging.DEBUG)
    latest_handler.setFormatter(formatter)
    logger.addHandler(latest_handler)

    # 3. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    # Usage
    log = setup_logger("worker_process")
    
    log.info("Starting process...")
    log.debug(f"Process ID: {os.getpid()}")
    try:
        # Simulate work
        x = 1 / 0
    except Exception as e:
        log.exception("An error occurred during execution")