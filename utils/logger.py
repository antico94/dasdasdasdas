import logging
import os
from datetime import datetime

def setup_logger(name: str = "backtest_logger", log_dir: str = "logs") -> logging.Logger:
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Close and remove all handlers from all existing loggers to free file locks
    for logger_name in logging.root.manager.loggerDict:
        logger_instance = logging.getLogger(logger_name)
        handlers = logger_instance.handlers[:]
        for handler in handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Warning: Failed to close handler: {e}")
            logger_instance.removeHandler(handler)

    # Delete all existing log files in the log directory
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete log file {file_path}: {e}")

    # Create log file path with current timestamp
    log_path = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Get the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers (should be empty after the above loop)
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter that logs timestamp, level, filename, line number, function name, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
