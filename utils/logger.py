import logging
from datetime import datetime
import yaml
from pathlib import Path


def setup_logger(name: str = "DefaultLogger") -> logging.Logger:
    # Determine project root based on the location of this file (utils/logger.py)
    current_path = Path(__file__).resolve()
    project_path = current_path.parent.parent  # Points to the project root

    # Build the config file path and verify its existence
    config_path = project_path / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Build the absolute log directory path based on the config setting
    log_dir = (project_path / config['logging']["log_folder"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Define the log file path as {name}.log (without a timestamp)
    log_path = log_dir / f"{name}.log"

    # Remove the existing log file for this logger (if any), so that it starts fresh
    if log_path.exists():
        try:
            log_path.unlink()
        except Exception as e:
            print(f"Warning: Could not delete existing log file {log_path}: {e}")

    # Optional: Remove all handlers from all existing loggers to free file locks
    for logger_name in logging.root.manager.loggerDict:
        logger_instance = logging.getLogger(logger_name)
        handlers = logger_instance.handlers[:]
        for handler in handlers:
            try:
                handler.close()
            except Exception as e:
                print(f"Warning: Failed to close handler: {e}")
            logger_instance.removeHandler(handler)

    # Create and configure the logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Clear any existing handlers on this logger instance
    logger.handlers = []

    # Create a file handler that writes to log_path in write mode (so it overwrites previous logs)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a console handler for logging to the console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Set formatter to include timestamp, level, filename, line number, function name, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
