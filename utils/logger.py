import logging
import sys
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

    # Remove all handlers from the root logger and the named logger
    logging.root.handlers = []

    # Create and configure the logger instance - use root logger to prevent propagation issues
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Ensure propagation is disabled to prevent duplicate logs
    logger.propagate = False
    # Clear any existing handlers on this logger instance
    logger.handlers = []

    # Create a file handler that writes to log_path in write mode (so it overwrites previous logs)
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.DEBUG)

    # Create a console handler for logging to the console - use sys.stdout explicitly
    ch = logging.StreamHandler(sys.stdout)
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

    # Add a custom method to force flushing after each log message
    original_info = logger.info
    original_debug = logger.debug
    original_warning = logger.warning
    original_error = logger.error
    original_critical = logger.critical

    def force_flush_info(msg, *args, **kwargs):
        result = original_info(msg, *args, **kwargs)
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
        return result

    def force_flush_debug(msg, *args, **kwargs):
        result = original_debug(msg, *args, **kwargs)
        for handler in logger.handlers:
            handler.flush()
        return result

    def force_flush_warning(msg, *args, **kwargs):
        result = original_warning(msg, *args, **kwargs)
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
        return result

    def force_flush_error(msg, *args, **kwargs):
        result = original_error(msg, *args, **kwargs)
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
        return result

    def force_flush_critical(msg, *args, **kwargs):
        result = original_critical(msg, *args, **kwargs)
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
        return result

    # Replace the logger methods with our custom flushing versions
    logger.info = force_flush_info
    logger.debug = force_flush_debug
    logger.warning = force_flush_warning
    logger.error = force_flush_error
    logger.critical = force_flush_critical

    # Log a test message to verify setup
    logger.info(f"Logger '{name}' initialized successfully")

    return logger
