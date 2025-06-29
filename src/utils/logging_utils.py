import logging
import sys
from pathlib import Path
from typing import Optional, Union

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str,
    level_str: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """Sets up a logger with consistent formatting, optional file output,
    and controllable console output.

    Args:
        name (str): The name of the logger.
        level_str (str): The logging level string (e.g., "INFO", "DEBUG").
                         Defaults to "INFO".
        log_file (Optional[Union[str, Path]]): Optional path to a log file.
        format_string (Optional[str]): Optional custom format string for log messages.
        log_to_console (bool): If True, logs to console. Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    log_level_int = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(log_level_int)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    current_format_string = format_string if format_string is not None else DEFAULT_LOG_FORMAT
    formatter = logging.Formatter(current_format_string)

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_file_path = Path(log_file)
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(
                f"Warning: Could not set up file logging for {name} at {log_file_path}. Error: {e}",
                file=sys.stderr,
            )
            if not log_to_console:
                console_handler_fallback = logging.StreamHandler(sys.stdout)
                console_handler_fallback.setFormatter(formatter)
                logger.addHandler(console_handler_fallback)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieves an existing logger by name. If it has no handlers,
    it sets it up with default console logging.

    Args:
        name (str): The name of the logger to retrieve or create.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class LoggerMixin:
    """A mixin class that provides a convenient `logger` property.

    Classes inheriting from LoggerMixin will have a `self.logger` attribute
    that returns a `logging.Logger` instance named after the class.
    The logger is configured via `get_logger` on first access.
    """

    @property
    def logger(self) -> logging.Logger:
        """Gets the logger instance for the current class.

        The logger is created and cached on the first access using get_logger.
        """
        if not hasattr(self, "_logger_instance_cache"):
            self._logger_instance_cache = get_logger(self.__class__.__name__)
        return self._logger_instance_cache
