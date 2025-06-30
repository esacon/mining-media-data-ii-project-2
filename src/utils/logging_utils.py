import logging
import sys
from pathlib import Path
from typing import Literal, Optional, Union


class LoggingConfig:
    """Encapsulates configuration for a single logging handler."""

    def __init__(
        self,
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        log_to_console: bool = True,
    ):
        self.level = level
        self.log_file = Path(log_file) if log_file else None
        self.format_string = (
            format_string
            if format_string is not None
            else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_to_console = log_to_console


class LoggerFactory:
    """Manages the creation and configuration of logging.Logger instances."""

    def __init__(self):
        self._default_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def create_logger(
        self,
        name: str,
        config: Optional[LoggingConfig] = None,
    ) -> logging.Logger:
        """
        Creates and configures a logger instance based on the provided configuration.

        Args:
            name: The name of the logger.
            config: Optional LoggingConfig object specifying logger settings.

        Returns:
            The configured logging.Logger instance.
        """
        logger = logging.getLogger(name)
        self._clear_handlers(logger)

        effective_config = config if config else self._get_default_logging_config()
        self._apply_logger_settings(logger, effective_config)

        logger.propagate = False
        return logger

    def _clear_handlers(self, logger: logging.Logger):
        """Removes all existing handlers from a logger."""
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    def _get_default_logging_config(self) -> LoggingConfig:
        """Provides a default LoggingConfig instance."""
        return LoggingConfig(format_string=self._default_format)

    def _apply_logger_settings(self, logger: logging.Logger, config: LoggingConfig):
        """Applies the settings from a LoggingConfig to a logger."""
        log_level_int = getattr(logging, config.level.upper(), logging.INFO)
        logger.setLevel(log_level_int)

        formatter = logging.Formatter(config.format_string)

        if config.log_to_console:
            self._add_console_handler(logger, formatter)

        if config.log_file:
            self._add_file_handler(
                logger, formatter, config.log_file, config.log_to_console
            )

    def _add_console_handler(
        self, logger: logging.Logger, formatter: logging.Formatter
    ):
        """Adds a console handler to the logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def _add_file_handler(
        self,
        logger: logging.Logger,
        formatter: logging.Formatter,
        log_file_path: Path,
        fallback_to_console: bool,
    ):
        """Adds a file handler to the logger, with an optional console fallback."""
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_file_path, mode="a", encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            logger.warning(
                f"Could not set up file logging for {logger.name} at {log_file_path}."
            )
            if not fallback_to_console:
                self._add_console_handler(logger, formatter)


_logger_factory = LoggerFactory()


def setup_logger(
    name: str,
    level_str: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Sets up a logger with consistent formatting, optional file output,
    and controllable console output. This is a convenience function.

    Args:
        name: The name of the logger.
        level_str: The logging level string (e.g., "INFO", "DEBUG").
        log_file: Optional path to a log file.
        format_string: Optional custom format string for log messages.
        log_to_console: If True, logs to console.

    Returns:
        The configured logging.Logger instance.
    """
    config = LoggingConfig(
        level=level_str,
        log_file=log_file,
        format_string=format_string,
        log_to_console=log_to_console,
    )
    return _logger_factory.create_logger(name, config)


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves an existing logger by name. If it has no handlers,
    it sets it up with default console logging using the global factory.

    Args:
        name: The name of the logger to retrieve or create.

    Returns:
        The logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return _logger_factory.create_logger(name)
    return logger


class LoggerMixin:
    """
    A mixin class that provides a convenient `logger` property.

    Classes inheriting from LoggerMixin will have a `self.logger` attribute
    that returns a `logging.Logger` instance named after the class.
    The logger is configured via `get_logger` on first access.
    """

    @property
    def logger(self) -> logging.Logger:
        """
        Gets the logger instance for the current class.

        The logger is created and cached on the first access using get_logger.
        """
        if not hasattr(self, "_logger_instance_cache"):
            self._logger_instance_cache = get_logger(self.__class__.__name__)
        return self._logger_instance_cache
