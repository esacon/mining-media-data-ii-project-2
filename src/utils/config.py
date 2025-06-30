"""
Configuration loading and validation utilities.
"""

import logging
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import yaml

from .logging_utils import get_logger


class ModelConfig(TypedDict):
    """Represents the 'model' section of the configuration."""

    name: str
    max_seq_length: int
    num_labels: int


class TrainingConfig(TypedDict):
    """Represents the 'training' section of the configuration."""

    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    warmup_steps: int
    device: Literal["cpu", "cuda", "auto"]
    mixed_precision: bool
    gradient_accumulation_steps: int


class DataConfig(TypedDict):
    """Represents the 'data' section of the configuration."""

    train_data_dir: str
    gold_labels_dir: str
    train_ratio: float
    val_ratio: float
    random_seed: int
    num_workers: int
    pin_memory: bool


class PathsConfig(TypedDict):
    """Represents the 'paths' section of the configuration."""

    checkpoints_dir: str
    logs_dir: str
    results_dir: str


class LoggingConfig(TypedDict):
    """Represents the 'logging' section of the configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_interval: int
    save_interval: int


class AppConfig(TypedDict):
    """Represents the complete application configuration."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathsConfig
    logging: LoggingConfig


class ConfigLoader:
    """Handles configuration loading and validation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self._config: Optional[AppConfig] = None

    def load_config(self, config_path: Union[str, Path]) -> AppConfig:
        """
        Load and parse configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.

        Returns:
            The parsed and validated configuration.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration is invalid.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        config = self._parse_and_validate_config(raw_config)
        self._config = config
        self.logger.info("Configuration loaded successfully.")

        return config

    def _parse_and_validate_config(self, raw_config: dict) -> AppConfig:
        """
        Parses and validates the raw configuration dictionary.

        Args:
            raw_config: The raw dictionary loaded from the YAML file.

        Returns:
            The validated and typed application configuration.

        Raises:
            ValueError: If any required section or value is missing or invalid.
        """
        self._validate_required_sections(raw_config)

        defaults = self._get_default_config()
        config: AppConfig = self._merge_config_with_defaults(raw_config, defaults)

        self._validate_model_config(config["model"])
        self._validate_training_config(config["training"])
        self._validate_data_config(config["data"])
        self._validate_paths_config(config["paths"])
        self._validate_logging_config(config["logging"])

        return config

    def _validate_required_sections(self, config: dict):
        """Ensures all top-level required sections are present."""
        required_sections = ["model", "training", "data", "paths", "logging"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

    def _get_default_config(self) -> AppConfig:
        """Provides default values for the application configuration."""
        return AppConfig(
            model=ModelConfig(
                name="distilbert-base-multilingual-cased",
                max_seq_length=512,
                num_labels=2,
            ),
            training=TrainingConfig(
                batch_size=16,
                learning_rate=2e-5,
                num_epochs=3,
                weight_decay=0.01,
                warmup_steps=500,
                device="auto",
                mixed_precision=True,
                gradient_accumulation_steps=1,
            ),
            data=DataConfig(
                train_data_dir="data/catastrophic_errors",
                gold_labels_dir="data/catastrophic_errors_goldlabels",
                train_ratio=0.8,
                val_ratio=0.1,
                random_seed=42,
                num_workers=4,
                pin_memory=True,
            ),
            paths=PathsConfig(
                checkpoints_dir="results/checkpoints",
                logs_dir="logs",
                results_dir="results",
            ),
            logging=LoggingConfig(
                level="INFO",
                log_interval=100,
                save_interval=1000,
            ),
        )

    def _merge_config_with_defaults(
        self, config: dict, defaults: AppConfig
    ) -> AppConfig:
        """Recursively merges the provided config with default values."""
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
            elif isinstance(default_value, dict) and isinstance(config[key], dict):
                config[key] = self._merge_config_with_defaults(
                    config[key], default_value
                )
        return AppConfig(**config)

    def _validate_model_config(self, model_config: ModelConfig):
        """Validates the 'model' section of the configuration."""
        model_config["num_labels"] = int(model_config["num_labels"])
        if model_config["num_labels"] < 1:
            raise ValueError("model.num_labels must be positive.")

        model_config["max_seq_length"] = int(model_config["max_seq_length"])
        if not (1 <= model_config["max_seq_length"] <= 4096):
            raise ValueError("model.max_seq_length must be between 1 and 4096.")

    def _validate_training_config(self, training_config: TrainingConfig):
        """Validates the 'training' section of the configuration."""
        training_config["batch_size"] = int(training_config["batch_size"])
        if training_config["batch_size"] < 1:
            raise ValueError("training.batch_size must be positive.")

        training_config["learning_rate"] = float(training_config["learning_rate"])
        if training_config["learning_rate"] <= 0:
            raise ValueError("training.learning_rate must be positive.")

        training_config["num_epochs"] = int(training_config["num_epochs"])
        if training_config["num_epochs"] < 1:
            raise ValueError("training.num_epochs must be positive.")

        training_config["weight_decay"] = float(training_config["weight_decay"])
        training_config["warmup_steps"] = int(training_config["warmup_steps"])

        valid_devices = ["cpu", "cuda", "auto"]
        if training_config["device"] not in valid_devices:
            raise ValueError(f"training.device must be one of {valid_devices}.")

        training_config["mixed_precision"] = bool(training_config["mixed_precision"])
        training_config["gradient_accumulation_steps"] = int(
            training_config["gradient_accumulation_steps"]
        )
        if training_config["gradient_accumulation_steps"] < 1:
            raise ValueError("training.gradient_accumulation_steps must be positive.")

    def _validate_data_config(self, data_config: DataConfig):
        """Validates the 'data' section of the configuration."""
        data_config["train_ratio"] = float(data_config["train_ratio"])
        data_config["val_ratio"] = float(data_config["val_ratio"])

        if not (0 < data_config["train_ratio"] < 1):
            raise ValueError("data.train_ratio must be between 0 and 1 (exclusive).")
        if not (0 < data_config["val_ratio"] < 1):
            raise ValueError("data.val_ratio must be between 0 and 1 (exclusive).")
        if data_config["train_ratio"] + data_config["val_ratio"] >= 1:
            raise ValueError("data.train_ratio + data.val_ratio must be less than 1.")

        data_config["random_seed"] = int(data_config["random_seed"])
        data_config["num_workers"] = int(data_config["num_workers"])
        data_config["pin_memory"] = bool(data_config["pin_memory"])

    def _validate_paths_config(self, paths_config: PathsConfig):
        """Validates the 'paths' section of the configuration and creates directories."""
        for path_key, path_value in paths_config.items():
            if not path_key.endswith("_dir"):
                continue
            Path(path_value).mkdir(parents=True, exist_ok=True)

    def _validate_logging_config(self, logging_config: LoggingConfig):
        """Validates the 'logging' section of the configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if logging_config["level"].upper() not in valid_levels:
            raise ValueError(f"logging.level must be one of {valid_levels}.")
        # Normalize to uppercase
        logging_config["level"] = logging_config["level"].upper()

        logging_config["log_interval"] = int(logging_config["log_interval"])
        if logging_config["log_interval"] < 1:
            raise ValueError("logging.log_interval must be positive.")

        logging_config["save_interval"] = int(logging_config["save_interval"])
        if logging_config["save_interval"] < 1:
            raise ValueError("logging.save_interval must be positive.")

    def get_config(self) -> AppConfig:
        """
        Retrieves the loaded configuration.

        Returns:
            The loaded application configuration.

        Raises:
            RuntimeError: If no configuration has been loaded yet.
        """
        if self._config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._config

    def save_config(self, config: AppConfig, output_path: Union[str, Path]):
        """
        Saves the configuration to a YAML file.

        Args:
            config: The configuration to save.
            output_path: The path to the output YAML file.
        """
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to: {output_path}")


def load_config(
    config_path: Union[str, Path], logger: Optional[logging.Logger] = None
) -> AppConfig:
    """
    A convenience function to load and validate configuration.

    Args:
        config_path: Path to the configuration YAML file.
        logger: Optional logger instance to use.

    Returns:
        The parsed and validated configuration.
    """
    loader = ConfigLoader(logger)
    return loader.load_config(config_path)
