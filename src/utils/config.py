"""
Configuration loading and validation utilities.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .logging_utils import get_logger


class ConfigLoader:
    """Handles configuration loading and validation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger(__name__)
        self._config = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with validation.

        Args:
            config_path: Path to config YAML file

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Validate and set defaults
        config = self._validate_and_set_defaults(config)

        self._config = config
        self.logger.info("Configuration loaded successfully")

        return config

    def _validate_and_set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and set default values."""

        # Ensure required sections exist
        required_sections = ["model", "training", "data", "paths", "logging"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # Set default values
        defaults = self._get_default_config()
        config = self._merge_with_defaults(config, defaults)

        # Validate specific values
        self._validate_config_values(config)

        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "model": {
                "name": "distilbert-base-multilingual-cased",
                "max_seq_length": 512,
                "num_labels": 2,
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 2e-5,
                "num_epochs": 3,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "device": "auto",
                "mixed_precision": True,
                "gradient_accumulation_steps": 1,
            },
            "data": {
                "train_data_dir": "data/catastrophic_errors",
                "gold_labels_dir": "data/catastrophic_errors_goldlabels",
                "train_ratio": 0.8,
                "val_ratio": 0.1,
                "random_seed": 42,
                "num_workers": 4,
                "pin_memory": True,
            },
            "paths": {
                "checkpoints_dir": "results/checkpoints",
                "logs_dir": "logs",
                "results_dir": "results",
            },
            "logging": {"level": "INFO", "log_interval": 100, "save_interval": 1000},
        }

    def _merge_with_defaults(
        self, config: Dict[str, Any], defaults: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge config with defaults, keeping existing values."""
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
            elif isinstance(default_value, dict) and isinstance(config[key], dict):
                config[key] = self._merge_with_defaults(config[key], default_value)
        return config

    def _validate_config_values(self, config: Dict[str, Any]):
        """Validate specific configuration values."""

        # Validate model config
        num_labels = int(config["model"]["num_labels"])
        if num_labels < 1:
            raise ValueError("num_labels must be positive")

        max_seq_length = int(config["model"]["max_seq_length"])
        if max_seq_length < 1 or max_seq_length > 4096:
            raise ValueError("max_seq_length must be between 1 and 4096")

        # Validate training config
        batch_size = int(config["training"]["batch_size"])
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        learning_rate = float(config["training"]["learning_rate"])
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        num_epochs = int(config["training"]["num_epochs"])
        if num_epochs < 1:
            raise ValueError("num_epochs must be positive")

        # Validate data split ratios
        train_ratio = float(config["data"]["train_ratio"])
        val_ratio = float(config["data"]["val_ratio"])

        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")

        if not (0 < val_ratio < 1):
            raise ValueError("val_ratio must be between 0 and 1")

        if train_ratio + val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")

        # Validate paths exist
        for path_key, path_value in config["paths"].items():
            if not path_key.endswith("_dir"):
                continue

            # Create directories if they don't exist
            Path(path_value).mkdir(parents=True, exist_ok=True)

    def get_config(self) -> Dict[str, Any]:
        """Get the loaded configuration."""
        if self._config is None:
            raise RuntimeError("No configuration loaded. Call load_config() first.")
        return self._config

    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Configuration saved to: {output_path}")


# Convenience function for simple config loading
def load_config(config_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    Simple function to load configuration.

    Args:
        config_path: Path to config file
        logger: Optional logger instance

    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(logger)
    return loader.load_config(config_path)
