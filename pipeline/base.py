"""
Base pipeline functionality shared across all pipeline components.
"""

from pathlib import Path
from typing import Any, Dict

import torch
from transformers import DistilBertConfig, DistilBertTokenizer

from src.models.distilbert_classifier import DistilBERTClassifier
from src.utils.config import load_config
from src.utils.logging_utils import get_logger


class BasePipeline:
    """Base class for all pipeline components."""

    def __init__(self, config_path: str, device: str = "auto"):
        """
        Initialize the base pipeline.

        Args:
            config_path: Path to configuration file
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        # Load configuration
        self.config = load_config(config_path)
        self.logger = get_logger(self.__class__.__name__)

        # Setup device
        self.device = self._setup_device(device)
        self.logger.info(f"Using device: {self.device}")

        # Set random seeds
        self._set_random_seeds()

        # Initialize components (will be set by subclasses)
        self.tokenizer = None
        self.model = None

    def _setup_device(self, device: str) -> torch.device:
        """Setup and return the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config["data"]["random_seed"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_tokenizer(self):
        """Load the tokenizer."""
        if self.tokenizer is None:
            model_name = self.config["model"]["name"]
            self.logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def _initialize_model(self, from_pretrained: bool = True):
        """Initialize the model."""
        model_name = self.config["model"]["name"]
        num_labels = self.config["model"]["num_labels"]

        self.logger.info(f"Initializing model: {model_name}")

        if from_pretrained:
            model_config = DistilBertConfig.from_pretrained(
                model_name, num_labels=num_labels)
            self.model = DistilBERTClassifier.from_pretrained(
                model_name, config=model_config)
        else:
            # Load from checkpoint (handled separately)
            pass

        self.model.to(self.device)

    def _load_model_from_checkpoint(self, model_path: str):
        """Load model from checkpoint."""
        self.logger.info(f"Loading model from: {model_path}")

        try:
            # Try loading with weights_only=False for compatibility with tokenizer objects
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint with weights_only=False: {e}")
            # Fallback to default loading
            checkpoint = torch.load(model_path, map_location=self.device)

        # Get model configuration
        if "config" in checkpoint:
            model_config = checkpoint["config"]
            model_name = model_config["model"]["name"]
            num_labels = model_config["model"]["num_labels"]
        else:
            # Fallback to current config
            model_name = self.config["model"]["name"]
            num_labels = self.config["model"]["num_labels"]

        # Initialize model
        self.model = DistilBERTClassifier.from_pretrained(
            model_name, num_labels=num_labels)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        # Load tokenizer
        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]
        else:
            self._load_tokenizer()

    def _get_dataloader_params(self) -> Dict[str, Any]:
        """Get common dataloader parameters."""
        return {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": self.config["data"]["num_workers"],
            "pin_memory": self.config["data"]["pin_memory"],
        }

    def _create_output_directory(self, output_path: str) -> Path:
        """Create output directory if it doesn't exist."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
