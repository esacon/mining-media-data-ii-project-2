"""
Training pipeline for Critical Error Detection.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CriticalErrorDataset
from src.models.trainer import Trainer

from .base import BasePipeline


class TrainingPipeline(BasePipeline):
    """Pipeline for training models."""

    def __init__(self, config_path: str, device: str = "auto"):
        super().__init__(config_path, device)
        self.trainer = None

    def run(self, data_path: str, language_pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the training pipeline.

        Args:
            data_path: Path to training data
            language_pair: Optional language pair filter

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting training pipeline")

        # Initialize model and tokenizer
        self._load_tokenizer()
        self._initialize_model(from_pretrained=True)

        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(data_path, language_pair)

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )

        # Train the model
        save_dir = self.config["paths"]["checkpoints_dir"]
        results = self.trainer.train(train_loader, val_loader, save_dir)

        self.logger.info("Training completed successfully")

        return results

    def _create_data_loaders(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        self._load_tokenizer()

        # Load dataset
        dataset = CriticalErrorDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_length"],
            language_pair=language_pair,
        )

        # Split dataset
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]

        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))
        test_size = len(dataset) - train_size - val_size

        train_dataset, temp_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, len(dataset) - train_size],
            generator=torch.Generator().manual_seed(self.config["data"]["random_seed"]),
        )

        val_dataset, _ = torch.utils.data.random_split(
            temp_dataset,
            [val_size, test_size],
            generator=torch.Generator().manual_seed(self.config["data"]["random_seed"]),
        )

        # Get dataloader parameters
        dl_params = self._get_dataloader_params()

        train_loader = DataLoader(train_dataset, shuffle=True, **dl_params)

        val_loader = DataLoader(val_dataset, shuffle=False, **dl_params)

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader
