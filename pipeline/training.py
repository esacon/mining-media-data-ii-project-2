"""
Training pipeline for Critical Error Detection.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CriticalErrorDataset
from src.data.data_loader import WMT21DataLoader
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
            data_path: Path to training data (file or directory)
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

    def _prepare_training_data(self, data_path: str, language_pair: Optional[str] = None) -> str:
        """
        Prepare training data by combining files if directory is provided.
        
        Args:
            data_path: Path to training data (file or directory)
            language_pair: Optional language pair filter
            
        Returns:
            Path to the prepared training data file
        """
        if os.path.isfile(data_path):
            # Single file provided, use as-is
            self.logger.info(f"Using single training file: {data_path}")
            return data_path
        
        elif os.path.isdir(data_path):
            # Directory provided, find and combine training files
            self.logger.info(f"Processing training data directory: {data_path}")
            
            # Find all training files
            data_dir = Path(data_path)
            train_files = list(data_dir.glob("*_train.tsv"))
            
            if not train_files:
                raise ValueError(f"No training files (*_train.tsv) found in directory: {data_path}")
            
            # Filter by language pair if specified
            if language_pair:
                # Convert language pair format (e.g., 'en-de' -> 'ende')
                lang_code = language_pair.replace('-', '')
                train_files = [f for f in train_files if lang_code in f.name]
                
                if not train_files:
                    raise ValueError(f"No training files found for language pair: {language_pair}")
                
                self.logger.info(f"Found {len(train_files)} training file(s) for {language_pair}: {[f.name for f in train_files]}")
            else:
                self.logger.info(f"Found {len(train_files)} training files: {[f.name for f in train_files]}")
            
            # Combine all training files
            loader = WMT21DataLoader()
            combined_data = []
            
            for train_file in train_files:
                self.logger.info(f"Loading data from: {train_file}")
                df = loader.load_train_dev_data(str(train_file))
                combined_data.append(df)
            
            # Combine all dataframes
            if len(combined_data) == 1:
                combined_df = combined_data[0]
            else:
                combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Save combined data to temporary file
            output_dir = Path(self.config["paths"]["results_dir"])
            output_dir.mkdir(exist_ok=True)
            
            if language_pair:
                temp_file = output_dir / f"combined_train_{language_pair}.tsv"
            else:
                temp_file = output_dir / "combined_train_all.tsv"
            
            # For our own combined file, save with all necessary columns including language_pair
            # Format: ID  source  target  scores  binary_label  language_pair
            combined_df[['id', 'source', 'target', 'scores', 'binary_label', 'language_pair']].to_csv(
                temp_file, sep='\t', index=False, header=False
            )
            
            self.logger.info(f"Combined training data saved to: {temp_file}")
            self.logger.info(f"Total samples: {len(combined_df)}")
            self.logger.info(f"Label distribution: {combined_df['binary_label'].value_counts().to_dict()}")
            
            return str(temp_file)
        
        else:
            raise ValueError(f"Data path does not exist: {data_path}")

    def _create_data_loaders(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        self._load_tokenizer()

        # Prepare training data (handle directory inputs)
        prepared_data_path = self._prepare_training_data(data_path, language_pair)

        # Create dataset and filter by language pair
        full_dataset = CriticalErrorDataset(str(prepared_data_path), self.tokenizer, language_pair=language_pair)
        
        self.logger.info(f"Full dataset size: {len(full_dataset)}")
        if len(full_dataset) == 0:
            raise ValueError(f"No samples found for language pair: {language_pair}")
        
        # Split the dataset
        train_ratio = self.config["data"]["train_ratio"] 
        val_ratio = self.config["data"]["val_ratio"]
        
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Ensure we have at least 1 sample for each split
        if train_size < 1:
            train_size = 1
            val_size = max(1, total_size - train_size - 1) if total_size > 2 else 0
            test_size = max(0, total_size - train_size - val_size)
        elif val_size < 1 and total_size > 1:
            val_size = 1
            test_size = max(0, total_size - train_size - val_size)
        
        self.logger.info(f"Splitting into: train={train_size}, val={val_size}, test={test_size}")
        
        # Use random_split with proper generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        # Get dataloader parameters
        dl_params = self._get_dataloader_params()

        train_loader = DataLoader(train_dataset, shuffle=True, **dl_params)

        val_loader = DataLoader(val_dataset, shuffle=False, **dl_params)

        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")

        return train_loader, val_loader
