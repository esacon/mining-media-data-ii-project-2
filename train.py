#!/usr/bin/env python3
"""
Main training script for WMT21 Task 3 Critical Error Detection.

This script fine-tunes a DistilBERT model for binary classification of 
critical translation errors.
"""

import argparse
import os
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer, DistilBertConfig
from sklearn.model_selection import train_test_split

from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.trainer import Trainer
from src.data.dataset import CriticalErrorDataset
from src.utils.logging_utils import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DistilBERT for WMT21 Task 3 Critical Error Detection"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to training data file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/checkpoints",
        help="Directory to save model checkpoints"
    )
    
    parser.add_argument(
        "--language_pair",
        type=str,
        default=None,
        help="Specific language pair to train on (e.g., 'en-de')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Setup training device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    
    return device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(
    data_path: str,
    tokenizer: DistilBertTokenizer,
    config: dict,
    language_pair: str = None
) -> tuple:
    """
    Create training and validation data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset
    dataset = CriticalErrorDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config['model']['max_seq_length'],
        language_pair=language_pair
    )
    
    # Get data split ratios
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, temp_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )
    
    val_dataset, _ = torch.utils.data.random_split(
        temp_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(config['data']['random_seed'])
    )
    
    # Create data loaders
    batch_size = config['model']['training']['batch_size']
    num_workers = config['hardware']['dataloader_num_workers']
    pin_memory = config['hardware']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.debug else config['logging']['level']
    setup_logging(
        level=log_level,
        log_to_file=config['logging']['file'],
        log_dir=config['paths']['logs_dir']
    )
    logger = get_logger(__name__)
    
    logger.info("Starting WMT21 Task 3 Critical Error Detection training")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Language pair: {args.language_pair}")
    
    # Setup device
    device = setup_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['reproducibility']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['reproducibility']['seed'])
    
    # Load tokenizer
    model_name = config['model']['model_name']
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        args.data_path, tokenizer, config, args.language_pair
    )
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model_config = DistilBertConfig.from_pretrained(
        model_name,
        num_labels=config['model']['num_labels']
    )
    model = DistilBERTClassifier.from_pretrained(
        model_name,
        config=model_config
    )
    
    # Log model size
    model_size = model.get_model_size()
    logger.info(f"Model parameters: {model_size['total_parameters']:,}")
    logger.info(f"Trainable parameters: {model_size['trainable_parameters']:,}")
    logger.info(f"Estimated disk footprint: {model_size['disk_footprint_bytes']:,} bytes")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        save_dir=args.output_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Best MCC: {results['best_mcc']:.4f}")
    logger.info(f"Best model: {results['best_model_path']}")
    logger.info(f"Final model: {results['final_model_path']}")


if __name__ == "__main__":
    main() 