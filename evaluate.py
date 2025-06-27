#!/usr/bin/env python3
"""
Evaluation script for WMT21 Task 3 Critical Error Detection.

This script evaluates a trained DistilBERT model and generates predictions
in the WMT21 submission format.
"""

import argparse
import os
import torch
import yaml
import json
import pandas as pd
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.trainer import Trainer
from src.data.dataset import CriticalErrorDataset
from src.utils.logging_utils import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DistilBERT for WMT21 Task 3 Critical Error Detection"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to test data file"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--language_pair",
        type=str,
        default=None,
        help="Specific language pair to evaluate (e.g., 'en-de')"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for evaluation"
    )
    
    parser.add_argument(
        "--submission_format",
        action="store_true",
        help="Generate output in WMT21 submission format"
    )
    
    return parser.parse_args()


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_name = config['model']['model_name']
        num_labels = config['model']['num_labels']
    else:
        # Fallback to default values
        model_name = "distilbert-base-multilingual-cased"
        num_labels = 2
    
    # Load model
    model = DistilBERTClassifier.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load tokenizer
    if 'tokenizer' in checkpoint:
        tokenizer = checkpoint['tokenizer']
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    return model, tokenizer, checkpoint.get('config', {})


def create_confusion_matrix_plot(y_true, y_pred, output_path: str):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Critical Error', 'Critical Error'],
        yticklabels=['No Critical Error', 'Critical Error']
    )
    plt.title('Confusion Matrix - Critical Error Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_submission_file(
    predictions: list,
    probabilities: list,
    test_data: pd.DataFrame,
    output_path: str,
    model_size: dict
):
    """Generate submission file in WMT21 format."""
    submission_data = []
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        submission_data.append({
            'id': i,
            'prediction': pred,
            'probability': prob,
            'source': test_data.iloc[i]['source'] if 'source' in test_data.columns else '',
            'target': test_data.iloc[i]['target'] if 'target' in test_data.columns else '',
            'language_pair': test_data.iloc[i]['language_pair'] if 'language_pair' in test_data.columns else ''
        })
    
    # Save as TSV (WMT21 format)
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_path, sep='\t', index=False)
    
    # Also save metadata
    metadata = {
        'model_info': {
            'model_type': 'DistilBERT',
            'model_name': 'distilbert-base-multilingual-cased',
            'total_parameters': model_size.get('total_parameters', 0),
            'disk_footprint_bytes': model_size.get('disk_footprint_bytes', 0)
        },
        'submission_info': {
            'num_predictions': len(predictions),
            'positive_predictions': sum(predictions),
            'negative_predictions': len(predictions) - sum(predictions)
        }
    }
    
    metadata_path = output_path.replace('.tsv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(
        level=config['logging']['level'],
        log_to_file=config['logging']['file'],
        log_dir=config['paths']['logs_dir']
    )
    logger = get_logger(__name__)
    
    logger.info("Starting WMT21 Task 3 evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer, model_config = load_model(args.model_path, device)
    model_size = model.get_model_size()
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = CriticalErrorDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config['model']['max_seq_length'],
        language_pair=args.language_pair
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['dataloader_num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Initialize trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        logger=logger
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    
    # Check if test data has labels for evaluation
    has_labels = 'label' in test_dataset.data.columns
    
    if has_labels:
        metrics = trainer.evaluate(test_loader)
        logger.info("Evaluation Results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  MCC: {metrics['mcc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    # Make predictions
    logger.info("Generating predictions...")
    predictions, probabilities = trainer.predict(test_loader)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results
    results = {
        'predictions': predictions,
        'probabilities': probabilities,
        'model_size': model_size
    }
    
    if has_labels:
        results['metrics'] = metrics
        true_labels = test_dataset.data['label'].tolist()
        
        # Generate classification report
        report = classification_report(
            true_labels, predictions,
            target_names=['No Critical Error', 'Critical Error'],
            output_dict=True
        )
        results['classification_report'] = report
        
        # Create confusion matrix plot
        confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        create_confusion_matrix_plot(true_labels, predictions, confusion_matrix_path)
        logger.info(f"Confusion matrix saved: {confusion_matrix_path}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
    
    # Generate submission format if requested
    if args.submission_format:
        submission_path = os.path.join(args.output_dir, 'submission.tsv')
        generate_submission_file(
            predictions, probabilities, test_dataset.data, 
            submission_path, model_size
        )
        logger.info(f"Submission file saved: {submission_path}")
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"Critical errors detected: {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")


if __name__ == "__main__":
    main() 