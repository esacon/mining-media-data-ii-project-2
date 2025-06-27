"""
Dataset class for WMT21 Task 3 Critical Error Detection.

This module implements a PyTorch dataset for handling critical error detection data.
"""

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
from typing import List, Dict, Any, Optional
import pandas as pd
import json


class CriticalErrorDataset(Dataset):
    """
    PyTorch Dataset for WMT21 Task 3 Critical Error Detection.
    
    This dataset handles source sentences, target translations, and binary labels
    indicating whether the translation contains critical errors.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
        language_pair: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (TSV or JSON)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
            language_pair: Language pair filter (e.g., 'en-de')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_pair = language_pair
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Filter by language pair if specified
        if self.language_pair:
            self.data = self.data[self.data['language_pair'] == self.language_pair]
        
        # Reset index after filtering
        self.data = self.data.reset_index(drop=True)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            DataFrame with columns: source, target, label, language_pair
        """
        from .data_loader import WMT21DataLoader
        
        # Use the WMT21 data loader for proper format handling
        loader = WMT21DataLoader()
        
        if data_path.endswith('.tsv'):
            # Check if it's a test file (no labels) or train/dev file (with labels)
            if 'test_blind' in data_path:
                df = loader.load_test_data(data_path)
                # Add dummy labels for test data
                df['label'] = 0  # Will be ignored during inference
            else:
                df = loader.load_train_dev_data(data_path)
                df = loader.prepare_for_training(df)
        elif data_path.endswith('.json') or data_path.endswith('.jsonl'):
            # Load JSON format (fallback for custom data)
            if data_path.endswith('.jsonl'):
                # JSON Lines format
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                df = pd.DataFrame(data)
            else:
                # Regular JSON format
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_columns = ['source', 'target', 'label']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Add language_pair column if not present
            if 'language_pair' not in df.columns:
                df['language_pair'] = 'unknown'
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        return df

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        row = self.data.iloc[idx]
        
        # Get text inputs
        source_text = str(row['source'])
        target_text = str(row['target'])
        label = int(row['label'])
        
        # Combine source and target with [SEP] token
        # Format: [CLS] source [SEP] target [SEP]
        combined_text = f"{source_text} {self.tokenizer.sep_token} {target_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text,
            'language_pair': row['language_pair']
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset.
        
        Returns:
            Dictionary with label counts
        """
        label_counts = self.data['label'].value_counts().to_dict()
        return {
            'no_critical_error': label_counts.get(0, 0),
            'critical_error': label_counts.get(1, 0),
            'total': len(self.data)
        }

    def get_language_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of language pairs in the dataset.
        
        Returns:
            Dictionary with language pair counts
        """
        return self.data['language_pair'].value_counts().to_dict() 