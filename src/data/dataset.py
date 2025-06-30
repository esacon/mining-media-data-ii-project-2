from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

from src.data.data_loader import CustomDataLoader


class CriticalErrorDataset(Dataset):
    """
    PyTorch Dataset for Critical Error Detection.

    This dataset handles source sentences, target translations, and binary labels
    indicating whether the translation contains critical errors.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
        language_pair: Optional[str] = None,
        sample_size: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the data file (TSV or JSON)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
            language_pair: Language pair filter (e.g., 'en-de')
            sample_size: Limit dataset to this many samples (for quick testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_pair = language_pair

        self.data = self._load_data(data_path)

        if self.language_pair:
            self.data = self.data[self.data["language_pair"] == self.language_pair]

        if sample_size and len(self.data) > sample_size:
            self.data = self.data.sample(n=sample_size, random_state=42)

        self.data = self.data.reset_index(drop=True)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            data_path: Path to data file

        Returns:
            DataFrame with columns: source, target, label, language_pair
        """
        loader = CustomDataLoader()

        from pathlib import Path

        path_obj = Path(data_path)

        if path_obj.is_dir():
            return self._load_from_directory(path_obj, loader)

        elif data_path.endswith(".tsv"):
            if "test_blind" in data_path:
                df = loader.load_test_data(data_path)
                df["label"] = 0
                return df
            else:
                df = loader.load_train_dev_data(data_path)
                df = loader.prepare_for_training(df)
                return df
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _load_from_directory(self, data_dir, loader) -> pd.DataFrame:
        """
        Load training and dev data from directory.

        Args:
            data_dir: Directory containing TSV files
            loader: Data loader instance

        Returns:
            Combined DataFrame from train and dev files
        """
        prefix_map = {
            "en-de": "ende",
            "en-ja": "enja",
            "en-zh": "enzh",
            "en-cs": "encs",
        }

        all_data = []

        if self.language_pair and self.language_pair in prefix_map:
            prefixes = [prefix_map[self.language_pair]]
        else:
            prefixes = list(prefix_map.values())

        for prefix in prefixes:
            train_file = data_dir / f"{prefix}_majority_train.tsv"
            dev_file = data_dir / f"{prefix}_majority_dev.tsv"

            if train_file.exists():
                train_df = loader.load_train_dev_data(str(train_file))
                train_df = loader.prepare_for_training(train_df)
                all_data.append(train_df)

            if dev_file.exists():
                dev_df = loader.load_train_dev_data(str(dev_file))
                dev_df = loader.prepare_for_training(dev_df)
                all_data.append(dev_df)

        if not all_data:
            raise ValueError(f"No training data found in directory: {data_dir}")

        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

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

        source_text = str(row["source"])
        target_text = str(row["target"])

        try:
            label = int(float(row["label"]))
        except (ValueError, TypeError):
            label = 0

        combined_text = f"{source_text} {self.tokenizer.sep_token} {target_text}"

        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "id": str(row["id"]),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text,
            "language_pair": row["language_pair"],
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset.

        Returns:
            Dictionary with label counts
        """
        label_counts = self.data["label"].value_counts().to_dict()
        return {
            "no_critical_error": label_counts.get(0, 0),
            "critical_error": label_counts.get(1, 0),
            "total": len(self.data),
        }

    def get_language_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of language pairs in the dataset.

        Returns:
            Dictionary with language pair counts
        """
        return self.data["language_pair"].value_counts().to_dict()

    def get_test_ids(self) -> List[str]:
        """
        Get list of test IDs from the dataset.

        Returns:
            List of test ID strings
        """
        return [str(test_id) for test_id in self.data["id"].tolist()]
