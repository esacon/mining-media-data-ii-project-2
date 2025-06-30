import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

from src.data.data_loader import WMT21DataLoader
from src.data.dataset import CriticalErrorDataset
from src.models.distilbert_classifier import DistilBERTClassifier
from src.models.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logger


class Runner:
    def __init__(self, config_path: str, device: str = "auto"):
        self.config_path = config_path
        self.device_str = device
        self.config = load_config(config_path)
        setup_logger(
            name="Runner",
            level_str=self.config["logging"]["level"],
            log_file=f"{self.config['paths']['logs_dir']}/runner.log",
            log_to_console=True,
        )
        self.logger = get_logger(__name__)
        self.device = self._setup_device(device)
        self._set_random_seeds()
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _setup_device(self, device: str) -> torch.device:
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
        seed = self.config["data"]["random_seed"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self.config["model"]["name"]
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def _initialize_model(self, from_pretrained: bool = True):
        """Initialize the model with proper configuration."""
        model_name = self.config["model"]["name"]
        num_labels = self.config["model"]["num_labels"]

        if from_pretrained:
            # Load the configuration from the pre-trained model
            config = DistilBertConfig.from_pretrained(model_name)
            # Set our specific parameters
            config.num_labels = num_labels

            # Create our custom classifier with the configuration
            self.model = DistilBERTClassifier(config)

            # Load only the DistilBERT weights from the pre-trained model
            pretrained_distilbert = DistilBertModel.from_pretrained(model_name)

            # Copy the DistilBERT weights to our model
            self.model.distilbert.load_state_dict(pretrained_distilbert.state_dict())

            self.logger.info(
                f"✅ Loaded pre-trained DistilBERT weights from {model_name}"
            )
            self.logger.info(
                f"🔧 Initialized new classification head for {num_labels} labels"
            )
        else:
            # Create from scratch
            config = DistilBertConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            self.model = DistilBERTClassifier(config)

        self.model.to(self.device)

    def _load_model_from_checkpoint(self, model_path: str):
        """Load model from a saved checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if "config" in checkpoint:
            model_config = checkpoint["config"]
            num_labels = model_config["model"]["num_labels"]
            model_name = model_config["model"]["name"]
        else:
            model_name = self.config["model"]["name"]
            num_labels = self.config["model"]["num_labels"]

        # Create model with proper configuration
        config = DistilBertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        self.model = DistilBERTClassifier(config)

        # Load the saved state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        if "tokenizer" in checkpoint:
            self.tokenizer = checkpoint["tokenizer"]
        else:
            self._load_tokenizer()

        self.logger.info(f"✅ Loaded model from checkpoint: {model_path}")

    def _get_dataloader_params(self) -> Dict[str, Any]:
        return {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": self.config["data"]["num_workers"],
            "pin_memory": self.config["data"]["pin_memory"],
        }

    def train(
        self,
        data_path: str,
        language_pair: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        self._load_tokenizer()
        self._initialize_model(from_pretrained=True)
        train_loader, val_loader = self._create_data_loaders(data_path, language_pair)
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )
        save_dir = self.config["paths"]["checkpoints_dir"]
        results = self.trainer.train(train_loader, val_loader, save_dir, language_pair)
        if save_results:
            # Add metadata to results
            results["language_pair"] = language_pair
            results["timestamp"] = results.get("timestamp", "unknown")

            # Save with unique filename if language pair is specified
            if language_pair:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_results_{timestamp}_{language_pair}.json"
                output_path = Path(self.config["paths"]["results_dir"]) / filename
            else:
                output_path = (
                    Path(self.config["paths"]["results_dir"]) / "training_results.json"
                )
            self._save_results(results, output_path)
        return results

    def evaluate(
        self,
        model_path: str,
        data_path: str,
        language_pair: Optional[str] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        self._load_model_from_checkpoint(model_path)
        test_loader = self._create_test_data_loader(data_path, language_pair)
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )
        results = self.trainer.evaluate(test_loader)
        predictions, probabilities = self.trainer.predict(test_loader)
        results["predictions"] = predictions
        results["probabilities"] = probabilities
        if save_results:
            output_path = (
                Path(self.config["paths"]["results_dir"]) / "evaluation_results.json"
            )
            self._save_results(results, output_path)
        return results

    def predict(
        self,
        model_path: str,
        data_path: str,
        language_pair: Optional[str] = None,
        save_results: bool = True,
    ) -> Tuple[List[int], List[float]]:
        self._load_model_from_checkpoint(model_path)
        test_loader = self._create_test_data_loader(data_path, language_pair)
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )
        predictions, probabilities = self.trainer.predict(test_loader)
        if save_results:
            results = {
                "predictions": predictions,
                "probabilities": probabilities,
                "data_path": data_path,
                "model_path": model_path,
                "language_pair": language_pair,
            }
            output_path = (
                Path(self.config["paths"]["results_dir"]) / "prediction_results.json"
            )
            self._save_results(results, output_path)
        return predictions, probabilities

    def analyze_data(
        self,
        data_dir: Optional[str] = None,
        language_pairs: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        if data_dir is None:
            data_dir = self.config["data"]["train_data_dir"]
        if language_pairs is None:
            language_pairs = ["en-de", "en-ja", "en-zh", "en-cs"]
        loader = WMT21DataLoader()
        results = {}
        for language_pair in language_pairs:
            analysis = self._analyze_language_pair(loader, data_dir, language_pair)
            results[language_pair] = analysis
        if save_results:
            output_path = (
                Path(self.config["paths"]["results_dir"]) / "data_analysis_results.json"
            )
            self._save_results(results, output_path)
        return results

    def run_full_experiment(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> Dict[str, Any]:
        training_results = self.train(data_path, language_pair, save_results=False)
        checkpoints_dir = Path(self.config["paths"]["checkpoints_dir"])
        best_model_path = checkpoints_dir / "best_model.pt"
        if not best_model_path.exists():
            checkpoints = list(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                best_model_path = checkpoints[0]
            else:
                raise FileNotFoundError("No model checkpoints found after training")
        evaluation_results = self.evaluate(
            str(best_model_path), data_path, language_pair, save_results=False
        )
        experiment_results = {
            "training": training_results,
            "evaluation": evaluation_results,
            "model_path": str(best_model_path),
            "data_path": data_path,
            "language_pair": language_pair,
        }
        output_path = (
            Path(self.config["paths"]["results_dir"]) / "experiment_results.json"
        )
        self._save_results(experiment_results, output_path)
        return experiment_results

    def _save_results(self, results: Dict[str, Any], output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    def _create_data_loaders(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader]:
        self._load_tokenizer()
        full_dataset = CriticalErrorDataset(
            data_path, self.tokenizer, language_pair=language_pair
        )
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        generator = torch.Generator().manual_seed(self.config["data"]["random_seed"])
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        dl_params = self._get_dataloader_params()
        train_loader = DataLoader(train_dataset, shuffle=True, **dl_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_params)
        return train_loader, val_loader

    def _create_test_data_loader(
        self, data_path: str, language_pair: Optional[str] = None
    ) -> DataLoader:
        self._load_tokenizer()
        test_dataset = CriticalErrorDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_length"],
            language_pair=language_pair,
        )
        dl_params = self._get_dataloader_params()
        test_loader = DataLoader(test_dataset, shuffle=False, **dl_params)
        return test_loader

    def _analyze_language_pair(
        self, loader: WMT21DataLoader, data_dir: str, language_pair: str
    ) -> Dict[str, Any]:
        prefix_map = {
            "en-de": "ende",
            "en-ja": "enja",
            "en-zh": "enzh",
            "en-cs": "encs",
        }
        prefix = prefix_map.get(language_pair, "ende")
        data_path = Path(data_dir)
        train_file = data_path / f"{prefix}_majority_train.tsv"
        dev_file = data_path / f"{prefix}_majority_dev.tsv"
        test_file = data_path / f"{prefix}_majority_test_blind.tsv"
        analysis = {"language_pair": language_pair, "files_found": {}}
        if train_file.exists():
            train_df = loader.load_train_dev_data(str(train_file))
            analysis["train"] = {
                "samples": len(train_df),
                "critical_errors": int(train_df["binary_label"].sum()),
                "error_rate": float(train_df["binary_label"].mean()),
                "no_errors": int((train_df["binary_label"] == 0).sum()),
            }
            analysis["files_found"]["train"] = str(train_file)
        if dev_file.exists():
            dev_df = loader.load_train_dev_data(str(dev_file))
            analysis["dev"] = {
                "samples": len(dev_df),
                "critical_errors": int(dev_df["binary_label"].sum()),
                "error_rate": float(dev_df["binary_label"].mean()),
                "no_errors": int((dev_df["binary_label"] == 0).sum()),
            }
            analysis["files_found"]["dev"] = str(dev_file)
        if test_file.exists():
            test_df = loader.load_test_data(str(test_file))
            analysis["test"] = {
                "samples": len(test_df),
                "note": "Labels not available for test data",
            }
            analysis["files_found"]["test"] = str(test_file)
        if "train" in analysis and "dev" in analysis:
            total_samples = analysis["train"]["samples"] + analysis["dev"]["samples"]
            total_errors = (
                analysis["train"]["critical_errors"]
                + analysis["dev"]["critical_errors"]
            )
            analysis["overall"] = {
                "total_samples": total_samples,
                "total_critical_errors": total_errors,
                "overall_error_rate": (
                    total_errors / total_samples if total_samples > 0 else 0.0
                ),
            }
        return analysis
