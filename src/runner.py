import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer

from src.data import CriticalErrorDataset, CustomDataLoader
from src.models import DistilBERTClassifier, EvaluationFormatter, Trainer
from src.types import (
    EvaluationResults,
    ExperimentResults,
    LanguagePairAnalysis,
    PredictionResults,
    TrainingResults,
)
from src.utils import (
    DatasetFileLocator,
    LanguagePairMapper,
    get_logger,
    load_config,
    setup_logger,
)


class Runner:
    """Manages the end-to-end workflow for training, evaluation, prediction, and
    data analysis."""

    def __init__(
        self, config_path: str, device: str = "auto", quick_test: bool = False
    ):
        """
        Initializes the Runner with configuration and sets up the device and logger.

        Args:
            config_path: Path to the configuration file.
            device: Device to use for computations ('auto', 'cuda', 'mps', 'cpu').
            quick_test: If True, applies quick test settings for faster training.
        """
        self.config_path: str = config_path
        self.device_str: str = device
        self.quick_test: bool = quick_test
        self.config: Dict[str, Dict[str, str]] = load_config(config_path)

        if quick_test and "quick_test" in self.config:
            self._apply_quick_test_settings()

        setup_logger(
            name="Runner",
            level_str=self.config["logging"]["level"],
            log_file=f"{self.config['paths']['logs_dir']}/runner.log",
            log_to_console=True,
        )
        self.logger = get_logger(__name__)
        self.device: torch.device = self._setup_device(device)
        self._set_random_seeds()
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[DistilBERTClassifier] = None
        self.trainer: Optional[Trainer] = None

    def _setup_device(self, device: str) -> torch.device:
        """
        Determines and returns the appropriate torch.device.

        Args:
            device: Preferred device string ('auto', 'cuda', 'mps', 'cpu').

        Returns:
            The torch.device instance.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _set_random_seeds(self):
        """Sets random seeds for reproducibility across torch and CUDA."""
        seed: int = self.config["data"]["random_seed"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _apply_quick_test_settings(self):
        """Applies quick test configuration settings for faster training/testing."""
        quick_config = self.config["quick_test"]

        self.config["training"]["batch_size"] = quick_config.get(
            "batch_size", self.config["training"]["batch_size"]
        )
        self.config["training"]["num_epochs"] = quick_config.get(
            "num_epochs", self.config["training"]["num_epochs"]
        )
        self.config["training"]["learning_rate"] = quick_config.get(
            "learning_rate", self.config["training"]["learning_rate"]
        )

        self.default_sample_size = quick_config.get("sample_size", 100)

    def _load_tokenizer(self):
        """Loads the DistilBertTokenizer if not already loaded."""
        if self.tokenizer is None:
            model_name: str = self.config["model"]["name"]
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def _initialize_model(self, from_pretrained: bool = True):
        """
        Initializes the DistilBERTClassifier model.

        Args:
            from_pretrained: If True, loads pre-trained DistilBERT weights and
                             initializes a new classification head. If False,
                             initializes from scratch.
        """
        model_name: str = self.config["model"]["name"]
        num_labels: int = self.config["model"]["num_labels"]

        config = DistilBertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        self.model = DistilBERTClassifier(config)

        if from_pretrained:
            pretrained_distilbert = DistilBertModel.from_pretrained(model_name)
            self.model.distilbert.load_state_dict(pretrained_distilbert.state_dict())

        self.model.to(self.device)

    def _load_model_from_checkpoint(self, model_path: str):
        """
        Loads the model and tokenizer from a saved checkpoint.

        Args:
            model_path: Path to the model checkpoint file.
        """
        checkpoint: Dict[str, torch.Tensor] = torch.load(
            model_path, map_location=self.device
        )

        model_config: Dict[str, int] = checkpoint.get("config", self.config)
        num_labels: int = model_config["model"]["num_labels"]
        model_name: str = model_config["model"]["name"]

        config = DistilBertConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        self.model = DistilBERTClassifier(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.tokenizer = checkpoint.get("tokenizer")
        if self.tokenizer is None:
            self._load_tokenizer()

    def _get_dataloader_params(self) -> Dict[str, int | bool]:
        """
        Retrieves DataLoader parameters from the configuration.

        Returns:
            A dictionary containing DataLoader parameters.
        """
        return {
            "batch_size": self.config["training"]["batch_size"],
            "num_workers": self.config["data"]["num_workers"],
            "pin_memory": self.config["data"]["pin_memory"],
        }

    def _create_data_loaders(
        self, data_path: str, **dataset_kwargs
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Creates training and validation DataLoaders.

        Args:
            data_path: Path to the dataset.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair, sample_size).

        Returns:
            A tuple containing the training DataLoader and validation DataLoader.
        """
        self._load_tokenizer()
        full_dataset = CriticalErrorDataset(
            data_path,
            self.tokenizer,
            max_length=self.config["model"]["max_seq_length"],
            **dataset_kwargs,
        )

        train_ratio: float = self.config["data"]["train_ratio"]
        val_ratio: float = self.config["data"]["val_ratio"]
        total_size: int = len(full_dataset)
        train_size: int = int(train_ratio * total_size)
        val_size: int = int(val_ratio * total_size)
        test_size: int = total_size - train_size - val_size

        generator = torch.Generator().manual_seed(self.config["data"]["random_seed"])
        train_dataset, val_dataset, _ = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )

        dl_params: Dict[str, int | bool] = self._get_dataloader_params()
        train_loader = DataLoader(train_dataset, shuffle=True, **dl_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **dl_params)
        return train_loader, val_loader

    def _create_test_data_loader(self, data_path: str, **dataset_kwargs) -> DataLoader:
        """
        Creates a test DataLoader.

        Args:
            data_path: Path to the dataset.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair, sample_size).

        Returns:
            The test DataLoader.
        """
        self._load_tokenizer()
        test_dataset = CriticalErrorDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config["model"]["max_seq_length"],
            **dataset_kwargs,
        )
        dl_params: Dict[str, int | bool] = self._get_dataloader_params()
        test_loader = DataLoader(test_dataset, shuffle=False, **dl_params)
        return test_loader

    def _save_results(
        self, results: Dict[str, Union[str, int, float, List, Dict]], output_path: Path
    ):
        """
        Saves results to a JSON file.

        Args:
            results: Dictionary of results to save.
            output_path: Path where the results JSON file will be saved.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    def train(
        self,
        data_path: str,
        save_results: bool = True,
        **dataset_kwargs,
    ) -> TrainingResults:
        """
        Initiates the training process for the model.

        Args:
            data_path: Path to the dataset for training and validation.
            save_results: Whether to save the training results to a file.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair, sample_size).

        Returns:
            A dictionary containing the training results.
        """
        if (
            self.quick_test
            and hasattr(self, "default_sample_size")
            and "sample_size" not in dataset_kwargs
        ):
            dataset_kwargs["sample_size"] = self.default_sample_size

        self._load_tokenizer()
        self._initialize_model(from_pretrained=True)
        train_loader, val_loader = self._create_data_loaders(
            data_path, **dataset_kwargs
        )

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )

        save_dir: str = self.config["paths"]["checkpoints_dir"]
        results: TrainingResults = self.trainer.train(
            train_loader, val_loader, save_dir, dataset_kwargs.get("language_pair")
        )

        if save_results:
            from datetime import datetime

            results.update(dataset_kwargs)
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            language_pair = dataset_kwargs.get("language_pair")
            filename: str = (
                f"training_results_{timestamp}_{language_pair}.json"
                if language_pair
                else f"training_results_{timestamp}.json"
            )
            output_path: Path = Path(self.config["paths"]["results_dir"]) / filename
            self._save_results(results, output_path)

        return results

    def evaluate(
        self,
        model_path: str,
        data_path: str,
        save_results: bool = True,
        **dataset_kwargs,
    ) -> EvaluationResults:
        """
        Evaluates the model on a given dataset.

        Args:
            model_path: Path to the pre-trained model checkpoint.
            data_path: Path to the dataset for evaluation.
            save_results: Whether to save the evaluation results to a file.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair, sample_size).

        Returns:
            A dictionary containing the evaluation results, including predictions
            and probabilities.
        """
        if (
            self.quick_test
            and hasattr(self, "default_sample_size")
            and "sample_size" not in dataset_kwargs
        ):
            dataset_kwargs["sample_size"] = self.default_sample_size

        self._load_model_from_checkpoint(model_path)
        test_loader: DataLoader = self._create_test_data_loader(
            data_path, **dataset_kwargs
        )

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )

        predictions: List[int]
        probabilities: List[float]
        predictions, probabilities, _ = self.trainer.predict(test_loader)

        true_labels = []
        for batch in test_loader:
            true_labels.extend(batch["labels"].cpu().numpy())

        from sklearn.metrics import (
            accuracy_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        accuracy = accuracy_score(true_labels, predictions)
        mcc = matthews_corrcoef(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="binary", zero_division=0
        )
        auc_roc = roc_auc_score(true_labels, probabilities)

        results: EvaluationResults = {
            "accuracy": accuracy,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "predictions": predictions,
            "probabilities": probabilities,
        }
        results.update(dataset_kwargs)

        if save_results:
            output_path: Path = (
                Path(self.config["paths"]["results_dir"]) / "evaluation_results.json"
            )
            self._save_results(results, output_path)

        return results

    def predict(
        self,
        model_path: str,
        data_path: str,
        save_results: bool = True,
        create_evaluation_format: bool = True,
        method_name: str = "distilbert",
        **dataset_kwargs,
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Performs predictions using a trained model on new data.

        Args:
            model_path: Path to the pre-trained model checkpoint.
            data_path: Path to the dataset for prediction.
            save_results: Whether to save the prediction results to a file.
            create_evaluation_format: Whether to create WMT evaluation format files.
            method_name: Name of the method for evaluation format files.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair, sample_size).

        Returns:
            A tuple containing lists of predictions, probabilities, and test IDs.
        """
        if (
            self.quick_test
            and hasattr(self, "default_sample_size")
            and "sample_size" not in dataset_kwargs
        ):
            dataset_kwargs["sample_size"] = self.default_sample_size

        self._load_model_from_checkpoint(model_path)
        test_loader: DataLoader = self._create_test_data_loader(
            data_path, **dataset_kwargs
        )

        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config,
            device=self.device,
            logger=self.logger,
        )

        predictions: List[int]
        probabilities: List[float]
        test_ids: List[str]
        predictions, probabilities, test_ids = self.trainer.predict(test_loader)

        if save_results:
            results: PredictionResults = {
                "predictions": predictions,
                "probabilities": probabilities,
                "test_ids": test_ids,
                "data_path": data_path,
                "model_path": model_path,
            }
            results.update(dataset_kwargs)
            output_path: Path = (
                Path(self.config["paths"]["results_dir"]) / "prediction_results.json"
            )
            self._save_results(results, output_path)

        # Create evaluation format files if requested
        if create_evaluation_format and dataset_kwargs.get("language_pair"):
            self._create_evaluation_format_files(
                predictions=predictions,
                probabilities=probabilities,
                test_ids=test_ids,
                language_pair=dataset_kwargs["language_pair"],
                method_name=method_name,
            )

        return predictions, probabilities, test_ids

    def _create_evaluation_format_files(
        self,
        predictions: List[int],
        probabilities: List[float],
        test_ids: List[str],
        language_pair: str,
        method_name: str,
    ):
        """
        Create evaluation format files compatible with WMT evaluation script.

        Args:
            predictions: Binary predictions (0 or 1)
            probabilities: Prediction probabilities for class 1
            test_ids: Test sample IDs
            language_pair: Language pair code (e.g., 'en-de')
            method_name: Name of the method/system
        """
        model_size = self.model.get_model_size()

        formatter = EvaluationFormatter(logger=self.logger)

        results_dir = Path(self.config["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)

        eval_file_path = results_dir / f"evaluation_{language_pair}_{method_name}.tsv"
        formatter.create_evaluation_file(
            predictions=predictions,
            probabilities=probabilities,
            test_ids=test_ids,
            language_pair=language_pair,
            method_name=method_name,
            disk_footprint_bytes=model_size["disk_footprint_bytes"],
            model_parameters=model_size["total_parameters"],
            output_path=str(eval_file_path),
        )

        metadata_path = (
            results_dir / f"evaluation_metadata_{language_pair}_{method_name}.json"
        )
        formatter.create_metadata_file(
            predictions=predictions,
            language_pair=language_pair,
            method_name=method_name,
            model_info=model_size,
            output_path=str(metadata_path),
        )

        self.logger.info(f"Created evaluation files for {language_pair}")
        self.logger.info(f"Evaluation file: {eval_file_path}")
        self.logger.info(f"Metadata file: {metadata_path}")

    def analyze_data(
        self,
        data_dir: Optional[str] = None,
        groups: Optional[List[str]] = None,
        save_results: bool = True,
    ) -> Dict[str, LanguagePairAnalysis]:
        """
        Analyzes the dataset for specified groups.

        Args:
            data_dir: Directory containing the data files. Defaults to the
                      'train_data_dir' from the configuration.
            groups: List of groups to analyze. If None, uses default groups.
            save_results: Whether to save the analysis results to a file.

        Returns:
            A dictionary containing the data analysis results for each group.
        """
        data_to_analyze_dir: str = (
            data_dir if data_dir is not None else self.config["data"]["train_data_dir"]
        )

        if groups is None:
            mapper = LanguagePairMapper()
            file_locator = DatasetFileLocator(mapper)
            groups_to_analyze = file_locator.find_available_language_pairs(
                data_to_analyze_dir
            )
        else:
            groups_to_analyze = groups

        loader = CustomDataLoader()
        results: Dict[str, LanguagePairAnalysis] = {}

        for group in groups_to_analyze:
            analysis = self._analyze_group(loader, data_to_analyze_dir, group)
            results[group] = analysis

        if save_results:
            output_path: Path = (
                Path(self.config["paths"]["results_dir"]) / "data_analysis_results.json"
            )
            self._save_results(results, output_path)

        return results

    def _analyze_group(
        self, loader: CustomDataLoader, data_dir: str, group: str
    ) -> LanguagePairAnalysis:
        """
        Analyzes data for a specific group.

        Args:
            loader: Data loader instance.
            data_dir: Directory containing the data files.
            group: The group to analyze (e.g., "en-de").

        Returns:
            Analysis results for the group.
        """
        mapper = LanguagePairMapper()
        file_locator = DatasetFileLocator(mapper)

        file_paths = file_locator.get_file_paths(data_dir, group)

        analysis: LanguagePairAnalysis = {
            "language_pair": group,
            "files_found": {split: str(path) for split, path in file_paths.items()},
        }

        if "train" in file_paths:
            train_df = loader.load_train_dev_data(str(file_paths["train"]))
            analysis["train"] = {
                "samples": len(train_df),
                "critical_errors": int(train_df["binary_label"].sum()),
                "error_rate": float(train_df["binary_label"].mean()),
                "no_errors": int((train_df["binary_label"] == 0).sum()),
            }

        if "dev" in file_paths:
            dev_df = loader.load_train_dev_data(str(file_paths["dev"]))
            analysis["dev"] = {
                "samples": len(dev_df),
                "critical_errors": int(dev_df["binary_label"].sum()),
                "error_rate": float(dev_df["binary_label"].mean()),
                "no_errors": int((dev_df["binary_label"] == 0).sum()),
            }

        if "test" in file_paths:
            test_df = loader.load_test_data(str(file_paths["test"]))
            analysis["test"] = {
                "samples": len(test_df),
                "note": "Labels not available for test data",
            }

        if "train" in analysis and "dev" in analysis:
            total_samples: int = (
                analysis["train"]["samples"] + analysis["dev"]["samples"]
            )
            total_errors: int = (
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

    def run_full_experiment(
        self, data_path: str, **dataset_kwargs
    ) -> ExperimentResults:
        """
        Runs a full experiment including training and evaluation, saving all results.

        Args:
            data_path: Path to the dataset for training and evaluation.
            **dataset_kwargs: Additional arguments for dataset creation
                               (e.g., language_pair).

        Returns:
            A dictionary containing combined training and evaluation results.
        """
        training_results: TrainingResults = self.train(
            data_path, save_results=False, **dataset_kwargs
        )

        best_model_path = training_results.get("best_model_path")

        if not best_model_path or not Path(best_model_path).exists():
            checkpoints_dir: Path = Path(self.config["paths"]["checkpoints_dir"])
            checkpoints: List[Path] = list(checkpoints_dir.glob("*.pt"))
            if checkpoints:
                best_model_path = str(checkpoints[0])
                self.logger.warning(f"Using fallback model: {best_model_path}")
            else:
                raise FileNotFoundError("No model checkpoints found after training")

        evaluation_results: EvaluationResults = self.evaluate(
            best_model_path, data_path, save_results=False, **dataset_kwargs
        )

        experiment_results: ExperimentResults = {
            "training": training_results,
            "evaluation": evaluation_results,
            "model_path": best_model_path,
            "data_path": data_path,
        }
        experiment_results.update(dataset_kwargs)

        output_path: Path = (
            Path(self.config["paths"]["results_dir"]) / "experiment_results.json"
        )
        self._save_results(experiment_results, output_path)

        return experiment_results
