import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup

from src.models import DistilBERTClassifier
from src.utils import get_logger


class Trainer:
    """
    Trainer for DistilBERT critical error detection model.
    """

    def __init__(
        self,
        model: DistilBERTClassifier,
        tokenizer: DistilBertTokenizer,
        config: Dict,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: DistilBERT classifier model
            tokenizer: DistilBERT tokenizer
            config: Configuration dictionary
            device: Training device
            logger: Logger instance
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.logger = logger or get_logger(__name__)

        # Training configuration
        self.num_epochs = config["training"]["num_epochs"]
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.weight_decay = config["training"]["weight_decay"]
        self.warmup_steps = config["training"]["warmup_steps"]

        # Optimizer and scheduler (initialized in train method)
        self.optimizer = None
        self.scheduler = None

        # Training history
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_mcc": [],
            "val_accuracy": [],
            "learning_rates": [],
        }

    def _setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        save_dir: str,
        language_pair: Optional[str] = None,
    ) -> Dict[str, dict]:
        """
        Train the model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            save_dir: Directory to save model checkpoints

        Returns:
            Dictionary with final metrics
        """
        # Setup optimizer and scheduler
        num_training_steps = len(train_dataloader) * self.num_epochs
        self._setup_optimizer_and_scheduler(num_training_steps)

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Generate timestamp and model prefix for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lang_suffix = f"_{language_pair}" if language_pair else ""
        model_prefix = f"{timestamp}{lang_suffix}"

        best_mcc = -1.0
        best_model_path = None

        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.logger.info(f"Total training steps: {num_training_steps}")
        self.logger.info(f"Model prefix: {model_prefix}")

        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            train_loss = self._train_epoch(train_dataloader)

            # Validation phase
            val_metrics = self.evaluate(val_dataloader)
            val_loss = val_metrics["loss"]
            val_mcc = val_metrics["mcc"]
            val_accuracy = val_metrics["accuracy"]

            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_mcc"].append(val_mcc)
            self.training_history["val_accuracy"].append(val_accuracy)
            self.training_history["learning_rates"].append(
                self.scheduler.get_last_lr()[0]
            )

            # Log metrics
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val MCC: {val_mcc:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            # Save best model
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_model_path = os.path.join(
                    save_dir, f"best_model_{model_prefix}_epoch_{epoch + 1}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "val_mcc": val_mcc,
                        "val_metrics": val_metrics,
                        "config": self.config,
                    },
                    best_model_path,
                )
                self.logger.info(f"New best model saved: {best_model_path}")

        # Save final model
        final_model_path = os.path.join(save_dir, f"final_model_{model_prefix}.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "tokenizer": self.tokenizer,
                "config": self.config,
                "training_history": self.training_history,
            },
            final_model_path,
        )

        # Save training history
        history_path = os.path.join(save_dir, f"training_history_{model_prefix}.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Training completed. Best MCC: {best_mcc:.4f}")

        return {
            "best_mcc": best_mcc,
            "best_model_path": best_model_path,
            "final_model_path": final_model_path,
            "timestamp": timestamp,
            "language_pair": language_pair,
            "model_prefix": model_prefix,
        }

    def _train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs["loss"]
                logits = outputs["logits"]

                total_loss += loss.item()

                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(
                    probabilities[:, 1].cpu().numpy()
                )  # Probability of class 1

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        mcc = matthews_corrcoef(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="binary"
        )
        auc_roc = roc_auc_score(all_labels, all_probabilities)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
        }

    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[List[int], List[float], List[str]]:
        """
        Make predictions on data.

        Args:
            dataloader: Data loader for prediction

        Returns:
            Tuple of (predictions, probabilities, test_ids)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_test_ids = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_test_ids.extend(batch["id"])

        return all_predictions, all_probabilities, all_test_ids
