"""LLM models for critical error detection."""

import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.prompts import PromptTemplate  # noqa: E402
from src.utils import get_logger  # noqa: E402


class LLMModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None

    def _setup_device(self, device: str) -> torch.device:
        """Setup device for model."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response from prompt."""
        pass

    def predict(
        self, source: str, target: str, prompt_template: PromptTemplate
    ) -> Tuple[str, str]:
        """
        Predict critical error label for source-target pair.

        Args:
            source: Source text
            target: Target translation
            prompt_template: Prompt template to use

        Returns:
            Tuple of (prediction, full_response)
        """
        prompt = prompt_template.format(source=source, target=target)
        full_response = self.generate_response(prompt)

        # Extract prediction from response
        prediction = self._extract_prediction(full_response)

        return prediction, full_response

    def _extract_prediction(self, response: str) -> str:
        """Extract ERR/NOT prediction from model response."""
        response = response.strip().upper()

        # Look for exact "ERR" or "NOT" responses
        if "ERR" in response and "NOT" not in response:
            return "ERR"
        elif "NOT" in response and "ERR" not in response:
            return "NOT"

        # Look for patterns like "Response: ERR" or "Answer: NOT"
        patterns = [
            r"RESPONSE:\s*(ERR|NOT)",
            r"ANSWER:\s*(ERR|NOT)",
            r"RESULT:\s*(ERR|NOT)",
            r"CLASSIFICATION:\s*(ERR|NOT)",
            r"\b(ERR|NOT)\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)

        # Default to NOT if unclear
        return "NOT"


class HuggingFaceLLM(LLMModel):
    """HuggingFace-based LLM model."""

    def __init__(
        self, model_name: str, device: str = "auto", load_in_8bit: bool = False
    ):
        super().__init__(model_name, device)
        self.load_in_8bit = load_in_8bit
        self.pipeline = None

    def load_model(self):
        """Load HuggingFace model and tokenizer."""
        logger = get_logger(f"llm.{self.__class__.__name__}")
        logger.info(f"Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": (
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
        }

        if self.load_in_8bit and self.device.type == "cuda":
            model_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        if not self.load_in_8bit:
            self.model = self.model.to(self.device)

        # Create generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        logger.info(f"Model loaded successfully on {self.device}")

    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        """Generate response using HuggingFace pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=256,  # Limit new tokens for efficiency
            temperature=0.1,  # Low temperature for consistency
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate response
        outputs = self.pipeline(
            prompt,
            generation_config=generation_config,
            return_full_text=False,  # Only return new tokens
        )

        response = outputs[0]["generated_text"]
        return response.strip()


class Llama3Model(HuggingFaceLLM):
    """Llama-style model for critical error detection."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        super().__init__(model_name, device, load_in_8bit)


class Phi3Model(HuggingFaceLLM):
    """Phi-3 model for critical error detection."""

    def __init__(
        self, model_size: str = "mini", device: str = "auto", load_in_8bit: bool = False
    ):
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        super().__init__(model_name, device, load_in_8bit)


class MixtralModel(HuggingFaceLLM):
    """Mixtral-style model for critical error detection."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        super().__init__(model_name, device, load_in_8bit)


def get_llm_model(model_type: str, **kwargs) -> LLMModel:
    """
    Get LLM model by type.

    Args:
        model_type: Type of model ("llama3", "phi3", "mixtral")
        **kwargs: Additional arguments for model initialization

    Returns:
        LLM model instance
    """
    models = {"llama3": Llama3Model, "phi3": Phi3Model, "mixtral": MixtralModel}

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](**kwargs)


class LLMEvaluator:
    """Evaluator for LLM-based critical error detection."""

    def __init__(self, model: LLMModel, prompt_template: PromptTemplate):
        self.model = model
        self.prompt_template = prompt_template

    def evaluate_dataset(
        self, test_data: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """
        Evaluate LLM on test dataset.

        Args:
            test_data: List of (source, target) pairs

        Returns:
            List of (prediction, full_response) tuples
        """
        predictions = []

        logger = get_logger("llm.evaluator")
        logger.info(f"Evaluating {len(test_data)} samples...")
        start_time = time.time()

        with tqdm(test_data, desc="Evaluating samples", unit="sample") as pbar:
            for i, (source, target) in enumerate(pbar):
                try:
                    prediction, response = self.model.predict(
                        source, target, self.prompt_template
                    )
                    predictions.append((prediction, response))

                    # Update progress bar with current metrics
                    if (i + 1) % 5 == 0:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / (i + 1)
                        pbar.set_postfix(
                            {
                                "avg_time": f"{avg_time:.2f}s/sample",
                                "predictions": (
                                    f'{len([p for p, _ in predictions if p == "ERR"])}'
                                    f"/{len(predictions)}"
                                ),
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    predictions.append(("NOT", f"Error: {e}"))

        total_time = time.time() - start_time
        logger.info(
            f"Evaluation completed in {total_time:.1f}s "
            f"({total_time/len(test_data):.2f}s per sample)"
        )

        return predictions

    def calculate_metrics(
        self, predictions: List[str], true_labels: List[str]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
        )

        logger = get_logger("llm.evaluator.metrics")

        # Convert to binary format
        pred_binary = [1 if p == "ERR" else 0 for p in predictions]
        true_binary = [1 if t == "ERR" else 0 for t in true_labels]

        accuracy = accuracy_score(true_binary, pred_binary)
        mcc = matthews_corrcoef(true_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_binary, pred_binary, average="binary", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logger.info(
            f"Calculated metrics: Acc={accuracy:.3f}, MCC={mcc:.3f}, F1={f1:.3f}"
        )
        return metrics
