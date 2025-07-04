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
    pipeline,
)

try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.prompts import PromptTemplate  # noqa: E402
from src.utils import get_logger  # noqa: E402


class LLMModel(ABC):
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.tokenizer = None
        self.model = None

    def _setup_device(self, device: str) -> torch.device:
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
        pass

    @abstractmethod
    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        pass

    def predict(
        self, source: str, target: str, prompt_template: PromptTemplate
    ) -> Tuple[str, str]:
        prompt = prompt_template.format(source=source, target=target)
        full_response = self.generate_response(prompt)
        prediction = self._extract_prediction(full_response)
        return prediction, full_response

    def _extract_prediction(self, response: str) -> str:
        response = response.strip()

        if not response:
            return "NOT"

        response_upper = response.upper()

        if response_upper == "ERR":
            return "ERR"
        elif response_upper == "NOT":
            return "NOT"

        if response_upper.startswith("ERR"):
            return "ERR"
        elif response_upper.startswith("NOT"):
            return "NOT"

        err_match = re.search(r"\bERR\b", response_upper)
        not_match = re.search(r"\bNOT\b", response_upper)

        if err_match and err_match.start() < 50:
            if (
                not_match
                and not_match.start() < 50
                and not_match.start() < err_match.start()
            ):
                return "NOT"
            return "ERR"
        elif not_match and not_match.start() < 50:
            return "NOT"

        return "NOT"


class HuggingFaceLLM(LLMModel):
    def __init__(
        self, model_name: str, device: str = "auto", load_in_8bit: bool = False
    ):
        super().__init__(model_name, device)
        self.load_in_8bit = load_in_8bit
        self.pipeline = None

    def load_model(self):
        logger = get_logger(f"llm.{self.__class__.__name__}")
        logger.info(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": (
                torch.float16 if self.device.type == "cuda" else torch.float32
            ),
        }

        if self.load_in_8bit and self.device.type == "cuda":
            if BITSANDBYTES_AVAILABLE:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                except Exception:
                    self.load_in_8bit = False
            else:
                self.load_in_8bit = False

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        if not self.load_in_8bit:
            self.model = self.model.to(self.device)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        logger.info(f"Model loaded successfully on {self.device}")

    def generate_response(self, prompt: str, max_length: int = 1024) -> str:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            generation_config = {
                "max_new_tokens": 48,
                "temperature": 0.1,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "return_full_text": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            outputs = self.pipeline(prompt, **generation_config)
            response = outputs[0]["generated_text"]

            return response.strip()

        except Exception as e:
            logger = get_logger(f"llm.{self.__class__.__name__}")
            logger.warning(
                f"Pipeline generation failed: {e}. Trying direct model generation."
            )

            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            if not self.load_in_8bit:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=48,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()


class Llama3Model(HuggingFaceLLM):
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        super().__init__(model_name, device, load_in_8bit)


class DeepSeekModel(HuggingFaceLLM):
    def __init__(
        self,
        model_size: str = "8b",
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
        super().__init__(model_name, device, load_in_8bit)


def get_llm_model(model_type: str, **kwargs) -> LLMModel:
    models = {"llama3": Llama3Model, "deepseek": DeepSeekModel}

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(models.keys())}"
        )

    return models[model_type](**kwargs)


class LLMEvaluator:
    def __init__(self, model: LLMModel, prompt_template: PromptTemplate):
        self.model = model
        self.prompt_template = prompt_template

    def evaluate_dataset(
        self, test_data: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
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
        from sklearn.metrics import (
            accuracy_score,
            matthews_corrcoef,
            precision_recall_fscore_support,
        )

        logger = get_logger("llm.evaluator.metrics")

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
