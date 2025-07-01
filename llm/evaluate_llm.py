#!/usr/bin/env python3
"""
LLM evaluation script for critical error detection.

This script compares generative models (Llama3, DeepSeek) against
the fine-tuned DistilBERT model using zero-shot and few-shot prompting.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.models import LLMEvaluator, get_llm_model  # noqa: E402
from llm.prompts import get_prompt_template  # noqa: E402
from src.data import CustomDataLoader  # noqa: E402
from src.utils import DatasetFileLocator, LanguagePairMapper, get_logger  # noqa: E402


def load_test_data(
    data_dir: str, language_pair: str, sample_size: Optional[int] = None
) -> List[Tuple[str, str, str]]:
    """
    Load test data for evaluation.

    Args:
        data_dir: Directory containing test data
        language_pair: Language pair (e.g., 'en-de')
        sample_size: Optional limit on number of samples

    Returns:
        List of (source, target, label) tuples
    """
    mapper = LanguagePairMapper()
    file_locator = DatasetFileLocator(mapper)

    # Get file paths
    file_paths = file_locator.get_file_paths(data_dir, language_pair)

    if "dev" not in file_paths:
        raise FileNotFoundError(f"No dev file found for {language_pair}")

    # Load data
    loader = CustomDataLoader()
    df = loader.load_train_dev_data(str(file_paths["dev"]))

    # Convert to list of tuples
    data = []
    for _, row in df.iterrows():
        source = row["source"]
        target = row["target"]
        label = "ERR" if row["binary_label"] == 1 else "NOT"
        data.append((source, target, label))

    # Sample if requested
    if sample_size and sample_size < len(data):
        data = data[:sample_size]
        logger = get_logger("llm_evaluation.data")
        logger.info(f"Sampled {sample_size} examples from {len(df)} total")

    return data


def evaluate_llm_model(
    model_type: str,
    prompt_type: str,
    language_pair: str,
    data_dir: str,
    sample_size: Optional[int] = None,
    device: str = "auto",
    load_in_8bit: bool = False,
) -> Dict:
    """
    Evaluate a single LLM model.

    Args:
        model_type: Type of LLM ("llama3", "deepseek")
        prompt_type: Type of prompt ("zero_shot", "few_shot", "chain_of_thought")
        language_pair: Language pair to evaluate on
        data_dir: Directory containing test data
        sample_size: Optional sample size limit
        device: Device to run model on
        load_in_8bit: Whether to load model in 8-bit

    Returns:
        Dictionary with evaluation results
    """
    logger = get_logger(f"llm_evaluation.{model_type}")

    logger.info(f"Starting evaluation: {model_type} with {prompt_type} prompting")
    logger.info(f"Language pair: {language_pair}")

    # Load test data
    logger.info("Loading test data...")
    test_data = load_test_data(data_dir, language_pair, sample_size)
    test_pairs = [(source, target) for source, target, _ in test_data]
    true_labels = [label for _, _, label in test_data]

    err_count = sum(1 for label in true_labels if label == "ERR")
    not_count = sum(1 for label in true_labels if label == "NOT")
    logger.info(f"Loaded {len(test_data)} test samples")
    logger.info(f"Label distribution: ERR={err_count}, NOT={not_count}")

    # Load model
    logger.info(f"Loading {model_type} model...")
    model = get_llm_model(model_type, device=device, load_in_8bit=load_in_8bit)
    model.load_model()

    # Load prompt template
    prompt_template = get_prompt_template(prompt_type)

    # Create evaluator
    evaluator = LLMEvaluator(model, prompt_template)

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_dataset(test_pairs)
    predictions = [pred for pred, _ in results]
    responses = [resp for _, resp in results]

    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = evaluator.calculate_metrics(predictions, true_labels)

    # Log results
    logger.info(f"Results for {model_type} + {prompt_type}:")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"MCC: {metrics['mcc']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"F1: {metrics['f1']:.3f}")

    return {
        "model_type": model_type,
        "prompt_type": prompt_type,
        "language_pair": language_pair,
        "sample_size": len(test_data),
        "metrics": metrics,
        "predictions": predictions,
        "true_labels": true_labels,
        "responses": (
            responses[:10] if len(responses) > 10 else responses
        ),  # Save first 10 responses as examples
    }


def main():
    """Main evaluation function."""
    logger = get_logger("llm_evaluation.main")
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models for critical error detection"
    )

    parser.add_argument(
        "--model",
        choices=["llama3", "deepseek", "all"],
        default="llama3",
        help="LLM model to evaluate",
    )

    parser.add_argument(
        "--prompt",
        choices=["zero_shot", "few_shot", "all"],
        default="zero_shot",
        help="Prompt type to use",
    )

    parser.add_argument(
        "--language-pair", default="en-de", help="Language pair to evaluate on"
    )

    parser.add_argument(
        "--data-dir",
        default="data/catastrophic_errors",
        help="Directory containing test data",
    )

    parser.add_argument(
        "--sample-size", type=int, help="Limit evaluation to N samples (for testing)"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run models on",
    )

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load models in 8-bit for memory efficiency",
    )

    parser.add_argument(
        "--output-dir",
        default="results/llm_evaluation",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models and prompts to evaluate
    models = ["llama3", "deepseek"] if args.model == "all" else [args.model]
    prompts = ["zero_shot", "few_shot"] if args.prompt == "all" else [args.prompt]

    all_results = []

    for model_type in models:
        for prompt_type in prompts:
            try:
                result = evaluate_llm_model(
                    model_type=model_type,
                    prompt_type=prompt_type,
                    language_pair=args.language_pair,
                    data_dir=args.data_dir,
                    sample_size=args.sample_size,
                    device=args.device,
                    load_in_8bit=args.load_in_8bit,
                )
                all_results.append(result)

                # Save individual result
                filename = (
                    f"llm_results_{model_type}_{prompt_type}_{args.language_pair}.json"
                )
                with open(output_dir / filename, "w") as f:
                    json.dump(result, f, indent=2)

            except Exception as e:
                logger.error(f"Error evaluating {model_type} + {prompt_type}: {e}")
                continue

    # Save combined results
    combined_filename = f"llm_evaluation_summary_{args.language_pair}.json"
    with open(output_dir / combined_filename, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<10} {'Prompt':<15} {'Accuracy':<10} {'MCC':<8} {'F1':<8}")
    print("-" * 80)

    for result in all_results:
        model = result["model_type"]
        prompt = result["prompt_type"]
        metrics = result["metrics"]
        print(
            f"{model:<10} {prompt:<15} {metrics['accuracy']:<10.3f} "
            f"{metrics['mcc']:<8.3f} {metrics['f1']:<8.3f}"
        )

    print(f"\nResults saved to: {output_dir}")
    logger.info(f"Evaluation completed. Results saved to: {output_dir}")
    logger.info(f"Evaluated {len(all_results)} model/prompt combinations")


if __name__ == "__main__":
    main()
