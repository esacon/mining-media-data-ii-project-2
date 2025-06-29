#!/usr/bin/env python3
"""
Data Analysis Script for WMT21 Task 3 Critical Error Detection.

This script analyzes the actual data structure and provides insights
into the critical error detection dataset.
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.data_loader import WMT21DataLoader
from src.utils.logging_utils import get_logger, setup_logger


def analyze_language_pair_data(data_dir: str, language_pair: str):
    """Analyze data for a specific language pair."""
    logger = get_logger(__name__)
    loader = WMT21DataLoader()

    # Determine file prefix based on language pair
    prefix_map = {"en-de": "ende", "en-ja": "enja", "en-zh": "enzh", "en-cs": "encs"}

    prefix = prefix_map.get(language_pair, "ende")
    data_path = Path(data_dir)

    # Load train and dev data
    train_file = data_path / f"{prefix}_majority_train.tsv"
    dev_file = data_path / f"{prefix}_majority_dev.tsv"
    test_file = data_path / f"{prefix}_majority_test_blind.tsv"

    results = {"language_pair": language_pair, "files_found": {}}

    # Analyze training data
    if train_file.exists():
        logger.info(f"Analyzing training data: {train_file}")
        train_df = loader.load_train_dev_data(str(train_file))

        results["files_found"]["train"] = True
        results["train"] = {
            "total_samples": len(train_df),
            "critical_errors": sum(train_df["binary_label"]),
            "no_errors": len(train_df) - sum(train_df["binary_label"]),
            "error_rate": sum(train_df["binary_label"]) / len(train_df),
            "agreement_stats": loader.analyze_annotator_agreement(train_df),
        }

        # Analyze individual scores distribution
        score_patterns = Counter()
        for scores in train_df["individual_scores"]:
            score_patterns[tuple(scores)] += 1

        results["train"]["score_patterns"] = dict(score_patterns.most_common(10))

        logger.info(
            f"Train - Samples: {len(train_df)}, Critical Errors: {sum(train_df['binary_label'])} ({sum(train_df['binary_label'])/len(train_df)*100:.1f}%)"
        )
    else:
        results["files_found"]["train"] = False
        logger.warning(f"Training file not found: {train_file}")

    # Analyze development data
    if dev_file.exists():
        logger.info(f"Analyzing development data: {dev_file}")
        dev_df = loader.load_train_dev_data(str(dev_file))

        results["files_found"]["dev"] = True
        results["dev"] = {
            "total_samples": len(dev_df),
            "critical_errors": sum(dev_df["binary_label"]),
            "no_errors": len(dev_df) - sum(dev_df["binary_label"]),
            "error_rate": sum(dev_df["binary_label"]) / len(dev_df),
            "agreement_stats": loader.analyze_annotator_agreement(dev_df),
        }

        logger.info(
            f"Dev - Samples: {len(dev_df)}, Critical Errors: {sum(dev_df['binary_label'])} ({sum(dev_df['binary_label'])/len(dev_df)*100:.1f}%)"
        )
    else:
        results["files_found"]["dev"] = False
        logger.warning(f"Development file not found: {dev_file}")

    # Analyze test data
    if test_file.exists():
        logger.info(f"Analyzing test data: {test_file}")
        test_df = loader.load_test_data(str(test_file))

        results["files_found"]["test"] = True
        results["test"] = {"total_samples": len(test_df)}

        logger.info(f"Test - Samples: {len(test_df)} (blind test, no labels)")
    else:
        results["files_found"]["test"] = False
        logger.warning(f"Test file not found: {test_file}")

    return results


def create_visualizations(results_by_language, output_dir: str):
    """Create visualizations of the data analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Error rates by language pair
    language_pairs = []
    train_error_rates = []
    dev_error_rates = []

    for lp, data in results_by_language.items():
        if "train" in data and "dev" in data:
            language_pairs.append(lp)
            train_error_rates.append(data["train"]["error_rate"] * 100)
            dev_error_rates.append(data["dev"]["error_rate"] * 100)

    if language_pairs:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(language_pairs))
        width = 0.35

        ax.bar([i - width / 2 for i in x], train_error_rates, width, label="Training", alpha=0.8)
        ax.bar([i + width / 2 for i in x], dev_error_rates, width, label="Development", alpha=0.8)

        ax.set_xlabel("Language Pair")
        ax.set_ylabel("Critical Error Rate (%)")
        ax.set_title("Critical Error Rates by Language Pair")
        ax.set_xticks(x)
        ax.set_xticklabels(language_pairs)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / "error_rates_by_language.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 2. Sample size distribution
    if language_pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        train_sizes = [results_by_language[lp]["train"]["total_samples"] for lp in language_pairs]
        dev_sizes = [results_by_language[lp]["dev"]["total_samples"] for lp in language_pairs]
        test_sizes = [
            results_by_language[lp].get("test", {}).get("total_samples", 0) for lp in language_pairs
        ]

        # Training and dev sizes
        ax1.bar(language_pairs, train_sizes, alpha=0.8, label="Training")
        ax1.bar(language_pairs, dev_sizes, alpha=0.8, label="Development")
        ax1.set_title("Dataset Sizes by Language Pair")
        ax1.set_ylabel("Number of Samples")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)

        # Test sizes
        ax2.bar(language_pairs, test_sizes, alpha=0.8, color="orange")
        ax2.set_title("Test Dataset Sizes")
        ax2.set_ylabel("Number of Samples")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / "sample_sizes_by_language.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Annotator agreement statistics
    if language_pairs:
        agreement_data = []
        for lp in language_pairs:
            if "train" in results_by_language[lp]:
                stats = results_by_language[lp]["train"]["agreement_stats"]
                agreement_data.append(
                    {
                        "Language Pair": lp,
                        "Mean Agreement": stats["mean_agreement"],
                        "Full Agreement": stats["full_agreement_ratio"],
                        "Majority Agreement": stats["majority_agreement_ratio"],
                    }
                )

        if agreement_data:
            df_agreement = pd.DataFrame(agreement_data)

            fig, ax = plt.subplots(figsize=(12, 6))
            x = range(len(language_pairs))
            width = 0.25

            ax.bar(
                [i - width for i in x],
                df_agreement["Mean Agreement"],
                width,
                label="Mean Agreement",
                alpha=0.8,
            )
            ax.bar(x, df_agreement["Full Agreement"], width, label="Full Agreement", alpha=0.8)
            ax.bar(
                [i + width for i in x],
                df_agreement["Majority Agreement"],
                width,
                label="Majority Agreement",
                alpha=0.8,
            )

            ax.set_xlabel("Language Pair")
            ax.set_ylabel("Agreement Ratio")
            ax.set_title("Inter-Annotator Agreement by Language Pair")
            ax.set_xticks(x)
            ax.set_xticklabels(language_pairs)
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / "annotator_agreement.png", dpi=300, bbox_inches="tight")
            plt.close()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze WMT21 Task 3 data")
    parser.add_argument(
        "--data-dir",
        default="src/data/catastrophic_errors",
        help="Directory containing the WMT21 data files",
    )
    parser.add_argument(
        "--output-dir", default="results/data_analysis", help="Directory to save analysis results"
    )
    parser.add_argument(
        "--language-pairs",
        nargs="+",
        default=["en-de", "en-ja", "en-zh", "en-cs"],
        help="Language pairs to analyze",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(name="analyze_data", level_str="INFO", log_to_console=True)
    logger = get_logger(__name__)

    logger.info("Starting WMT21 Task 3 data analysis")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Language pairs: {args.language_pairs}")

    # Analyze each language pair
    results_by_language = {}

    for language_pair in args.language_pairs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {language_pair}")
        logger.info(f"{'='*50}")

        try:
            results = analyze_language_pair_data(args.data_dir, language_pair)
            results_by_language[language_pair] = results
        except Exception as e:
            logger.error(f"Error analyzing {language_pair}: {e}")
            continue

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    import json

    with open(output_path / "analysis_results.json", "w") as f:
        json.dump(results_by_language, f, indent=2)

    # Create visualizations
    logger.info("Creating visualizations...")
    create_visualizations(results_by_language, args.output_dir)

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")

    for lp, results in results_by_language.items():
        logger.info(f"\n{lp.upper()}:")

        if "train" in results:
            train = results["train"]
            logger.info(
                f"  Training: {train['total_samples']} samples, "
                f"{train['critical_errors']} errors ({train['error_rate']*100:.1f}%)"
            )

        if "dev" in results:
            dev = results["dev"]
            logger.info(
                f"  Development: {dev['total_samples']} samples, "
                f"{dev['critical_errors']} errors ({dev['error_rate']*100:.1f}%)"
            )

        if "test" in results:
            test = results["test"]
            logger.info(f"  Test: {test['total_samples']} samples (blind)")

    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main()
