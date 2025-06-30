#!/usr/bin/env python3
"""
Simple CLI for running the Critical Error Detection runner.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from src.runner import Runner  # noqa: E402


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Critical Error Detection Pipeline CLI"
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training/inference",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data", required=True, help="Path to training data")
    train_parser.add_argument(
        "--language-pair", help="Language pair filter (e.g., 'en-de')"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--data", required=True, help="Path to evaluation data")
    eval_parser.add_argument(
        "--language-pair", help="Language pair filter (e.g., 'en-de')"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model", required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--data", required=True, help="Path to data for prediction"
    )
    predict_parser.add_argument(
        "--language-pair", help="Language pair filter (e.g., 'en-de')"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument("--data-dir", help="Directory containing data files")
    analyze_parser.add_argument(
        "--language-pairs", nargs="+", help="Language pairs to analyze"
    )

    # Experiment command
    experiment_parser = subparsers.add_parser("experiment", help="Run full experiment")
    experiment_parser.add_argument(
        "--data", required=True, help="Path to training data"
    )
    experiment_parser.add_argument(
        "--language-pair", help="Language pair filter (e.g., 'en-de')"
    )

    # Summary command
    subparsers.add_parser("summary", help="Show runner summary")

    return parser.parse_args()


def main():
    """Main CLI function."""
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use --help for usage information.")
        sys.exit(1)

    try:
        # Initialize runner
        runner = Runner(args.config, args.device)

        if args.command == "train":
            print(f"🚀 Training model with data: {args.data}")
            if args.language_pair:
                print(f"   Language pair: {args.language_pair}")

            results = runner.train(args.data, args.language_pair)

            print("✅ Training completed!")
            print(f"   Final metrics: {results.get('final_metrics', 'N/A')}")

        elif args.command == "evaluate":
            print(f"📊 Evaluating model: {args.model}")
            print(f"   Data: {args.data}")
            if args.language_pair:
                print(f"   Language pair: {args.language_pair}")

            results = runner.evaluate(args.model, args.data, args.language_pair)

            print("✅ Evaluation completed!")
            print(f"   Accuracy: {results.get('accuracy', 0):.4f}")
            print(f"   MCC: {results.get('mcc', 0):.4f}")
            print(f"   F1: {results.get('f1', 0):.4f}")

        elif args.command == "predict":
            print(f"🔮 Making predictions with model: {args.model}")
            print(f"   Data: {args.data}")
            if args.language_pair:
                print(f"   Language pair: {args.language_pair}")

            predictions, _ = runner.predict(args.model, args.data, args.language_pair)

            print("✅ Predictions completed!")
            print(f"   Total predictions: {len(predictions)}")
            crit_errs = sum(predictions)
            percent = (crit_errs / len(predictions)) * 100
            print(f"   Critical errors detected: {crit_errs} ({percent:.1f}%)")

        elif args.command == "analyze":
            print("📈 Analyzing data...")
            if args.data_dir:
                print(f"   Data directory: {args.data_dir}")
            if args.language_pairs:
                print(f"   Language pairs: {args.language_pairs}")

            results = runner.analyze_data(args.data_dir, args.language_pairs)

            print("✅ Data analysis completed!")
            for lang_pair, analysis in results.items():
                print(f"\n   {lang_pair}:")
                if "train" in analysis:
                    print(
                        f"     Train: {analysis['train']['samples']} samples, "
                        f"{analysis['train']['error_rate']:.3f} error rate"
                    )
                if "dev" in analysis:
                    print(
                        f"     Dev: {analysis['dev']['samples']} samples, "
                        f"{analysis['dev']['error_rate']:.3f} error rate"
                    )

        elif args.command == "experiment":
            print(f"🧪 Running full experiment with data: {args.data}")
            if args.language_pair:
                print(f"   Language pair: {args.language_pair}")

            results = runner.run_full_experiment(args.data, args.language_pair)

            print("✅ Experiment completed!")
            print(
                f"\tTraining metrics: {results['training'].get('final_metrics', 'N/A')}"
            )
            print(f"\tEvaluation metrics: {results['evaluation']}")
            print(f"\tModel saved: {results['model_path']}")

        elif args.command == "summary":
            summary = runner.summary()

            print("📋 Pipeline Summary:")
            print(f"\tConfig: {summary['config_path']}")
            print(f"\tDevice: {summary['device']}")
            print(f"\tModel: {summary['model_name']}")
            print(f"\tData directory: {summary['data_directory']}")
            print(f"\tTraining config: {summary['training_config']}")
            print(
                f"\tAvailable operations: {', '.join(summary['available_operations'])}"
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
