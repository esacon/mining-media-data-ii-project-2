#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import Runner  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the Critical Error Detection CLI.

    Returns:
        An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Critical Error Detection Pipeline CLI"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training/inference",
    )

    parser.add_argument(
        "--debug-test",
        action="store_true",
        help="Enable quick test mode with reduced sample size and faster training",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--data", type=str, required=True, help="Path to training data"
    )
    train_parser.add_argument(
        "--language-pair",
        type=str,
        help="Language pair filter (e.g., 'en-de')",
    )
    train_parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit dataset to this many samples (for quick testing)",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--data", type=str, required=True, help="Path to evaluation data"
    )
    eval_parser.add_argument(
        "--language-pair",
        type=str,
        help="Language pair filter (e.g., 'en-de')",
    )
    eval_parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit dataset to this many samples (for quick testing)",
    )

    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    predict_parser.add_argument(
        "--data", type=str, required=True, help="Path to data for prediction"
    )
    predict_parser.add_argument(
        "--language-pair",
        type=str,
        help="Language pair filter (e.g., 'en-de')",
    )
    predict_parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit dataset to this many samples (for quick testing)",
    )
    predict_parser.add_argument(
        "--no-evaluation-format",
        action="store_true",
        help="Skip creating WMT evaluation format files",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument(
        "--data-dir", type=str, help="Directory containing data files"
    )
    analyze_parser.add_argument(
        "--language-pairs",
        nargs="+",
        type=str,
        help="Language pairs to analyze",
    )

    experiment_parser = subparsers.add_parser("experiment", help="Run full experiment")
    experiment_parser.add_argument(
        "--data", type=str, required=True, help="Path to training data"
    )
    experiment_parser.add_argument(
        "--language-pair",
        type=str,
        help="Language pair filter (e.g., 'en-de')",
    )
    experiment_parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit dataset to this many samples (for quick testing)",
    )

    return parser.parse_args()


def handle_train_command(runner: Runner, args: argparse.Namespace):
    """
    Handles the 'train' command, initiating model training.

    Args:
        runner: An initialized Runner instance.
        args: Parsed command-line arguments.
    """
    kwargs = {}
    if args.language_pair:
        kwargs["language_pair"] = args.language_pair
    if getattr(args, "sample_size", None):
        kwargs["sample_size"] = args.sample_size

    runner.train(args.data, **kwargs)


def handle_evaluate_command(runner: Runner, args: argparse.Namespace):
    """
    Handles the 'evaluate' command, evaluating a trained model.

    Args:
        runner: An initialized Runner instance.
        args: Parsed command-line arguments.
    """
    kwargs = {}
    if args.language_pair:
        kwargs["language_pair"] = args.language_pair
    if getattr(args, "sample_size", None):
        kwargs["sample_size"] = args.sample_size

    runner.evaluate(args.model, args.data, **kwargs)


def handle_predict_command(runner: Runner, args: argparse.Namespace):
    """
    Handles the 'predict' command, making predictions with a trained model.

    Args:
        runner: An initialized Runner instance.
        args: Parsed command-line arguments.
    """
    kwargs = {}
    if args.language_pair:
        kwargs["language_pair"] = args.language_pair
    if getattr(args, "sample_size", None):
        kwargs["sample_size"] = args.sample_size

    create_evaluation_format = not getattr(args, "no_evaluation_format", False)

    runner.predict(
        args.model,
        args.data,
        create_evaluation_format=create_evaluation_format,
        method_name="distilbert",
        **kwargs,
    )


def handle_analyze_command(runner: Runner, args: argparse.Namespace):
    """
    Handles the 'analyze' command, performing data analysis.

    Args:
        runner: An initialized Runner instance.
        args: Parsed command-line arguments.
    """
    runner.analyze_data(args.data_dir, args.language_pairs)


def handle_experiment_command(runner: Runner, args: argparse.Namespace):
    """
    Handles the 'experiment' command, running a full training and evaluation experiment.

    Args:
        runner: An initialized Runner instance.
        args: Parsed command-line arguments.
    """
    kwargs = {}
    if args.language_pair:
        kwargs["language_pair"] = args.language_pair
    if getattr(args, "sample_size", None):
        kwargs["sample_size"] = args.sample_size

    runner.run_full_experiment(args.data, **kwargs)


def main():
    """
    Main function for the Critical Error Detection CLI.
    Parses arguments and dispatches to appropriate command handlers.
    """
    args: argparse.Namespace = parse_args()

    if args.command is None:
        sys.exit(1)

    try:
        runner = Runner(args.config, args.device, getattr(args, "debug_test", False))

        if args.command == "train":
            handle_train_command(runner, args)
        elif args.command == "evaluate":
            handle_evaluate_command(runner, args)
        elif args.command == "predict":
            handle_predict_command(runner, args)
        elif args.command == "analyze":
            handle_analyze_command(runner, args)
        elif args.command == "experiment":
            handle_experiment_command(runner, args)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
