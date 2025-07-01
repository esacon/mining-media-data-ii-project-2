import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_json_safely(file_path: Path) -> Optional[dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_performance_status(mcc_score: float) -> str:
    if mcc_score > 0.7:
        return "🟢 EXCELLENT"
    elif mcc_score > 0.5:
        return "🟡 GOOD    "
    elif mcc_score > 0.3:
        return "🟠 FAIR    "
    elif mcc_score > 0.0:
        return "🔴 POOR    "
    else:
        return "❌ TERRIBLE"


def get_difficulty_level(avg_mcc: float) -> str:
    if avg_mcc > 0.6:
        return "🟢 EASY   "
    elif avg_mcc > 0.4:
        return "🟡 MEDIUM "
    elif avg_mcc > 0.2:
        return "🟠 HARD   "
    else:
        return "🔴 V.HARD "


def parse_training_filename(filename: str) -> Optional[tuple[str, str]]:
    parts = filename.replace("training_history_", "").split("_")
    if len(parts) >= 3:
        timestamp = "_".join(parts[:2])
        language = "_".join(parts[2:])
        return timestamp, language
    return None


def analyze_training_history(results_dir: str = "results") -> Optional[pd.DataFrame]:
    checkpoints_dir = Path(results_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found. Train a model first!")
        return None

    print("🔍 TRAINING HISTORY ANALYSIS")
    print("=" * 50)

    history_files = sorted(list(checkpoints_dir.glob("training_history_*.json")))

    if not history_files:
        print("❌ No training history files found!")
        return None

    all_results = []
    for history_file in history_files:
        data = load_json_safely(history_file)
        if data:
            parsed_info = parse_training_filename(history_file.stem)
            if parsed_info:
                timestamp, language = parsed_info

                val_mcc = data.get("val_mcc", [])
                val_accuracy = data.get("val_accuracy", [])
                val_loss = data.get("val_loss", [])
                train_loss = data.get("train_loss", [])

                final_epoch = len(val_mcc)
                final_mcc = val_mcc[-1] if val_mcc else 0
                final_accuracy = val_accuracy[-1] if val_accuracy else 0
                final_val_loss = val_loss[-1] if val_loss else 0
                final_train_loss = train_loss[-1] if train_loss else 0
                best_mcc = max(val_mcc) if val_mcc else 0
                best_accuracy = max(val_accuracy) if val_accuracy else 0
                min_val_loss = min(val_loss) if val_loss else float("inf")

                all_results.append(
                    {
                        "Language": language,
                        "Timestamp": timestamp,
                        "Epochs": final_epoch,
                        "Final_MCC": final_mcc,
                        "Best_MCC": best_mcc,
                        "Final_Accuracy": final_accuracy,
                        "Best_Accuracy": best_accuracy,
                        "Final_Val_Loss": final_val_loss,
                        "Min_Val_Loss": min_val_loss,
                        "Final_Train_Loss": final_train_loss,
                        "File": history_file.name,
                    }
                )

    if not all_results:
        print("❌ No valid training results found!")
        return None

    df = pd.DataFrame(all_results).sort_values(
        ["Language", "Best_MCC"], ascending=[True, False]
    )

    print(f"📊 Found {len(all_results)} trained models:\n")

    print("🎯 INDIVIDUAL MODEL PERFORMANCE:")
    print("-" * 80)
    for _, row in df.iterrows():
        status = get_performance_status(row["Best_MCC"])
        print(
            f"{status} | {row['Language']:>6} | MCC: {row['Best_MCC']:.3f} "
            f"| Acc: {row['Final_Accuracy']:.3f} | {row['Timestamp']}"
        )

    print("\n🏆 BEST MODELS BY LANGUAGE:")
    print("-" * 30)
    for language in sorted(df["Language"].unique()):
        lang_df = df[df["Language"] == language]
        best_row = lang_df.loc[lang_df["Best_MCC"].idxmax()]
        status = get_performance_status(best_row["Best_MCC"])
        print(
            f"{status} {language:>6}: MCC {best_row['Best_MCC']:.3f} "
            f"({best_row['Timestamp']})"
        )

    print("\n🌍 LANGUAGE DIFFICULTY RANKING:")
    print("-" * 35)
    lang_analysis = (
        df.groupby("Language")
        .agg(
            MCC_Mean=("Best_MCC", "mean"),
            MCC_Std=("Best_MCC", "std"),
            Acc_Mean=("Best_Accuracy", "mean"),
            Models_Count=("Language", "count"),
        )
        .round(3)
        .sort_values("MCC_Mean", ascending=False)
    )

    for language, row in lang_analysis.iterrows():
        difficulty = get_difficulty_level(row["MCC_Mean"])
        print(
            f"{difficulty} {language:>6}: "
            f"MCC {row['MCC_Mean']:.3f}±{row['MCC_Std']:.3f} | "
            f"Acc {row['Acc_Mean']:.3f} | Models: {row['Models_Count']}"
        )
    return df


def analyze_evaluation_results(
    results_dir: str = "results",
) -> Optional[Dict[str, Any]]:
    eval_file = Path(results_dir) / "evaluation_results.json"

    if not eval_file.exists():
        print("\n❌ No evaluation results found!")
        return None

    print("\n🎯 LATEST EVALUATION RESULTS")
    print("=" * 30)

    data = load_json_safely(eval_file)
    if data:
        metrics = {
            "accuracy": data.get("accuracy", 0),
            "mcc": data.get("mcc", 0),
            "precision": data.get("precision", 0),
            "recall": data.get("recall", 0),
            "f1": data.get("f1", 0),
            "auc_roc": data.get("auc_roc", 0),
        }

        print(f"📈 Accuracy:  {metrics['accuracy']:.3f}")
        print(f"📊 MCC:       {metrics['mcc']:.3f}")
        print(f"🎯 Precision: {metrics['precision']:.3f}")
        print(f"🔍 Recall:    {metrics['recall']:.3f}")
        print(f"⚖️  F1 Score:  {metrics['f1']:.3f}")
        print(f"📊 AUC-ROC:   {metrics['auc_roc']:.3f}")

        return metrics

    return None


def check_available_models(results_dir: str = "results") -> Dict[str, List[str]]:
    checkpoints_dir = Path(results_dir) / "checkpoints"

    model_info = {"best_models": [], "final_models": [], "all_models": []}

    if not checkpoints_dir.exists():
        print("\n❌ No checkpoints directory found!")
        return model_info

    print("\n💾 AVAILABLE MODEL CHECKPOINTS")
    print("=" * 35)

    best_models = sorted(list(checkpoints_dir.glob("best_model_*.pt")))
    final_models = sorted(list(checkpoints_dir.glob("final_model_*.pt")))
    all_model_files = sorted(list(checkpoints_dir.glob("*.pt")))

    model_info["best_models"] = [model.name for model in best_models]
    model_info["final_models"] = [model.name for model in final_models]
    model_info["all_models"] = [model.name for model in all_model_files]

    print(f"🏆 Best Models: {len(best_models)}")
    for model in best_models:
        print(f"   {model.name}")

    print(f"\n🎯 Final Models: {len(final_models)}")
    for model in final_models:
        print(f"   {model.name}")

    return model_info


def save_final_results(
    training_df: Optional[pd.DataFrame],
    evaluation_metrics: Optional[Dict[str, Any]],
    model_info: Dict[str, List[str]],
    output_dir: str = "results/analysis/traditional",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 SAVING RESULTS TO: {output_dir}")
    print("-" * 30)

    summary_stats: Dict[str, Any] = {
        "total_models": len(training_df) if training_df is not None else 0,
        "languages": (
            sorted(training_df["Language"].unique().tolist())
            if training_df is not None
            else []
        ),
        "available_models": model_info,
        "latest_evaluation": evaluation_metrics,
    }

    if training_df is not None:
        best_idx = training_df["Best_MCC"].idxmax()
        best_model = training_df.loc[best_idx]

        summary_stats["best_overall"] = {
            "language": best_model["Language"],
            "timestamp": best_model["Timestamp"],
            "mcc": best_model["Best_MCC"],
            "accuracy": best_model["Best_Accuracy"],
            "file": best_model["File"],
        }

        summary_stats["average_metrics"] = {
            "mcc": training_df["Best_MCC"].mean(),
            "accuracy": training_df["Best_Accuracy"].mean(),
            "final_val_loss": training_df["Final_Val_Loss"].mean(),
        }

    with open(output_path / "model_performance_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    if training_df is not None:
        training_df.to_csv(output_path / "training_detailed_results.csv", index=False)

        best_by_lang = {}
        for language in training_df["Language"].unique():
            lang_df = training_df[training_df["Language"] == language]
            best_row = lang_df.loc[lang_df["Best_MCC"].idxmax()]

            best_by_lang[language] = {
                "timestamp": best_row["Timestamp"],
                "mcc": best_row["Best_MCC"],
                "accuracy": best_row["Best_Accuracy"],
                "file": best_row["File"],
            }

        with open(output_path / "best_models_by_language.json", "w") as f:
            json.dump(best_by_lang, f, indent=2)

        lang_comparison = (
            training_df.groupby("Language")
            .agg(
                Best_MCC_Mean=("Best_MCC", "mean"),
                Best_MCC_Std=("Best_MCC", "std"),
                Best_MCC_Min=("Best_MCC", "min"),
                Best_MCC_Max=("Best_MCC", "max"),
                Best_Accuracy_Mean=("Best_Accuracy", "mean"),
                Best_Accuracy_Std=("Best_Accuracy", "std"),
                Final_Val_Loss_Mean=("Final_Val_Loss", "mean"),
                Final_Val_Loss_Std=("Final_Val_Loss", "std"),
            )
            .round(4)
        )
        lang_comparison.to_csv(output_path / "language_comparison.csv")

    if evaluation_metrics:
        with open(output_path / "latest_evaluation_metrics.json", "w") as f:
            json.dump(evaluation_metrics, f, indent=2)

    print(f"✅ Summary saved to: {output_path / 'model_performance_summary.json'}")
    if training_df is not None:
        print(
            f"✅ Training results saved to: {output_path / 'training_detailed_results.csv'}"  # noqa: E501
        )
        print(
            f"✅ Best models saved to: {output_path / 'best_models_by_language.json'}"  # noqa: E501
        )
        print(
            f"✅ Language comparison saved to: {output_path / 'language_comparison.csv'}"  # noqa: E501
        )
    if evaluation_metrics:
        print(
            f"✅ Evaluation metrics saved to: {output_path / 'latest_evaluation_metrics.json'}"  # noqa: E501
        )


def interpret_metrics():
    print("\n📚 MODEL EVALUATION METRICS GUIDE")
    print("=" * 35)
    print("🎯 MCC (Matthews Correlation Coefficient) - PRIMARY METRIC:")
    print("   🟢 0.7+  : Excellent model")
    print("   🟡 0.5-0.7: Good model")
    print("   🟠 0.3-0.5: Fair model")
    print("   🔴 0.0-0.3: Poor model")
    print("   ❌ <0.0  : Worse than random")

    print("\n📊 Other Metrics:")
    print("   • Accuracy: Overall correctness (aim for >0.8)")
    print("   • Precision: Of predicted errors, how many were real? (aim for >0.7)")
    print("   • Recall: Of real errors, how many did we catch? (aim for >0.7)")
    print("   • F1: Balance of precision and recall (aim for >0.7)")
    print("   • AUC-ROC: Model's ranking ability (aim for >0.8)")

    print("\n🏗️ Training Metrics:")
    print("   • Val Loss: Validation loss (lower is better)")
    print("   • Train Loss: Training loss (should decrease over epochs)")
    print("   • Overfitting: Large gap between train and val loss")

    print("\n🌍 Language Pairs:")
    print("   • en-cs: English to Czech")
    print("   • en-de: English to German")
    print("   • en-ja: English to Japanese")
    print("   • en-zh: English to Chinese")


def main():
    print("🚀 MODEL PERFORMANCE CHECKER")
    print("=" * 50)

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Results directory '{results_dir}' not found!")
        print("💡 Train a model first: make train LANG=en-de")
        return

    training_df = analyze_training_history(results_dir)
    evaluation_metrics = analyze_evaluation_results(results_dir)
    model_info = check_available_models(results_dir)

    save_final_results(training_df, evaluation_metrics, model_info)

    interpret_metrics()

    print("\n🛠️  QUICK COMMANDS:")
    print("   make status")
    print("   make evaluate LANG=en-de")
    print("   make check-performance")


if __name__ == "__main__":
    main()
