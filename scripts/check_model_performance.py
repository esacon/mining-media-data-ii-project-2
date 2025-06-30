import json
import sys
from pathlib import Path

import pandas as pd


def load_json_safely(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def analyze_training_history(results_dir="results"):
    checkpoints_dir = Path(results_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found. Train a model first!")
        return

    print("🔍 TRAINING HISTORY ANALYSIS")
    print("=" * 50)

    history_files = list(checkpoints_dir.glob("training_history_*.json"))

    if not history_files:
        print("❌ No training history files found!")
        return

    results = []

    for history_file in sorted(history_files):
        data = load_json_safely(str(history_file))
        if data:
            filename = history_file.stem
            parts = filename.replace("training_history_", "").split("_")
            timestamp = "_".join(parts[:2])
            language = "_".join(parts[2:]) if len(parts) > 2 else "unknown"

            final_epoch = len(data.get("val_mcc", []))
            final_mcc = data["val_mcc"][-1] if data.get("val_mcc") else 0
            final_accuracy = data["val_accuracy"][-1] if data.get("val_accuracy") else 0
            final_val_loss = data["val_loss"][-1] if data.get("val_loss") else 0
            best_mcc = max(data.get("val_mcc", [0]))

            results.append(
                {
                    "Language": language,
                    "Timestamp": timestamp,
                    "Epochs": final_epoch,
                    "Final_MCC": final_mcc,
                    "Best_MCC": best_mcc,
                    "Final_Accuracy": final_accuracy,
                    "Final_Val_Loss": final_val_loss,
                    "File": history_file.name,
                }
            )

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(["Language", "Best_MCC"], ascending=[True, False])

        print(f"📊 Found {len(results)} trained models:\n")

        for _, row in df.iterrows():
            status = (
                "🟢 EXCELLENT"
                if row["Best_MCC"] > 0.7
                else (
                    "🟡 GOOD"
                    if row["Best_MCC"] > 0.5
                    else "🟠 FAIR" if row["Best_MCC"] > 0.3 else "🔴 POOR"
                )
            )

            print(
                f"{status} | {row['Language']:>6} | MCC: {row['Best_MCC']:.3f} "
                f"| Acc: {row['Final_Accuracy']:.3f} | {row['Timestamp']}"
            )

        print("\n🏆 BEST MODELS BY LANGUAGE:")
        best_per_lang = df.groupby("Language")["Best_MCC"].idxmax()
        for idx in best_per_lang:
            row = df.loc[idx]
            print(
                f"  {row['Language']:>6}: MCC {row['Best_MCC']:.3f} "
                f"({row['Timestamp']})"
            )


def analyze_evaluation_results(results_dir="results"):
    eval_file = Path(results_dir) / "evaluation_results.json"

    if eval_file.exists():
        print("\n🎯 LATEST EVALUATION RESULTS")
        print("=" * 30)

        data = load_json_safely(str(eval_file))
        if data:
            print(f"📈 Accuracy:  {data.get('accuracy', 0):.3f}")
            print(f"📊 MCC:       {data.get('mcc', 0):.3f}")
            print(f"🎯 Precision: {data.get('precision', 0):.3f}")
            print(f"🔍 Recall:    {data.get('recall', 0):.3f}")
            print(f"⚖️  F1 Score:  {data.get('f1', 0):.3f}")
            print(f"📊 AUC-ROC:   {data.get('auc_roc', 0):.3f}")


def check_available_models(results_dir="results"):
    checkpoints_dir = Path(results_dir) / "checkpoints"

    if not checkpoints_dir.exists():
        print("❌ No checkpoints directory found!")
        return

    print("\n💾 AVAILABLE MODEL CHECKPOINTS")
    print("=" * 35)

    best_models = list(checkpoints_dir.glob("best_model_*.pt"))
    final_models = list(checkpoints_dir.glob("final_model_*.pt"))

    print(f"🏆 Best Models: {len(best_models)}")
    for model in sorted(best_models):
        print(f"   {model.name}")

    print(f"\n🎯 Final Models: {len(final_models)}")
    for model in sorted(final_models):
        print(f"   {model.name}")


def interpret_metrics():
    print("\n📚 METRIC INTERPRETATION GUIDE")
    print("=" * 32)
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


def main():
    print("🚀 MODEL PERFORMANCE CHECKER")
    print("=" * 50)

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results"

    if not Path(results_dir).exists():
        print(f"❌ Results directory '{results_dir}' not found!")
        print("💡 Train a model first: make train LANG=en-de")
        return

    analyze_training_history(results_dir)
    analyze_evaluation_results(results_dir)
    check_available_models(results_dir)
    interpret_metrics()

    print("\n🛠️  QUICK COMMANDS:")
    print("   make status")
    print("   make evaluate LANG=en-de")
    print("   ./scripts/test_all_languages.sh debug")


if __name__ == "__main__":
    main()
