import json
import sys
from pathlib import Path

import pandas as pd


def load_json_safely(file_path: Path) -> dict | None:
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


def parse_filename(filename: str) -> tuple[str, str, str] | None:
    parts = filename.replace("llm_results_", "").split("_")
    if len(parts) >= 4:
        model = parts[0]
        if parts[1] in ["few", "zero"] and parts[2] == "shot":
            prompt_type = f"{parts[1]}_shot"
            language_pair = "_".join(parts[3:])
        else:
            prompt_type = parts[1]
            language_pair = "_".join(parts[2:])
        return model, prompt_type, language_pair
    return None


def analyze_llm_results(
    results_dir: str = "results/llm_evaluation",
) -> pd.DataFrame | None:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ LLM evaluation directory '{results_dir}' not found!")
        print("💡 Run LLM evaluation first!")
        return None

    print("🤖 LLM EVALUATION PERFORMANCE ANALYSIS")
    print("=" * 50)

    result_files = sorted(list(results_path.glob("llm_results_*.json")))

    if not result_files:
        print("❌ No LLM result files found!")
        return None

    all_results = []
    for result_file in result_files:
        data = load_json_safely(result_file)
        if data:
            parsed_info = parse_filename(result_file.stem)
            if parsed_info:
                model, prompt_type, language_pair = parsed_info
                metrics = data.get("metrics", {})
                all_results.append(
                    {
                        "Model": model,
                        "Prompt_Type": prompt_type,
                        "Language_Pair": language_pair,
                        "Sample_Size": data.get("sample_size", 0),
                        "Accuracy": metrics.get("accuracy", 0),
                        "MCC": metrics.get("mcc", 0),
                        "Precision": metrics.get("precision", 0),
                        "Recall": metrics.get("recall", 0),
                        "F1": metrics.get("f1", 0),
                        "File": result_file.name,
                    }
                )

    if not all_results:
        print("❌ No valid LLM results found!")
        return None

    df = pd.DataFrame(all_results)

    print(f"📊 Found {len(all_results)} LLM evaluation results:\n")

    print("🎯 INDIVIDUAL MODEL PERFORMANCE:")
    print("-" * 80)
    df_sorted = df.sort_values(["Language_Pair", "Model", "Prompt_Type"])
    for _, row in df_sorted.iterrows():
        status = get_performance_status(row["MCC"])
        print(
            f"{status} | {row['Model']:>8} {row['Prompt_Type']:>9} | "
            f"{row['Language_Pair']:>5} | MCC: {row['MCC']:.3f} | "
            f"Acc: {row['Accuracy']:.3f} | F1: {row['F1']:.3f}"
        )

    print("\n🏆 BEST MODELS BY LANGUAGE PAIR:")
    print("-" * 40)
    for lang_pair in sorted(df["Language_Pair"].unique()):
        lang_df = df[df["Language_Pair"] == lang_pair]
        best_row = lang_df.loc[lang_df["MCC"].idxmax()]
        status = get_performance_status(best_row["MCC"])
        print(
            f"{status} {lang_pair:>5}: {best_row['Model']:>8} "
            f"{best_row['Prompt_Type']:>9} | MCC: {best_row['MCC']:.3f}"
        )

    print("\n📈 MODEL COMPARISON:")
    print("-" * 25)
    model_comparison = (
        df.groupby(["Model", "Prompt_Type"])
        .agg(
            MCC_Mean=("MCC", "mean"),
            MCC_Std=("MCC", "std"),
            Acc_Mean=("Accuracy", "mean"),
            F1_Mean=("F1", "mean"),
        )
        .round(3)
        .sort_values("MCC_Mean", ascending=False)
    )

    for (model, prompt), row in model_comparison.iterrows():
        status = get_performance_status(row["MCC_Mean"])
        print(
            f"{status} {model:>8} {prompt:>9} | "
            f"MCC: {row['MCC_Mean']:.3f}±{row['MCC_Std']:.3f} | "
            f"Acc: {row['Acc_Mean']:.3f} | F1: {row['F1_Mean']:.3f}"
        )

    print("\n🌍 LANGUAGE DIFFICULTY RANKING:")
    print("-" * 35)
    lang_analysis = (
        df.groupby("Language_Pair")
        .agg(
            MCC_Mean=("MCC", "mean"),
            MCC_Std=("MCC", "std"),
            Acc_Mean=("Accuracy", "mean"),
        )
        .round(3)
        .sort_values("MCC_Mean", ascending=False)
    )

    for lang_pair, row in lang_analysis.iterrows():
        difficulty = get_difficulty_level(row["MCC_Mean"])
        print(
            f"{difficulty} {lang_pair:>5}: "
            f"MCC {row['MCC_Mean']:.3f}±{row['MCC_Std']:.3f} | "
            f"Acc {row['Acc_Mean']:.3f}"
        )
    return df


def analyze_summary_files(results_dir: str = "results/llm_evaluation"):
    results_path = Path(results_dir)
    summary_files = sorted(list(results_path.glob("llm_evaluation_summary_*.json")))

    if not summary_files:
        return

    print("\n📋 SUMMARY ANALYSIS:")
    print("-" * 20)

    for summary_file in summary_files:
        data = load_json_safely(summary_file)
        if data and isinstance(data, list):
            lang_pair = summary_file.stem.replace("llm_evaluation_summary_", "")
            print(f"\n📊 {lang_pair.upper()} Summary:")

            for result in data:
                model = result.get("model_type", "unknown")
                prompt = result.get("prompt_type", "unknown")
                metrics = result.get("metrics", {})
                mcc = metrics.get("mcc", 0)
                accuracy = metrics.get("accuracy", 0)
                status = get_performance_status(mcc)
                print(
                    f"\t{status} {model:>8} {prompt:>9}: "
                    f"MCC {mcc:.3f}, Acc {accuracy:.3f}"
                )


def save_final_results(df: pd.DataFrame, output_dir: str = "results/analysis/llm"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 SAVING RESULTS TO: {output_dir}")
    print("-" * 30)

    summary_stats = {
        "total_evaluations": len(df),
        "models": sorted(df["Model"].unique().tolist()),
        "prompt_types": sorted(df["Prompt_Type"].unique().tolist()),
        "language_pairs": sorted(df["Language_Pair"].unique().tolist()),
        "best_overall": {
            "model": df.loc[df["MCC"].idxmax(), "Model"],
            "prompt_type": df.loc[df["MCC"].idxmax(), "Prompt_Type"],
            "language_pair": df.loc[df["MCC"].idxmax(), "Language_Pair"],
            "mcc": df["MCC"].max(),
            "accuracy": df.loc[df["MCC"].idxmax(), "Accuracy"],
        },
        "average_metrics": {
            "mcc": df["MCC"].mean(),
            "accuracy": df["Accuracy"].mean(),
            "precision": df["Precision"].mean(),
            "recall": df["Recall"].mean(),
            "f1": df["F1"].mean(),
        },
    }

    with open(output_path / "llm_evaluation_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    df.to_csv(output_path / "llm_detailed_results.csv", index=False)

    best_by_lang = {}
    for lang_pair in df["Language_Pair"].unique():
        lang_df = df[df["Language_Pair"] == lang_pair]
        best_row = lang_df.loc[lang_df["MCC"].idxmax()]
        best_by_lang[lang_pair] = {
            "model": best_row["Model"],
            "prompt_type": best_row["Prompt_Type"],
            "mcc": best_row["MCC"],
            "accuracy": best_row["Accuracy"],
            "f1": best_row["F1"],
        }

    with open(output_path / "best_models_by_language.json", "w") as f:
        json.dump(best_by_lang, f, indent=2)

    model_comparison = (
        df.groupby(["Model", "Prompt_Type"])
        .agg(
            {
                "MCC": ["mean", "std", "min", "max"],
                "Accuracy": ["mean", "std"],
                "F1": ["mean", "std"],
            }
        )
        .round(4)
    )
    model_comparison.to_csv(output_path / "model_comparison.csv")

    print(f"✅ Summary saved to: {output_path / 'llm_evaluation_summary.json'}")
    print(f"✅ Detailed results saved to: {output_path / 'llm_detailed_results.csv'}")
    print(f"✅ Best models saved to: {output_path / 'best_models_by_language.json'}")
    print(f"✅ Model comparison saved to: {output_path / 'model_comparison.csv'}")


def interpret_llm_metrics():
    print("\n📚 LLM EVALUATION METRICS GUIDE")
    print("=" * 35)
    print("🎯 MCC (Matthews Correlation Coefficient) - PRIMARY METRIC:")
    print("   🟢 0.7+  : Excellent error detection")
    print("   🟡 0.5-0.7: Good error detection")
    print("   🟠 0.3-0.5: Fair error detection")
    print("   🔴 0.0-0.3: Poor error detection")
    print("   ❌ <0.0  : Worse than random")

    print("\n📊 Other Metrics:")
    print("   • Accuracy: Overall correctness (aim for >0.8)")
    print("   • Precision: Of predicted errors, how many were real? (aim for >0.7)")
    print("   • Recall: Of real errors, how many did we catch? (aim for >0.7)")
    print("   • F1: Balance of precision and recall (aim for >0.7)")

    print("\n🤖 Model Types:")
    print("   • deepseek: DeepSeek-Coder model")
    print("   • llama3: Llama 3 model")

    print("\n📝 Prompt Types:")
    print("   • zero_shot: No examples provided")
    print("   • few_shot: Examples provided in prompt")

    print("\n🌍 Language Pairs:")
    print("   • en-cs: English to Czech")
    print("   • en-de: English to German")
    print("   • en-ja: English to Japanese")
    print("   • en-zh: English to Chinese")


def main():
    print("🤖 LLM PERFORMANCE CHECKER")
    print("=" * 50)

    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/llm_evaluation"
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ LLM evaluation directory '{results_dir}' not found!")
        print("💡 Run LLM evaluation first!")
        return

    df = analyze_llm_results(results_dir)
    if df is not None:
        analyze_summary_files(results_dir)
        save_final_results(df)
        interpret_llm_metrics()

        print("\n🛠️  QUICK COMMANDS:")
        print("   python llm/evaluate_llm.py")
        print("   python scripts/check_llm_performance.py")


if __name__ == "__main__":
    main()
