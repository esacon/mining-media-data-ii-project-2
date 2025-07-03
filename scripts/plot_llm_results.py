#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_json_safely(file_path: Path) -> Optional[Dict]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_analysis_results(analysis_dir: str = "results/analysis/llm") -> pd.DataFrame:
    analysis_path = Path(analysis_dir)
    if not analysis_path.exists():
        raise FileNotFoundError(
            f"Analysis directory '{analysis_dir}' not found!")

    detailed_results_file = analysis_path / "llm_detailed_results.csv"
    if not detailed_results_file.exists():
        raise FileNotFoundError(
            f"Detailed results file '{detailed_results_file}' not found!")

    df = pd.read_csv(detailed_results_file)

    return df


def setup_plotting_style():
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


def plot_model_comparison_heatmap(df: pd.DataFrame, metric: str = "MCC",
                                  save_path: Optional[str] = None):
    plt.figure(figsize=(12, 8))

    pivot_data = df.pivot_table(
        values=metric,
        index=['Model', 'Prompt_Type'],
        columns='Language_Pair',
        aggfunc='mean'
    )

    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.5 if metric == 'MCC' else 0.8,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': metric}
    )

    plt.title(f'LLM Model Performance Comparison ({metric})')
    plt.xlabel('Language Pair')
    plt.ylabel('Model Configuration')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_prompt_type_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['MCC', 'Accuracy', 'F1', 'Precision']

    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        grouped = df.groupby(['Model', 'Prompt_Type'])[
            metric].mean().reset_index()

        models = grouped['Model'].unique()
        x = np.arange(len(models))
        width = 0.35

        zero_shot_data = grouped[grouped['Prompt_Type'] == 'zero_shot']
        few_shot_data = grouped[grouped['Prompt_Type'] == 'few_shot']

        zero_shot = []
        few_shot = []

        for model in models:
            zs_val = zero_shot_data[zero_shot_data['Model'] == model][metric]
            fs_val = few_shot_data[few_shot_data['Model'] == model][metric]

            zero_shot.append(zs_val.iloc[0] if len(zs_val) > 0 else 0)
            few_shot.append(fs_val.iloc[0] if len(fs_val) > 0 else 0)

        ax.bar(x - width/2, zero_shot, width, label='Zero-shot', alpha=0.8)
        ax.bar(x + width/2, few_shot, width, label='Few-shot', alpha=0.8)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric}: Zero-shot vs Few-shot')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Prompt Type Performance Comparison')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_plots(df: pd.DataFrame, output_dir: str = "results/analysis/llm/plots"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plots = [
        ("model_comparison_heatmap.png", lambda: plot_model_comparison_heatmap(
            df, save_path=output_path / "model_comparison_heatmap.png")),
        ("prompt_type_comparison.png", lambda: plot_prompt_type_comparison(
            df, save_path=output_path / "prompt_type_comparison.png"))
    ]

    for plot_name, plot_func in plots:
        try:
            plot_func()
            plt.close('all')
        except Exception:
            pass


def main():
    analysis_dir = sys.argv[1] if len(sys.argv) > 1 else "results/analysis/llm"

    try:
        df = load_analysis_results(analysis_dir)
        setup_plotting_style()

        if len(sys.argv) > 2 and sys.argv[2] == "--save":
            save_plots(df)
        else:
            plot_model_comparison_heatmap(df)
            plot_prompt_type_comparison(df)

    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()
