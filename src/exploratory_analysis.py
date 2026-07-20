from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "reports/.matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, REPORTS_DIR, ensure_project_directories


def run_eda(dataset: pd.DataFrame) -> None:
    """Save a compact exploratory analysis summary and charts."""
    ensure_project_directories()
    summary_path = REPORTS_DIR / "eda_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Exploratory Data Analysis Summary\n")
        file.write("=================================\n\n")
        file.write(f"Dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns\n\n")
        file.write("Column names:\n")
        file.write(", ".join(dataset.columns) + "\n\n")
        file.write("Data types:\n")
        file.write(dataset.dtypes.astype(str).to_string() + "\n\n")
        file.write("Missing values:\n")
        file.write(dataset.isna().sum().to_string() + "\n\n")
        file.write("Target distribution:\n")
        file.write(dataset["match_result"].value_counts().to_string() + "\n\n")
        file.write("Summary statistics:\n")
        file.write(dataset.describe(include="all").to_string())

    _plot_target_distribution(dataset)
    _plot_goals_distribution(dataset)
    _plot_home_away_goals(dataset)
    _plot_correlation_heatmap(dataset)


def _plot_target_distribution(dataset: pd.DataFrame) -> None:
    counts = dataset["match_result"].value_counts().reindex(["Home Win", "Draw", "Away Win"])
    ax = counts.plot(kind="bar", color=["#1f77b4", "#7f7f7f", "#d62728"])
    ax.set_title("Target-Class Distribution")
    ax.set_xlabel("Match result")
    ax.set_ylabel("Matches")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close()


def _plot_goals_distribution(dataset: pd.DataFrame) -> None:
    total_goals = dataset["home_goals"] + dataset["away_goals"]
    ax = total_goals.plot(kind="hist", bins=12, color="#2ca02c", edgecolor="white")
    ax.set_title("Total Goals Distribution")
    ax.set_xlabel("Goals per match")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "goals_distribution.png", dpi=150)
    plt.close()


def _plot_home_away_goals(dataset: pd.DataFrame) -> None:
    means = pd.Series(
        {
            "Home goals": dataset["home_goals"].mean(),
            "Away goals": dataset["away_goals"].mean(),
        }
    )
    ax = means.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Average Home vs Away Goals")
    ax.set_ylabel("Average goals")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "home_away_goals.png", dpi=150)
    plt.close()


def _plot_correlation_heatmap(dataset: pd.DataFrame) -> None:
    columns = [
        "home_goals",
        "away_goals",
        "home_win_rate_before",
        "away_win_rate_before",
        "home_goals_scored_last_5",
        "away_goals_scored_last_5",
        "home_points_last_5",
        "away_points_last_5",
    ]
    corr = dataset[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
