from __future__ import annotations

import pandas as pd

from src.train_models import train_and_evaluate
from src.utils import DATA_DIR, require_file


def evaluate_saved_dataset() -> pd.DataFrame:
    """Re-train and evaluate models on the included dataset."""
    dataset_path = DATA_DIR / "champions_league_dataset.csv"
    require_file(dataset_path)
    dataset = pd.read_csv(dataset_path, parse_dates=["date"])
    _, _, metrics = train_and_evaluate(dataset)
    return metrics


if __name__ == "__main__":
    metrics_df = evaluate_saved_dataset()
    print(metrics_df.to_string(index=False))
