from __future__ import annotations

from src.data_preprocessing import build_dataset
from src.exploratory_analysis import run_eda
from src.predict import generate_predictions, read_predicted_winner
from src.train_models import train_and_evaluate
from src.utils import ensure_project_directories


def main() -> None:
    """Run the full assignment workflow."""
    ensure_project_directories()
    print("Building cleaned dataset...")
    dataset = build_dataset()
    print(f"Dataset ready: {len(dataset)} matches")

    print("Running exploratory analysis...")
    run_eda(dataset)

    print("Training and evaluating models...")
    best_model_name, _, metrics = train_and_evaluate(dataset)
    print(metrics.to_string(index=False))
    print(f"Best model: {best_model_name}")

    print("Generating prediction output...")
    predictions = generate_predictions()
    print(f"Saved {len(predictions)} predictions.")
    print(f"Predicted winner/team with highest projected strength: {read_predicted_winner()}")
    print("Workflow complete. See models/, reports/, predictions/, and data/.")


if __name__ == "__main__":
    main()
