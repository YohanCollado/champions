from __future__ import annotations

import joblib
import pandas as pd

from src.utils import DATA_DIR, MODELS_DIR, PREDICTIONS_DIR, ensure_project_directories, require_file


def generate_predictions() -> pd.DataFrame:
    """Load the saved model and create prediction output."""
    ensure_project_directories()
    model_path = MODELS_DIR / "best_model.joblib"
    input_path = DATA_DIR / "prediction_input.csv"
    require_file(model_path)
    require_file(input_path)

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]
    prediction_input = pd.read_csv(input_path)

    missing_columns = [column for column in feature_columns if column not in prediction_input.columns]
    if missing_columns:
        raise ValueError(f"Prediction input is missing required columns: {missing_columns}")

    features = prediction_input[feature_columns]
    output = prediction_input[["date", "league", "home_team", "away_team"]].copy()
    output["predicted_class"] = pipeline.predict(features)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(features)
        for index, class_name in enumerate(pipeline.classes_):
            output[f"probability_{class_name.lower().replace(' ', '_')}"] = probabilities[:, index]

    output["note"] = (
        "These are workflow demonstration predictions from historical sample input. "
        "They are not official 2026-2027 fixtures."
    )
    output_path = PREDICTIONS_DIR / "champions_league_2026_2027_predictions.csv"
    output.to_csv(output_path, index=False)
    save_team_winner_projection(output)
    return output


def save_team_winner_projection(predictions: pd.DataFrame) -> str:
    """Rank teams by expected points from the prediction file and save the top team."""
    required = {
        "home_team",
        "away_team",
        "probability_home_win",
        "probability_draw",
        "probability_away_win",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Cannot estimate a winner because predictions are missing: {sorted(missing)}")

    team_rows: list[dict[str, float | str]] = []
    for _, row in predictions.iterrows():
        home_expected_points = 3 * row["probability_home_win"] + row["probability_draw"]
        away_expected_points = 3 * row["probability_away_win"] + row["probability_draw"]
        team_rows.append({"team": row["home_team"], "expected_points": home_expected_points})
        team_rows.append({"team": row["away_team"], "expected_points": away_expected_points})

    ranking = (
        pd.DataFrame(team_rows)
        .groupby("team", as_index=False)
        .agg(matches=("expected_points", "size"), total_expected_points=("expected_points", "sum"))
    )
    ranking["average_expected_points"] = ranking["total_expected_points"] / ranking["matches"]
    ranking = ranking.sort_values(
        ["average_expected_points", "total_expected_points", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1

    ranking_path = PREDICTIONS_DIR / "team_strength_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    winner = str(ranking.loc[0, "team"])
    winner_path = PREDICTIONS_DIR / "predicted_winner.txt"
    winner_path.write_text(
        (
            f"Predicted winner/team with highest projected strength: {winner}\n"
            "Note: this is based on the included historical sample prediction input, "
            "not official UEFA Champions League 2026-2027 fixtures.\n"
        ),
        encoding="utf-8",
    )
    return winner


def read_predicted_winner() -> str:
    """Read the saved predicted winner text file."""
    winner_path = PREDICTIONS_DIR / "predicted_winner.txt"
    require_file(winner_path)
    first_line = winner_path.read_text(encoding="utf-8").splitlines()[0]
    return first_line.split(":", maxsplit=1)[1].strip()


if __name__ == "__main__":
    predictions = generate_predictions()
    print(f"Saved {len(predictions)} predictions.")
    print(f"Predicted winner/team with highest projected strength: {read_predicted_winner()}")
