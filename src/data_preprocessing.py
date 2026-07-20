from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_project_directories


LEAGUE_NAMES = {
    "E0": "England Premier League",
    "SP1": "Spain La Liga",
    "I1": "Italy Serie A",
    "D1": "Germany Bundesliga",
    "F1": "France Ligue 1",
}

RESULT_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

BASE_COLUMNS = [
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
]

LEAKAGE_COLUMNS = {
    "home_goals",
    "away_goals",
    "full_time_result_code",
    "match_result",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_fouls",
    "away_fouls",
    "home_corners",
    "away_corners",
    "home_yellow_cards",
    "away_yellow_cards",
    "home_red_cards",
    "away_red_cards",
    "goal_difference",
    "shot_difference",
    "shots_on_target_difference",
    "corner_difference",
    "foul_difference",
    "yellow_card_difference",
    "red_card_difference",
}


@dataclass
class TeamHistory:
    goals_for: list[int] = field(default_factory=list)
    goals_against: list[int] = field(default_factory=list)
    points: list[int] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    wins: int = 0
    matches: int = 0

    def features(self, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
        last_goals_for = self.goals_for[-5:]
        last_goals_against = self.goals_against[-5:]
        last_points = self.points[-5:]
        rest_days = np.nan
        if self.dates:
            rest_days = float((match_date - self.dates[-1]).days)
        return {
            f"{prefix}_matches_played_before": float(self.matches),
            f"{prefix}_win_rate_before": self.wins / self.matches if self.matches else np.nan,
            f"{prefix}_goals_scored_last_5": float(np.mean(last_goals_for)) if last_goals_for else np.nan,
            f"{prefix}_goals_conceded_last_5": float(np.mean(last_goals_against))
            if last_goals_against
            else np.nan,
            f"{prefix}_points_last_5": float(np.mean(last_points)) if last_points else np.nan,
            f"{prefix}_rest_days": rest_days,
        }

    def update(self, goals_for: int, goals_against: int, points: int, date: pd.Timestamp) -> None:
        self.goals_for.append(goals_for)
        self.goals_against.append(goals_against)
        self.points.append(points)
        self.dates.append(date)
        self.wins += int(points == 3)
        self.matches += 1


def load_raw_football_data(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load all downloaded football-data.co.uk CSV files."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No raw CSV files were found in {raw_dir}. The submitted project should include them."
        )

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        missing = [column for column in BASE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        frame = frame[BASE_COLUMNS].copy()
        frame["source_file"] = path.name
        frame["season"] = path.name.split("_")[0]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_matches(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Clean raw football-data.co.uk rows and keep columns useful for the assignment."""
    df = raw_matches.copy()
    df = df.rename(
        columns={
            "Div": "league_code",
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "full_time_result_code",
            "HS": "home_shots",
            "AS": "away_shots",
            "HST": "home_shots_on_target",
            "AST": "away_shots_on_target",
            "HF": "home_fouls",
            "AF": "away_fouls",
            "HC": "home_corners",
            "AC": "away_corners",
            "HY": "home_yellow_cards",
            "AY": "away_yellow_cards",
            "HR": "home_red_cards",
            "AR": "away_red_cards",
        }
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["league"] = df["league_code"].map(LEAGUE_NAMES).fillna(df["league_code"])
    df["match_result"] = df["full_time_result_code"].map(RESULT_LABELS)

    numeric_columns = [
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_fouls",
        "away_fouls",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "match_result"])
    df = df[df["home_goals"].between(0, 20) & df["away_goals"].between(0, 20)]
    df = df.drop_duplicates(subset=["date", "league_code", "home_team", "away_team"], keep="first")

    df["goal_difference"] = df["home_goals"] - df["away_goals"]
    df["shot_difference"] = df["home_shots"] - df["away_shots"]
    df["shots_on_target_difference"] = df["home_shots_on_target"] - df["away_shots_on_target"]
    df["corner_difference"] = df["home_corners"] - df["away_corners"]
    df["foul_difference"] = df["home_fouls"] - df["away_fouls"]
    df["yellow_card_difference"] = df["home_yellow_cards"] - df["away_yellow_cards"]
    df["red_card_difference"] = df["home_red_cards"] - df["away_red_cards"]
    return df.sort_values(["date", "league_code", "home_team", "away_team"]).reset_index(drop=True)


def add_rolling_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Add pre-match team form features without using the current match result."""
    histories: dict[tuple[str, str], TeamHistory] = {}
    rows: list[dict[str, float]] = []

    for _, row in matches.iterrows():
        league_code = str(row["league_code"])
        home_key = (league_code, str(row["home_team"]))
        away_key = (league_code, str(row["away_team"]))
        home_history = histories.setdefault(home_key, TeamHistory())
        away_history = histories.setdefault(away_key, TeamHistory())

        feature_row = {}
        feature_row.update(home_history.features(row["date"], "home"))
        feature_row.update(away_history.features(row["date"], "away"))
        feature_row["home_advantage"] = 1.0
        feature_row["team_experience_difference"] = (
            feature_row["home_matches_played_before"] - feature_row["away_matches_played_before"]
        )
        for metric in ["win_rate_before", "goals_scored_last_5", "goals_conceded_last_5", "points_last_5"]:
            feature_row[f"{metric}_difference"] = feature_row[f"home_{metric}"] - feature_row[f"away_{metric}"]
        rows.append(feature_row)

        home_points, away_points = _points_for_result(str(row["full_time_result_code"]))
        home_history.update(int(row["home_goals"]), int(row["away_goals"]), home_points, row["date"])
        away_history.update(int(row["away_goals"]), int(row["home_goals"]), away_points, row["date"])

    rolling = pd.DataFrame(rows)
    return pd.concat([matches.reset_index(drop=True), rolling], axis=1)


def _points_for_result(result_code: str) -> tuple[int, int]:
    if result_code == "H":
        return 3, 0
    if result_code == "A":
        return 0, 3
    return 1, 1


def get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return columns used for model training."""
    excluded = LEAKAGE_COLUMNS | {"date", "source_file", "league_code"}
    return [column for column in dataset.columns if column not in excluded]


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = dataset[get_feature_columns(dataset)].copy()
    target = dataset["match_result"].copy()
    return features, target


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def time_based_split(
    dataset: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = dataset.sort_values("date").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - test_size))
    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Dataset is too small for the requested train-test split.")
    train_df = ordered.iloc[:split_index].copy()
    test_df = ordered.iloc[split_index:].copy()
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)
    return x_train, x_test, y_train, y_test


def build_dataset() -> pd.DataFrame:
    """Build and save the complete dataset used by the assignment."""
    ensure_project_directories()
    raw = load_raw_football_data()
    cleaned = clean_matches(raw)
    dataset = add_rolling_features(cleaned)
    dataset_path = DATA_DIR / "champions_league_dataset.csv"
    processed_path = PROCESSED_DATA_DIR / "cleaned_matches.csv"
    dataset.to_csv(dataset_path, index=False)
    dataset.to_csv(processed_path, index=False)
    create_prediction_input(dataset)
    return dataset


def create_prediction_input(dataset: pd.DataFrame, rows: int = 24) -> pd.DataFrame:
    """Create a clearly labeled input file for the prediction script."""
    prediction_input = dataset.sort_values("date").tail(rows).copy()
    prediction_input = prediction_input.drop(columns=["match_result", "full_time_result_code"], errors="ignore")
    prediction_input["prediction_note"] = (
        "Historical sample input for demonstrating the 2026-2027 prediction workflow; "
        "replace with actual 2026-2027 fixtures when available."
    )
    path = DATA_DIR / "prediction_input.csv"
    prediction_input.to_csv(path, index=False)
    return prediction_input
