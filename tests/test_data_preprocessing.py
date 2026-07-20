from __future__ import annotations

import pandas as pd

from src.data_preprocessing import add_rolling_features, clean_matches, get_feature_columns, time_based_split


def _sample_raw_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Div": ["E0", "E0", "E0", "E0"],
            "Date": ["01/08/2023", "08/08/2023", "15/08/2023", "22/08/2023"],
            "HomeTeam": ["Arsenal", "Chelsea", "Arsenal", "Chelsea"],
            "AwayTeam": ["Chelsea", "Arsenal", "Chelsea", "Arsenal"],
            "FTHG": [2, 1, 0, 3],
            "FTAG": [1, 1, 2, 0],
            "FTR": ["H", "D", "A", "H"],
            "HS": [10, 8, 7, 12],
            "AS": [7, 9, 11, 5],
            "HST": [5, 3, 2, 6],
            "AST": [2, 4, 5, 1],
            "HF": [9, 10, 11, 8],
            "AF": [10, 9, 7, 12],
            "HC": [4, 5, 3, 8],
            "AC": [3, 4, 7, 2],
            "HY": [1, 2, 1, 3],
            "AY": [2, 1, 2, 0],
            "HR": [0, 0, 0, 1],
            "AR": [0, 0, 0, 0],
            "source_file": ["test.csv"] * 4,
            "season": ["2324"] * 4,
        }
    )


def test_clean_matches_creates_target_labels() -> None:
    cleaned = clean_matches(_sample_raw_matches())
    assert cleaned["match_result"].tolist() == ["Home Win", "Draw", "Away Win", "Home Win"]
    assert cleaned["date"].is_monotonic_increasing


def test_rolling_features_shift_before_current_match() -> None:
    dataset = add_rolling_features(clean_matches(_sample_raw_matches()))
    first_arsenal_home = dataset.iloc[0]
    second_arsenal_home = dataset.iloc[2]

    assert pd.isna(first_arsenal_home["home_goals_scored_last_5"])
    assert second_arsenal_home["home_goals_scored_last_5"] == 1.5
    assert second_arsenal_home["home_points_last_5"] == 2.0


def test_feature_columns_exclude_target_and_score_leakage() -> None:
    dataset = add_rolling_features(clean_matches(_sample_raw_matches()))
    features = get_feature_columns(dataset)

    assert "match_result" not in features
    assert "home_goals" not in features
    assert "away_goals" not in features
    assert "full_time_result_code" not in features


def test_time_based_split_keeps_latest_rows_for_test() -> None:
    dataset = add_rolling_features(clean_matches(_sample_raw_matches()))
    x_train, x_test, y_train, y_test = time_based_split(dataset, test_size=0.25)

    assert len(x_train) == 3
    assert len(x_test) == 1
    assert y_test.iloc[0] == "Home Win"
    assert "match_result" not in x_train.columns
