# Dataset Description

## Dataset File

- Main dataset: `data/champions_league_dataset.csv`
- Prediction input: `data/prediction_input.csv`
- Raw source files: `data/raw/*.csv`
- Rows in final dataset: 7,156
- Columns in final dataset: 48

## Source

The dataset was built from public CSV files downloaded from football-data.co.uk:

- Website: https://www.football-data.co.uk/
- Data page: https://www.football-data.co.uk/downloadm.php
- Seasons used: 2021-2022, 2022-2023, 2023-2024, 2024-2025
- Leagues used:
  - England Premier League
  - Spain La Liga
  - Italy Serie A
  - Germany Bundesliga
  - France Ligue 1

This is domestic-league historical data for clubs that commonly feed UEFA competitions. It is used as a practical assignment dataset for learning match-result prediction. It is not an official UEFA Champions League 2026-2027 fixture dataset.

## Prediction Target

The target column is `match_result`, derived from football-data.co.uk's `FTR` column.

| Raw value | Final label |
|---|---|
| `H` | `Home Win` |
| `D` | `Draw` |
| `A` | `Away Win` |

This makes the project a supervised multiclass classification problem.

## Important Columns

- `date`: Match date.
- `league`: Human-readable league name.
- `season`: Season code from the raw file name.
- `home_team`, `away_team`: Team names.
- `home_goals`, `away_goals`: Full-time goals. These are used to create the target and EDA charts, but are not used as model features.
- `home_shots`, `away_shots`, cards, fouls, and corners: Match statistics. These are useful for EDA but are not used as model features because they happen during the match.
- `home_*_before`, `away_*_before`: Pre-match rolling features calculated from previous matches only.
- `*_difference`: Difference between home and away pre-match form features.

## Feature Types

Raw columns include match date, league, teams, final score, result code, and match statistics from the source CSV files.

Engineered columns include:

- `match_result`
- `goal_difference`
- `shot_difference`
- `shots_on_target_difference`
- `corner_difference`
- `foul_difference`
- `yellow_card_difference`
- `red_card_difference`
- `home_matches_played_before`
- `away_matches_played_before`
- `home_win_rate_before`
- `away_win_rate_before`
- `home_goals_scored_last_5`
- `away_goals_scored_last_5`
- `home_goals_conceded_last_5`
- `away_goals_conceded_last_5`
- `home_points_last_5`
- `away_points_last_5`
- `home_rest_days`
- `away_rest_days`
- `home_advantage`
- `team_experience_difference`
- `win_rate_before_difference`
- `goals_scored_last_5_difference`
- `goals_conceded_last_5_difference`
- `points_last_5_difference`

## Missing Values

Early-season rolling features are missing when a team has no previous matches in the included data. The model preprocessing pipeline handles these values with median imputation for numeric features and most-frequent imputation for categorical features.

## Encoding and Scaling

- Categorical features are one-hot encoded.
- Numeric features are median-imputed and standardized.
- Preprocessing is fitted only on the training split to avoid data leakage.

## Leakage Prevention

The model does not use final scores, the raw result code, or match statistics from during the match. The model uses only match identity/context and rolling features calculated before each match.

## Limitations

- The included dataset is not a complete Champions League dataset.
- It uses major European domestic leagues as a proxy for club strength and match-outcome learning.
- `data/prediction_input.csv` is historical sample input used to demonstrate the prediction workflow. It should be replaced with real 2026-2027 fixtures when those fixtures are available.
- Predictions should be treated as an educational machine-learning output, not betting advice or official UEFA forecasting.
