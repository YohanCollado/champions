# Project Plan

## Current Repository Review

- The repository already had a basic folder structure with `data/`, `docs/`, `notebooks/`, `src/`, `README.md`, `requirements.txt`, and `main.py`.
- `README.md`, `requirements.txt`, `main.py`, and several source modules were empty or incomplete.
- `docs/data_source.md` listed possible features but did not identify real data sources.
- `src/features.py` contained a useful early feature-name list, but it had spelling issues and no working feature engineering code.
- No dataset was present in `data/`, so a real downloadable football dataset must be included for the final submission.

## Target Variable

- The selected prediction target is `match_result`.
- It is derived from football-data.co.uk's `FTR` column:
  - `H` means Home Win
  - `D` means Draw
  - `A` means Away Win

## Usable Features

- Raw match context: league, season, home team, away team.
- Raw match statistics available before target creation are used carefully.
- Engineered features include goal difference, shot difference, shots-on-target difference, corner difference, foul difference, card difference, and rolling team form.
- Direct score columns and the raw `FTR` column are removed from model features to avoid target leakage.

## Work To Complete

1. Download and include public historical football CSV data.
2. Build a cleaned dataset at `data/champions_league_dataset.csv`.
3. Create reproducible preprocessing, EDA, model training, evaluation, and prediction scripts.
4. Train Logistic Regression, Random Forest, Gradient Boosting, and a majority-class baseline.
5. Save metrics, figures, predictions, and the best trained model.
6. Write `DATASET_DESCRIPTION.md`, update `README.md`, and add tests.
7. Run the project end to end.
8. Package the final submission as `YohanCollado.zip`.
