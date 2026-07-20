# UEFA Champions League 2026-2027 Prediction Model

## Project Description

This project is a university machine-learning assignment that predicts football match outcomes using historical European club data. The selected prediction target is:

- `Home Win`
- `Draw`
- `Away Win`

This is a supervised multiclass classification problem. The project trains and compares a majority-class baseline, Logistic Regression, Random Forest, and Gradient Boosting. The best model is selected using weighted F1 score on a chronological test split.

The current best model is **Logistic Regression**.

## Project Files

- `main.py`: Runs the full workflow.
- `src/data_preprocessing.py`: Loads, cleans, engineers features, and builds the dataset.
- `src/exploratory_analysis.py`: Creates EDA summaries and charts.
- `src/train_models.py`: Trains and evaluates all models.
- `src/evaluate_models.py`: Re-runs model evaluation.
- `src/predict.py`: Loads the saved model and writes predictions.
- `data/champions_league_dataset.csv`: Final dataset used by the project.
- `data/prediction_input.csv`: Example prediction input.
- `models/best_model.joblib`: Saved best model.
- `reports/model_metrics.csv`: Model comparison table.
- `reports/figures/`: Saved charts.
- `predictions/champions_league_2026_2027_predictions.csv`: Prediction output.
- `predictions/team_strength_ranking.csv`: Team ranking based on prediction probabilities.
- `predictions/predicted_winner.txt`: One-line projected winner/team with highest predicted strength.
- `DATASET_DESCRIPTION.md`: Detailed dataset documentation.
- `PROJECT_PLAN.md`: Implementation plan.
- `SUBMISSION_CHECKLIST.md`: Final submission checklist.

## Dataset

The dataset was built from public CSV files from football-data.co.uk:

- https://www.football-data.co.uk/
- https://www.football-data.co.uk/downloadm.php

The included data covers the 2021-2022 through 2024-2025 seasons for:

- England Premier League
- Spain La Liga
- Italy Serie A
- Germany Bundesliga
- France Ligue 1

The final dataset has **7,156 rows** and **48 columns**.

The target variable is `match_result`, derived from the raw `FTR` column:

- `H` -> `Home Win`
- `D` -> `Draw`
- `A` -> `Away Win`

The main model features are pre-match rolling form features, including win rate before the match, goals scored in the previous five matches, goals conceded in the previous five matches, points from the previous five matches, rest days, home advantage, and home-away differences.

Final score columns and in-match statistics are kept for documentation and EDA, but they are excluded from training to prevent target leakage.

## Installation

Python 3.11 or newer is recommended. This project was verified with Python 3.13.

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

Run the full workflow:

```bash
python main.py
```

This command:

1. Loads the included raw CSV files.
2. Builds `data/champions_league_dataset.csv`.
3. Runs exploratory analysis.
4. Trains and evaluates all models.
5. Saves the best model.
6. Generates predictions.
7. Saves metrics, charts, and prediction outputs.

Individual scripts can also be run:

```bash
python -m src.collect_data
python -m src.evaluate_models
python -m src.predict
```

## Output Files

- Trained model: `models/best_model.joblib`
- Metrics table: `reports/model_metrics.csv`
- Classification reports: `reports/*_classification_report.json`
- EDA summary: `reports/eda_summary.txt`
- Figures: `reports/figures/`
- Predictions: `predictions/champions_league_2026_2027_predictions.csv`
- Projected winner: `predictions/predicted_winner.txt`
- Team ranking: `predictions/team_strength_ranking.csv`

## Model Evaluation

The models were evaluated using a chronological split: earlier matches were used for training and the most recent 20% were used for testing.

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| Logistic Regression | 0.481844 | 0.458002 |
| Gradient Boosting | 0.486034 | 0.449660 |
| Random Forest | 0.508380 | 0.434384 |
| Majority Class Baseline | 0.418994 | 0.247438 |

Logistic Regression was selected because it had the highest weighted F1 score on the chronological test set.

## Testing

Run tests with:

```bash
python -m pytest -q
```

Current verification result:

```text
4 passed
```

## Known Limitations

- The included dataset is based on major European domestic leagues, not a complete Champions League-only dataset.
- The prediction input file is historical sample input used to demonstrate the workflow. It does not contain official UEFA Champions League 2026-2027 fixtures.
