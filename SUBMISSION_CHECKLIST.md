# Submission Checklist

## Assignment Requirements

- [x] All Python source code is included in `src/` and `main.py`.
- [x] The complete dataset used by the project is included in `data/champions_league_dataset.csv`.
- [x] Raw source CSV files are included in `data/raw/`.
- [x] A README file explains installation, usage, outputs, results, and troubleshooting.
- [x] Additional files required for the code to work are included.
- [x] A trained model is included at `models/best_model.joblib`.
- [x] Evaluation metrics are included at `reports/model_metrics.csv`.
- [x] Charts are included in `reports/figures/`.
- [x] Prediction output is included at `predictions/champions_league_2026_2027_predictions.csv`.
- [x] Projected winner output is included at `predictions/predicted_winner.txt`.
- [x] Team strength ranking is included at `predictions/team_strength_ranking.csv`.
- [x] Tests are included in `tests/`.
- [x] `python main.py` was run successfully.
- [x] `python -m pytest -q` was run successfully.
- [x] No API keys are required.
- [x] `.env` is excluded.
- [x] `.git/` is excluded.
- [x] Virtual environments are excluded.
- [x] Python cache folders are excluded.
- [x] Final ZIP file is named `YohanCollado.zip`.

## Verification Results

- `python3 main.py`: completed successfully.
- `python3 -m pytest -q`: `4 passed`.
- Best model: Logistic Regression.
- Best weighted F1: 0.458002.
- Final dataset size: 7,156 rows x 48 columns.
- ZIP inspection confirmed no `.env`, `.git`, `venv`, `.venv`, `__pycache__`, `.pytest_cache`, `.DS_Store`, or Matplotlib cache files.

## ZIP Contents

The final archive contains:

- `YohanCollado/PROJECT_PLAN.md`
- `YohanCollado/requirements.txt`
- `YohanCollado/DATASET_DESCRIPTION.md`
- `YohanCollado/SUBMISSION_CHECKLIST.md`
- `YohanCollado/README.md`
- `YohanCollado/.gitignore`
- `YohanCollado/pytest.ini`
- `YohanCollado/main.py`
- `YohanCollado/tests/test_data_preprocessing.py`
- `YohanCollado/models/best_model.joblib`
- `YohanCollado/docs/data_source.md`
- `YohanCollado/data/champions_league_dataset.csv`
- `YohanCollado/data/prediction_input.csv`
- `YohanCollado/data/processed/cleaned_matches.csv`
- `YohanCollado/data/raw/2122_d1.csv`
- `YohanCollado/data/raw/2122_e0.csv`
- `YohanCollado/data/raw/2122_f1.csv`
- `YohanCollado/data/raw/2122_i1.csv`
- `YohanCollado/data/raw/2122_sp1.csv`
- `YohanCollado/data/raw/2223_d1.csv`
- `YohanCollado/data/raw/2223_e0.csv`
- `YohanCollado/data/raw/2223_f1.csv`
- `YohanCollado/data/raw/2223_i1.csv`
- `YohanCollado/data/raw/2223_sp1.csv`
- `YohanCollado/data/raw/2324_d1.csv`
- `YohanCollado/data/raw/2324_e0.csv`
- `YohanCollado/data/raw/2324_f1.csv`
- `YohanCollado/data/raw/2324_i1.csv`
- `YohanCollado/data/raw/2324_sp1.csv`
- `YohanCollado/data/raw/2425_d1.csv`
- `YohanCollado/data/raw/2425_e0.csv`
- `YohanCollado/data/raw/2425_f1.csv`
- `YohanCollado/data/raw/2425_i1.csv`
- `YohanCollado/data/raw/2425_sp1.csv`
- `YohanCollado/predictions/champions_league_2026_2027_predictions.csv`
- `YohanCollado/predictions/predicted_winner.txt`
- `YohanCollado/predictions/team_strength_ranking.csv`
- `YohanCollado/reports/model_metrics.csv`
- `YohanCollado/reports/model_summary.json`
- `YohanCollado/reports/eda_summary.txt`
- `YohanCollado/reports/*_classification_report.json`
- `YohanCollado/reports/figures/*.png`
- `YohanCollado/src/__init__.py`
- `YohanCollado/src/collect_data.py`
- `YohanCollado/src/data_preprocessing.py`
- `YohanCollado/src/evaluate_models.py`
- `YohanCollado/src/exploratory_analysis.py`
- `YohanCollado/src/features.py`
- `YohanCollado/src/predict.py`
- `YohanCollado/src/train_models.py`
- `YohanCollado/src/utils.py`
