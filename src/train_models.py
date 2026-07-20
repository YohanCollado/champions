from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "reports/.matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import get_feature_columns, make_preprocessor, time_based_split
from src.utils import FIGURES_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR, ensure_project_directories


def train_and_evaluate(dataset: pd.DataFrame) -> tuple[str, Pipeline, pd.DataFrame]:
    """Train all models, save metrics and figures, and persist the best model."""
    ensure_project_directories()
    x_train, x_test, y_train, y_test = time_based_split(dataset)
    models = _candidate_models()
    metrics: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(x_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics.append(_classification_metrics(name, y_test, predictions))
        fitted_models[name] = pipeline
        _save_classification_report(name, y_test, predictions)
        if name != "Majority Class Baseline":
            _save_confusion_matrix(name, y_test, predictions, pipeline.classes_)

    metrics_df = pd.DataFrame(metrics).sort_values("weighted_f1", ascending=False)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    joblib.dump(
        {
            "model_name": best_model_name,
            "pipeline": best_model,
            "feature_columns": get_feature_columns(dataset),
            "target": "match_result",
        },
        MODELS_DIR / "best_model.joblib",
    )
    _save_feature_importance(best_model_name, best_model, x_train.columns)
    _save_model_summary(best_model_name, metrics_df, dataset)
    return best_model_name, best_model, metrics_df


def _candidate_models() -> dict[str, object]:
    return {
        "Majority Class Baseline": DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }


def _classification_metrics(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _safe_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def _save_classification_report(model_name: str, y_true: pd.Series, y_pred: pd.Series) -> None:
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    path = REPORTS_DIR / f"{_safe_name(model_name)}_classification_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _save_confusion_matrix(
    model_name: str, y_true: pd.Series, y_pred: pd.Series, labels: list[str] | pd.Index
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", xticks_rotation=30)
    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{_safe_name(model_name)}_confusion_matrix.png", dpi=150)
    plt.close()


def _save_feature_importance(model_name: str, pipeline: Pipeline, original_columns: pd.Index) -> None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out(original_columns)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        values = abs(model.coef_).mean(axis=0)
    importances = pd.Series(values, index=feature_names).sort_values(ascending=False).head(15)
    ax = importances.sort_values().plot(kind="barh", color="#9467bd")
    ax.set_title(f"{model_name} Top Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()


def _save_model_summary(best_model_name: str, metrics_df: pd.DataFrame, dataset: pd.DataFrame) -> None:
    summary = {
        "best_model": best_model_name,
        "selection_metric": "weighted_f1 on chronological test set",
        "rows": int(len(dataset)),
        "columns": int(dataset.shape[1]),
        "target": "match_result",
        "metrics": metrics_df.to_dict(orient="records"),
    }
    Path(REPORTS_DIR / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
