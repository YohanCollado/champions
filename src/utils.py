from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"


RANDOM_SEED = 42


def ensure_project_directories() -> None:
    """Create output folders used by the project."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a helpful error when a required project file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. Run `python main.py` from the project root."
        )
