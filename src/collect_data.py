from __future__ import annotations

from pathlib import Path

from src.data_preprocessing import load_raw_football_data
from src.utils import RAW_DATA_DIR


def list_included_raw_files(raw_dir: Path = RAW_DATA_DIR) -> list[str]:
    """Return the raw CSV files included with the project submission."""
    load_raw_football_data(raw_dir)
    return [path.name for path in sorted(raw_dir.glob("*.csv"))]


if __name__ == "__main__":
    files = list_included_raw_files()
    print("Included raw data files:")
    for file_name in files:
        print(f"- {file_name}")
