"""
Downloads the UCI Pima Indians Diabetes dataset reproducibly.

Source: Jason Brownlee's well-known ML datasets mirror on GitHub. The raw
file has no header row - column names are added here based on the
accompanying .names documentation (768 instances, 8 features + binary
Outcome, all Pima Indian heritage women aged 21+).
"""
import urllib.request
from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.csv"
)
OUTPUT_PATH = Path(__file__).parent.parent / "datasets" / "pima_diabetes.csv"

COLUMN_NAMES = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree", "age", "outcome",
]


def download_diabetes_dataset(force: bool = False) -> Path:
    if OUTPUT_PATH.exists() and not force:
        print(f"Already present: {OUTPUT_PATH} (pass force=True to re-download)")
        return OUTPUT_PATH

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {DATASET_URL} ...")
    raw_path = OUTPUT_PATH.with_suffix(".raw.csv")
    urllib.request.urlretrieve(DATASET_URL, raw_path)

    # Source file has no header - add column names and save with one.
    df = pd.read_csv(raw_path, names=COLUMN_NAMES)
    df.to_csv(OUTPUT_PATH, index=False)
    raw_path.unlink()

    print(f"Saved to {OUTPUT_PATH} ({len(df)} rows)")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_diabetes_dataset()
