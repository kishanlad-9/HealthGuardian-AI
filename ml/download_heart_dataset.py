"""
Downloads the UCI Heart Disease (Cleveland) dataset reproducibly, instead
of requiring datasets/heart_disease_uci.csv to be committed to git (it
isn't - see datasets/README.md and .gitignore) or manually placed.

Source: a community mirror of the UCI Cleveland processed dataset,
303 patients x 13 clinical features + target. See datasets/README.md
for a known data-quality caveat in this specific mirror (ca=4, thal=0).
"""
import urllib.request
from pathlib import Path

DATASET_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)
OUTPUT_PATH = Path(__file__).parent.parent / "datasets" / "heart_disease_uci.csv"


def download_heart_dataset(force: bool = False) -> Path:
    """Downloads the dataset if not already present. Set force=True to
    re-download even if the file exists."""
    if OUTPUT_PATH.exists() and not force:
        print(f"Already present: {OUTPUT_PATH} (pass force=True to re-download)")
        return OUTPUT_PATH

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    download_heart_dataset()
