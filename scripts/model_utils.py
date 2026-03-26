import os
import pickle
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wdbc.data"
MODELS_DIR = ROOT / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
COLUMNS = ["id", "diagnosis"] + [f"{name}_{part}" for part in ("mean", "se", "worst") for name in FEATURES]


def load_df():
    df = pd.read_csv(DATA, names=COLUMNS)
    df["target"] = (df["diagnosis"] == "M").astype(int)
    return df


def save_bundle(path: Path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))


def load_bundle(path: Path):
    return pickle.loads(path.read_bytes())


def validate_row_index(df, row_index: int):
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(f"Row index must be between 0 and {len(df) - 1}")


def print_prediction(row_index: int, row, probability: float, artifact: str, seed: int):
    pred = "Malignant" if probability >= 0.5 else "Benign"
    actual = "Malignant" if row["target"] == 1 else "Benign"

    print("Row index:", row_index)
    print("Patient ID:", row["id"])
    print("Actual diagnosis:", actual)
    print("Predicted diagnosis:", pred)
    print("Malignant probability:", round(probability, 4))
    print("Model artifact:", artifact)
    print("Training split seed:", seed)


@contextmanager
def suppress_native_stderr():
    saved_fd = os.dup(2)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        try:
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
