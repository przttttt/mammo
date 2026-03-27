import json
import os
import pickle
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wdbc.data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

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


def split_features_target(df):
    X = df.drop(columns=["id", "diagnosis", "target"])
    y = df["target"]
    return X, y


def split_holdout(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_logreg_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, solver="liblinear")),
        ]
    )


def build_svm_model():
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])


def require_tensorflow():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mammo-mpl-config"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
        with suppress_native_stderr():
            import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is not installed. Run: pip install -r requirements/requirements-tensorflow.txt") from exc
    return tf


def build_tensorflow_model(tf, input_dim: int):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_bundle(path: Path, bundle):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(bundle))


def load_bundle(path: Path):
    return pickle.loads(path.read_bytes())


def validate_row_index(df, row_index: int):
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(f"Row index must be between 0 and {len(df) - 1}")


def print_prediction(probability: float, artifact: str, seed: int, threshold: float, row_index: Optional[int] = None, row=None):
    pred = "Malignant" if probability >= threshold else "Benign"

    if row_index is not None:
        print("Row index:", row_index)
    if row is not None and "id" in row:
        print("Patient ID:", row["id"])
    if row is not None and "target" in row:
        actual = "Malignant" if row["target"] == 1 else "Benign"
        print("Actual diagnosis:", actual)
    print("Predicted diagnosis:", pred)
    print("Malignant probability:", round(probability, 4))
    print("Decision threshold:", round(threshold, 4))
    print("Model artifact:", artifact)
    print("Training split seed:", seed)


def load_input_json(columns, input_json: str):
    path = Path(input_json).expanduser()
    payload = path.read_text(encoding="utf-8") if path.exists() else input_json
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit("Input must be a JSON object or a path to a JSON file.") from exc
    if not isinstance(data, dict):
        raise SystemExit("Input JSON must be a key-value object.")

    missing = [column for column in columns if column not in data]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"Input JSON is missing required features. Missing: {preview}")

    values = {}
    for column in columns:
        try:
            values[column] = float(data[column])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Feature '{column}' must be numeric.") from exc
    return pd.DataFrame([values])


def tune_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best_index = int(np.nanargmax(scores))
    return float(round(float(thresholds[best_index]), 4))


def compute_metrics(y_true, probabilities, threshold: float):
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
    }


def summarize_cv_metrics(fold_metrics):
    summary = {"folds": len(fold_metrics)}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier"):
        values = np.asarray([fold[key] for fold in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = round(float(values.mean()), 4)
        summary[f"{key}_std"] = round(float(values.std(ddof=0)), 4)
    return summary


def tune_sklearn_threshold(builder, X_train, y_train):
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    out_of_fold = np.zeros(len(X_train), dtype=float)
    fold_metrics = []
    for train_idx, valid_idx in splitter.split(X_train, y_train):
        model = builder()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_prob = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
        out_of_fold[valid_idx] = fold_prob
        fold_metrics.append(compute_metrics(y_train.iloc[valid_idx], fold_prob, 0.5))

    threshold = tune_threshold(y_train, out_of_fold)
    return threshold, summarize_cv_metrics(fold_metrics)


def tune_tensorflow_threshold(X_train, y_train, epochs: int = 20, batch_size: int = 32):
    tf = require_tensorflow()
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    out_of_fold = np.zeros(len(X_train), dtype=float)
    fold_metrics = []
    for train_idx, valid_idx in splitter.split(X_train, y_train):
        scaler = StandardScaler()
        X_fold_train = scaler.fit_transform(X_train.iloc[train_idx])
        X_fold_valid = scaler.transform(X_train.iloc[valid_idx])
        model = build_tensorflow_model(tf, X_fold_train.shape[1])
        with suppress_native_stderr():
            model.fit(X_fold_train, y_train.iloc[train_idx], epochs=epochs, batch_size=batch_size, verbose=0)
            fold_prob = model.predict(X_fold_valid, verbose=0).ravel()
        out_of_fold[valid_idx] = fold_prob
        fold_metrics.append(compute_metrics(y_train.iloc[valid_idx], fold_prob, 0.5))

    threshold = tune_threshold(y_train, out_of_fold)
    return threshold, summarize_cv_metrics(fold_metrics), tf.__version__


def calibration_summary(y_true, probabilities, bins: int = 5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
                "count": int(mask.sum()),
                "predicted_mean": round(float(probabilities[mask].mean()), 4),
                "actual_rate": round(float(y_true[mask].mean()), 4),
            }
        )
    return rows


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
