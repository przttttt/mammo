import argparse
import os
import tempfile
import warnings
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model_utils import (
    DATA,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    load_bundle,
    load_df,
    print_prediction,
    save_bundle,
    suppress_native_stderr,
    validate_row_index,
)

ARTIFACT_DIR = MODELS_DIR / "tensorflow_model"
MODEL = ARTIFACT_DIR / "model.keras"
META = ARTIFACT_DIR / "metadata.pkl"


def require_tensorflow():
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammo-mpl-config"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
        with suppress_native_stderr():
            import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is not installed. Run: pip install -r requirements/requirements-tensorflow.txt") from exc
    return tf


def train():
    tf = require_tensorflow()
    df = load_df()
    X = df.drop(columns=["id", "diagnosis", "target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
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
    with suppress_native_stderr():
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

    with suppress_native_stderr():
        prob = model.predict(X_test_scaled, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    roc_auc = float(round(roc_auc_score(y_test, prob), 4))
    accuracy = float(round(accuracy_score(y_test, pred), 4))

    print("TensorFlow NN holdout ROC-AUC:", roc_auc)
    print("TensorFlow NN holdout accuracy:", accuracy)
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL)
    bundle = {
        "columns": X.columns.tolist(),
        "scaler": scaler,
        "metadata": {
            "artifact": "evaluated_holdout_model",
            "model_name": "TensorFlow NN",
            "dataset": DATA.name,
            "split": {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
            "metrics": {
                "holdout_roc_auc": roc_auc,
                "holdout_accuracy": accuracy,
            },
            "tensorflow_version": tf.__version__,
        },
    }
    save_bundle(META, bundle)
    print("Saved artifact directory to:", ARTIFACT_DIR)
    print("Saved model file to:", MODEL)
    print("Saved metadata file to:", META)
    print("Saved artifact:", bundle["metadata"]["artifact"])


def predict(row_index: int):
    tf = require_tensorflow()
    if not MODEL.exists() or not META.exists():
        raise SystemExit("Model not found. Run: python scripts/tensorflow_cli.py train")

    bundle = load_bundle(META)
    with suppress_native_stderr():
        model = tf.keras.models.load_model(MODEL)
    df = load_df()

    validate_row_index(df, row_index)
    row = df.iloc[row_index]
    X_row = row[bundle["columns"]].to_frame().T
    X_row_scaled = bundle["scaler"].transform(X_row)
    with suppress_native_stderr():
        prob = float(model.predict(X_row_scaled, verbose=0).ravel()[0])
    metadata = bundle["metadata"]
    print_prediction(row_index, row, prob, metadata["artifact"], metadata["split"]["random_state"])


parser = argparse.ArgumentParser(description="TensorFlow CLI for the WDBC dataset")
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("train", help="Train TensorFlow NN on the WDBC dataset and save the model")

predict_parser = sub.add_parser("predict", help="Predict diagnosis for an existing dataset row")
predict_parser.add_argument("--row-index", type=int, required=True, help="Row index from data/wdbc.data")

args = parser.parse_args()

if args.command == "train":
    train()
else:
    predict(args.row_index)
