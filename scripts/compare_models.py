import os
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_utils import RANDOM_STATE, TEST_SIZE, load_df, suppress_native_stderr


def evaluate(name, model, X_train, X_test, y_train, y_test):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
    }


def evaluate_tensorflow(X_train, X_test, y_train, y_test):
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mammo-mpl-config"))
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")
        with suppress_native_stderr():
            import tensorflow as tf
    except ImportError:
        return None, "TensorFlow not installed. Install requirements/requirements-tensorflow.txt to include TensorFlow NN in the comparison."

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
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
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    with suppress_native_stderr():
        prob = model.predict(X_test, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return {
        "model": "TensorFlow NN",
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
    }, None


def main():
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

    rows = [
        evaluate(
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE, solver="liblinear")),
                ]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        ),
        evaluate(
            "SVM",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
                ]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        ),
    ]

    tensorflow_row, tensorflow_note = evaluate_tensorflow(X_train_scaled, X_test_scaled, y_train, y_test)
    if tensorflow_row is not None:
        rows.append(tensorflow_row)

    summary = pd.DataFrame(rows).sort_values(["roc_auc", "accuracy", "f1"], ascending=False).reset_index(drop=True)
    print(summary.to_string(index=False))
    if tensorflow_note:
        print("\nNote:", tensorflow_note)

    output = Path(__file__).resolve().parents[1] / "models" / "comparison_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print("\nSaved comparison summary to:", output)


if __name__ == "__main__":
    main()
