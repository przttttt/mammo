import argparse
from typing import Optional

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from model_utils import (
    ARTIFACT_SCHEMA_VERSION,
    DATA,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_tensorflow_model,
    compute_metrics,
    load_bundle,
    load_df,
    load_input_json,
    print_prediction,
    require_tensorflow,
    save_bundle,
    split_features_target,
    split_holdout,
    suppress_native_stderr,
    tune_tensorflow_threshold,
    validate_row_index,
)

ARTIFACT_DIR = MODELS_DIR / "tensorflow_model"
MODEL = ARTIFACT_DIR / "model.keras"
META = ARTIFACT_DIR / "metadata.pkl"


def train():
    tf = require_tensorflow()
    df = load_df()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_holdout(X, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    threshold, cv_metrics, tensorflow_version = tune_tensorflow_threshold(X_train, y_train)
    model = build_tensorflow_model(tf, X_train_scaled.shape[1])
    with suppress_native_stderr():
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

    with suppress_native_stderr():
        prob = model.predict(X_test_scaled, verbose=0).ravel()
    metrics = compute_metrics(y_test, prob, threshold)
    pred = (prob >= threshold).astype(int)

    print("TensorFlow NN holdout ROC-AUC:", metrics["roc_auc"])
    print("TensorFlow NN holdout accuracy:", metrics["accuracy"])
    print("TensorFlow NN decision threshold:", metrics["threshold"])
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL)
    bundle = {
        "columns": X.columns.tolist(),
        "scaler": scaler,
        "metadata": {
            "artifact": "evaluated_holdout_model",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_name": "TensorFlow NN",
            "dataset": DATA.name,
            "split": {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            },
            "threshold": metrics["threshold"],
            "metrics": metrics,
            "cv_metrics": cv_metrics,
            "tensorflow_version": tensorflow_version,
        },
    }
    save_bundle(META, bundle)
    print("Saved artifact directory to:", ARTIFACT_DIR)
    print("Saved model file to:", MODEL)
    print("Saved metadata file to:", META)
    print("Saved artifact:", bundle["metadata"]["artifact"])


def predict(row_index: Optional[int], input_json: Optional[str]):
    tf = require_tensorflow()
    if not MODEL.exists() or not META.exists():
        raise SystemExit("Model not found. Run: python scripts/tensorflow_cli.py train")

    bundle = load_bundle(META)
    with suppress_native_stderr():
        model = tf.keras.models.load_model(MODEL)
    metadata = bundle["metadata"]
    threshold = metadata.get("threshold", 0.5)

    row = None
    if row_index is not None:
        df = load_df()
        validate_row_index(df, row_index)
        row = df.iloc[row_index]
        X_row = row[bundle["columns"]].to_frame().T
    else:
        X_row = load_input_json(bundle["columns"], input_json)
    X_row_scaled = bundle["scaler"].transform(X_row)
    with suppress_native_stderr():
        prob = float(model.predict(X_row_scaled, verbose=0).ravel()[0])
    print_prediction(prob, metadata["artifact"], metadata["split"]["random_state"], threshold, row_index=row_index, row=row)


parser = argparse.ArgumentParser(description="TensorFlow CLI for the WDBC dataset")
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("train", help="Train TensorFlow NN on the WDBC dataset and save the model")

predict_parser = sub.add_parser("predict", help="Predict diagnosis for an existing dataset row")
predict_group = predict_parser.add_mutually_exclusive_group(required=True)
predict_group.add_argument("--row-index", type=int, help="Row index from data/wdbc.data")
predict_group.add_argument("--input-json", type=str, help="JSON object or path to JSON file with feature values")

args = parser.parse_args()

if args.command == "train":
    train()
else:
    predict(args.row_index, args.input_json)
