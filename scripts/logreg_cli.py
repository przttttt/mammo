import argparse
from typing import Optional
import warnings

from sklearn import __version__ as sklearn_version
from sklearn.metrics import classification_report

from model_utils import (
    ARTIFACT_SCHEMA_VERSION,
    DATA,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    build_logreg_model,
    compute_metrics,
    load_bundle,
    load_df,
    load_input_json,
    print_prediction,
    save_bundle,
    split_features_target,
    split_holdout,
    tune_sklearn_threshold,
    validate_row_index,
)

MODEL = MODELS_DIR / "logreg_model.pkl"


def train():
    df = load_df()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_holdout(X, y)
    threshold, cv_metrics = tune_sklearn_threshold(build_logreg_model, X_train, y_train)

    model = build_logreg_model()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, prob, threshold)
    pred = (prob >= threshold).astype(int)

    print("Logistic Regression holdout ROC-AUC:", metrics["roc_auc"])
    print("Logistic Regression holdout accuracy:", metrics["accuracy"])
    print("Logistic Regression decision threshold:", metrics["threshold"])
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "metadata": {
            "artifact": "evaluated_holdout_model",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_name": "Logistic Regression",
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
            "sklearn_version": sklearn_version,
        },
    }
    save_bundle(MODEL, bundle)
    print("Saved model to:", MODEL)
    print("Saved artifact:", bundle["metadata"]["artifact"])


def predict(row_index: Optional[int], input_json: Optional[str]):
    if not MODEL.exists():
        raise SystemExit("Model not found. Run: python scripts/logreg_cli.py train")

    bundle = load_bundle(MODEL, require_model=True)
    model = bundle["model"]
    metadata = bundle["metadata"]
    threshold = metadata["threshold"]

    row = None
    if row_index is not None:
        df = load_df()
        validate_row_index(df, row_index)
        row = df.iloc[row_index]
        X_row = row[bundle["columns"]].to_frame().T
    else:
        X_row = load_input_json(bundle["columns"], input_json)
    prob = float(model.predict_proba(X_row)[0, 1])
    print_prediction(prob, metadata["artifact"], metadata["split"]["random_state"], threshold, row_index=row_index, row=row)


parser = argparse.ArgumentParser(description="Logistic Regression CLI for the WDBC dataset")
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("train", help="Train Logistic Regression on the WDBC dataset and save the model")

predict_parser = sub.add_parser("predict", help="Predict diagnosis for an existing dataset row")
predict_group = predict_parser.add_mutually_exclusive_group(required=True)
predict_group.add_argument("--row-index", type=int, help="Row index from data/wdbc.data")
predict_group.add_argument("--input-json", type=str, help="JSON object or path to JSON file with feature values")

args = parser.parse_args()

if args.command == "train":
    train()
else:
    predict(args.row_index, args.input_json)
