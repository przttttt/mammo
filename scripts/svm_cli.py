import argparse

from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from model_utils import (
    DATA,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    load_bundle,
    load_df,
    print_prediction,
    save_bundle,
    validate_row_index,
)

MODEL = MODELS_DIR / "svm_model.pkl"


def train():
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
    model = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    roc_auc = float(round(roc_auc_score(y_test, prob), 4))
    accuracy = float(round(accuracy_score(y_test, pred), 4))

    print("SVM holdout ROC-AUC:", roc_auc)
    print("SVM holdout accuracy:", accuracy)
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))

    bundle = {
        "model": model,
        "columns": X.columns.tolist(),
        "metadata": {
            "artifact": "evaluated_holdout_model",
            "model_name": "SVM (RBF kernel)",
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
            "sklearn_version": sklearn_version,
        },
    }
    save_bundle(MODEL, bundle)
    print("Saved model to:", MODEL)
    print("Saved artifact:", bundle["metadata"]["artifact"])


def predict(row_index: int):
    if not MODEL.exists():
        raise SystemExit("Model not found. Run: python scripts/svm_cli.py train")

    bundle = load_bundle(MODEL)
    model = bundle["model"]
    df = load_df()

    validate_row_index(df, row_index)
    row = df.iloc[row_index]
    X_row = row[bundle["columns"]].to_frame().T
    prob = float(model.predict_proba(X_row)[0, 1])
    metadata = bundle["metadata"]
    print_prediction(row_index, row, prob, metadata["artifact"], metadata["split"]["random_state"])


parser = argparse.ArgumentParser(description="Minimal SVM CLI for the WDBC dataset")
sub = parser.add_subparsers(dest="command", required=True)
sub.add_parser("train", help="Train SVM on the WDBC dataset and save the model")

predict_parser = sub.add_parser("predict", help="Predict diagnosis for an existing dataset row")
predict_parser.add_argument("--row-index", type=int, required=True, help="Row index from data/wdbc.data")

args = parser.parse_args()

if args.command == "train":
    train()
else:
    predict(args.row_index)
