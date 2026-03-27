import json
import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler

from model_utils import (
    RESULTS_DIR,
    build_logreg_model,
    build_svm_model,
    build_tensorflow_model,
    calibration_summary,
    compute_metrics,
    load_df,
    require_tensorflow,
    split_features_target,
    split_holdout,
    suppress_native_stderr,
    tune_sklearn_threshold,
    tune_tensorflow_threshold,
)


def evaluate_sklearn(name, builder, X_train, X_test, y_train, y_test):
    threshold, cv_metrics = tune_sklearn_threshold(builder, X_train, y_train)
    model = builder()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, probabilities, threshold)
    return {
        "model": name,
        **metrics,
        "cv_accuracy": cv_metrics["accuracy_mean"],
        "cv_f1": cv_metrics["f1_mean"],
        "cv_roc_auc": cv_metrics["roc_auc_mean"],
        "calibration_bins": calibration_summary(y_test, probabilities),
        "cv_metrics": cv_metrics,
    }


def evaluate_tensorflow(X_train, X_test, y_train, y_test):
    if os.environ.get("MAMMO_SKIP_TENSORFLOW") == "1":
        return None, "TensorFlow skipped because MAMMO_SKIP_TENSORFLOW=1."

    tf = require_tensorflow()
    threshold, cv_metrics, tensorflow_version = tune_tensorflow_threshold(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = build_tensorflow_model(tf, X_train_scaled.shape[1])
    with suppress_native_stderr():
        model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)
        probabilities = model.predict(X_test_scaled, verbose=0).ravel()

    metrics = compute_metrics(y_test, probabilities, threshold)
    return {
        "model": "TensorFlow NN",
        **metrics,
        "cv_accuracy": cv_metrics["accuracy_mean"],
        "cv_f1": cv_metrics["f1_mean"],
        "cv_roc_auc": cv_metrics["roc_auc_mean"],
        "calibration_bins": calibration_summary(y_test, probabilities),
        "cv_metrics": cv_metrics,
        "tensorflow_version": tensorflow_version,
    }, None


def main():
    df = load_df()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_holdout(X, y)

    rows = [
        evaluate_sklearn("Logistic Regression", build_logreg_model, X_train, X_test, y_train, y_test),
        evaluate_sklearn("SVM", build_svm_model, X_train, X_test, y_train, y_test),
    ]

    tensorflow_row, note = evaluate_tensorflow(X_train, X_test, y_train, y_test)
    if tensorflow_row is not None:
        rows.append(tensorflow_row)

    summary = pd.DataFrame(
        [
            {
                "model": row["model"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "brier": row["brier"],
                "threshold": row["threshold"],
                "cv_roc_auc": row["cv_roc_auc"],
            }
            for row in rows
        ]
    ).sort_values(["roc_auc", "accuracy", "f1"], ascending=False).reset_index(drop=True)

    print(summary.to_string(index=False))
    if note:
        print("\nNote:", note)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "comparison_summary.csv"
    json_path = RESULTS_DIR / "comparison_report.json"
    summary.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nSaved comparison summary to:", csv_path)
    print("Saved comparison report to:", json_path)


if __name__ == "__main__":
    main()
