import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "results" / "comparison_report.json"

SNAPSHOT_START = "<!-- snapshot-table:start -->"
SNAPSHOT_END = "<!-- snapshot-table:end -->"
COMPARE_START = "<!-- compare-table:start -->"
COMPARE_END = "<!-- compare-table:end -->"

ARTIFACTS = {
    "Logistic Regression": ("models/logreg_model.pkl", "strongest lightweight baseline"),
    "SVM": ("models/svm_model.pkl", "margin-based baseline"),
    "TensorFlow NN": ("models/tensorflow_model/", "neural-network baseline"),
}


def format_snapshot(rows):
    lines = [
        "| Model | Holdout Snapshot | Artifact | Position |",
        "|---|---|---|---|",
    ]
    for row in rows:
        artifact, position = ARTIFACTS.get(row["model"], ("generated artifact", "deployable candidate"))
        lines.append(
            f'| {row["model"]} | `{row["accuracy"]:.4f} acc` / `{row["roc_auc"]:.4f} auc` | `{artifact}` | {position} |'
        )
    return "\n".join(lines)


def format_compare(rows):
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["model"]} | {row["accuracy"]:.4f} | {row["precision"]:.4f} | {row["recall"]:.4f} | {row["f1"]:.4f} | {row["roc_auc"]:.4f} |'
        )
    return "\n".join(lines)


def replace_block(text, start_marker, end_marker, replacement):
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    return text[:start] + "\n" + replacement + "\n" + text[end:]


def main():
    if not REPORT.exists():
        raise SystemExit("Comparison report not found. Run: python scripts/compare_models.py")

    rows = json.loads(REPORT.read_text(encoding="utf-8"))
    rows.sort(key=lambda row: (-row["roc_auc"], -row["accuracy"], -row["f1"]))

    readme = README.read_text(encoding="utf-8")
    readme = replace_block(readme, SNAPSHOT_START, SNAPSHOT_END, format_snapshot(rows))
    readme = replace_block(readme, COMPARE_START, COMPARE_END, format_compare(rows))
    README.write_text(readme, encoding="utf-8")
    print("Updated README metrics from:", REPORT)


if __name__ == "__main__":
    main()
