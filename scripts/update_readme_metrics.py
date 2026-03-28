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


def block_contents(text, start_marker, end_marker):
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    return text[start:end].strip()


def parse_markdown_rows(block):
    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if stripped.startswith("|---"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if not parts or parts[0] == "Model":
            continue
        rows.append((parts[0], stripped))
    return rows


def merge_missing_rows(rows, existing_rows):
    seen = {row["model"] for row in rows}
    merged = list(rows)
    for model, line in existing_rows:
        if model not in seen:
            merged.append({"model": model, "_existing_line": line})
    return merged


def format_snapshot(rows, existing_rows):
    rows = merge_missing_rows(rows, existing_rows)
    lines = [
        "| Model | Holdout Snapshot | Artifact | Position |",
        "|---|---|---|---|",
    ]
    for row in rows:
        if "_existing_line" in row:
            lines.append(row["_existing_line"])
            continue
        artifact, position = ARTIFACTS.get(row["model"], ("generated artifact", "deployable candidate"))
        lines.append(
            f'| {row["model"]} | `{row["accuracy"]:.4f} acc` / `{row["roc_auc"]:.4f} auc` | `{artifact}` | {position} |'
        )
    return "\n".join(lines)


def format_compare(rows, existing_rows):
    rows = merge_missing_rows(rows, existing_rows)
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if "_existing_line" in row:
            lines.append(row["_existing_line"])
            continue
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
    existing_snapshot = parse_markdown_rows(block_contents(readme, SNAPSHOT_START, SNAPSHOT_END))
    existing_compare = parse_markdown_rows(block_contents(readme, COMPARE_START, COMPARE_END))
    readme = replace_block(readme, SNAPSHOT_START, SNAPSHOT_END, format_snapshot(rows, existing_snapshot))
    readme = replace_block(readme, COMPARE_START, COMPARE_END, format_compare(rows, existing_compare))
    README.write_text(readme, encoding="utf-8")
    print("Updated README metrics from:", REPORT)


if __name__ == "__main__":
    main()
