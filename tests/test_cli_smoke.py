import json
import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run_cli(*args):
    return subprocess.run(
        [str(PYTHON), *args],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )


class TestCliSmoke(unittest.TestCase):
    def test_svm_train_and_predict(self):
        train = run_cli("scripts/svm_cli.py", "train")
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("Saved artifact: evaluated_holdout_model", train.stdout)
        self.assertTrue((ROOT / "models" / "svm_model.pkl").exists())
        bundle = pickle.loads((ROOT / "models" / "svm_model.pkl").read_bytes())
        self.assertEqual(bundle["metadata"]["artifact_schema_version"], 1)
        self.assertIn("threshold", bundle["metadata"])
        self.assertIn("cv_metrics", bundle["metadata"])

        predict = run_cli("scripts/svm_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)
        self.assertIn("Decision threshold:", predict.stdout)

    def test_logreg_train_and_predict(self):
        train = run_cli("scripts/logreg_cli.py", "train")
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("Saved artifact: evaluated_holdout_model", train.stdout)
        self.assertTrue((ROOT / "models" / "logreg_model.pkl").exists())
        bundle = pickle.loads((ROOT / "models" / "logreg_model.pkl").read_bytes())
        self.assertEqual(bundle["metadata"]["artifact_schema_version"], 1)
        self.assertIn("threshold", bundle["metadata"])
        self.assertIn("cv_metrics", bundle["metadata"])

        predict = run_cli("scripts/logreg_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)
        self.assertIn("Decision threshold:", predict.stdout)

    def test_logreg_predict_from_input_json_and_invalid_row(self):
        run_cli("scripts/logreg_cli.py", "train")
        payload = {
            "radius_mean": 17.99,
            "texture_mean": 10.38,
            "perimeter_mean": 122.8,
            "area_mean": 1001.0,
            "smoothness_mean": 0.1184,
            "compactness_mean": 0.2776,
            "concavity_mean": 0.3001,
            "concave_points_mean": 0.1471,
            "symmetry_mean": 0.2419,
            "fractal_dimension_mean": 0.07871,
            "radius_se": 1.095,
            "texture_se": 0.9053,
            "perimeter_se": 8.589,
            "area_se": 153.4,
            "smoothness_se": 0.006399,
            "compactness_se": 0.04904,
            "concavity_se": 0.05373,
            "concave_points_se": 0.01587,
            "symmetry_se": 0.03003,
            "fractal_dimension_se": 0.006193,
            "radius_worst": 25.38,
            "texture_worst": 17.33,
            "perimeter_worst": 184.6,
            "area_worst": 2019.0,
            "smoothness_worst": 0.1622,
            "compactness_worst": 0.6656,
            "concavity_worst": 0.7119,
            "concave_points_worst": 0.2654,
            "symmetry_worst": 0.4601,
            "fractal_dimension_worst": 0.1189,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            input_path = handle.name

        predict = run_cli("scripts/logreg_cli.py", "predict", "--input-json", input_path)
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)
        self.assertNotIn("Actual diagnosis:", predict.stdout)

        invalid = run_cli("scripts/logreg_cli.py", "predict", "--row-index", "9999")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("Row index must be between", invalid.stderr)

    def test_logreg_rejects_invalid_artifact_schema(self):
        run_cli("scripts/logreg_cli.py", "train")
        artifact_path = ROOT / "models" / "logreg_model.pkl"
        original_bytes = artifact_path.read_bytes()
        try:
            bundle = pickle.loads(original_bytes)
            bundle["metadata"].pop("artifact_schema_version", None)
            artifact_path.write_bytes(pickle.dumps(bundle))

            predict = run_cli("scripts/logreg_cli.py", "predict", "--row-index", "0")
            self.assertNotEqual(predict.returncode, 0)
            self.assertIn("Retrain the model artifact", predict.stderr)
        finally:
            artifact_path.write_bytes(original_bytes)

    def test_compare_models_exports_results(self):
        compare = subprocess.run(
            [str(PYTHON), "scripts/compare_models.py"],
            cwd=ROOT,
            env={**os.environ, "MAMMO_SKIP_TENSORFLOW": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(compare.returncode, 0, compare.stderr)
        self.assertTrue((ROOT / "results" / "comparison_summary.csv").exists())
        self.assertTrue((ROOT / "results" / "comparison_report.json").exists())
        report = json.loads((ROOT / "results" / "comparison_report.json").read_text(encoding="utf-8"))
        self.assertTrue(any(row["model"] == "Logistic Regression" for row in report))
        self.assertTrue(all("calibration_bins" in row for row in report))

    def test_update_readme_metrics_from_report(self):
        compare = subprocess.run(
            [str(PYTHON), "scripts/compare_models.py"],
            cwd=ROOT,
            env={**os.environ, "MAMMO_SKIP_TENSORFLOW": "1"},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(compare.returncode, 0, compare.stderr)

        readme_path = ROOT / "README.md"
        original = readme_path.read_text(encoding="utf-8")
        try:
            modified = original.replace("| Logistic Regression | `0.9737 acc` / `0.9960 auc` |", "| Logistic Regression | `0.0000 acc` / `0.0000 auc` |")
            readme_path.write_text(modified, encoding="utf-8")

            update = run_cli("scripts/update_readme_metrics.py")
            self.assertEqual(update.returncode, 0, update.stderr)

            refreshed = readme_path.read_text(encoding="utf-8")
            self.assertIn("| Logistic Regression | `0.9737 acc` / `0.9960 auc` |", refreshed)
            self.assertIn("| SVM | 0.9649 | 0.9524 | 0.9524 | 0.9524 | 0.9947 |", refreshed)
            self.assertIn("| TensorFlow NN | `0.9649 acc` / `0.9917 auc` | `models/tensorflow_model/` | neural-network baseline |", refreshed)
            self.assertIn("| TensorFlow NN | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9917 |", refreshed)
        finally:
            readme_path.write_text(original, encoding="utf-8")

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
    def test_tensorflow_train_and_predict(self):
        train = run_cli("scripts/tensorflow_cli.py", "train")
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("Saved artifact: evaluated_holdout_model", train.stdout)
        self.assertTrue((ROOT / "models" / "tensorflow_model" / "model.keras").exists())
        self.assertTrue((ROOT / "models" / "tensorflow_model" / "metadata.pkl").exists())
        bundle = pickle.loads((ROOT / "models" / "tensorflow_model" / "metadata.pkl").read_bytes())
        self.assertEqual(bundle["metadata"]["artifact_schema_version"], 1)
        self.assertIn("threshold", bundle["metadata"])
        self.assertIn("cv_metrics", bundle["metadata"])

        predict = run_cli("scripts/tensorflow_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)
