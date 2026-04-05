import contextlib
import json
import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CORE_ML_DEPS_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas", "sklearn"))
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


def run_train_cli(script_path: str):
    train = run_cli(script_path, "train")
    if train.returncode != 0:
        details = train.stderr or train.stdout or "training command failed without output"
        raise AssertionError(f"{script_path} train failed:\n{details}")
    return train


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_block(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    return text[:start] + "\n" + replacement + "\n" + text[end:]


class TestCliSmoke(unittest.TestCase):
    def test_svm_train_and_predict(self):
        train = run_train_cli("scripts/svm_cli.py")
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
        train = run_train_cli("scripts/logreg_cli.py")
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
        run_train_cli("scripts/logreg_cli.py")
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
        run_train_cli("scripts/logreg_cli.py")
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

    def test_logreg_rejects_artifact_missing_columns(self):
        run_train_cli("scripts/logreg_cli.py")
        artifact_path = ROOT / "models" / "logreg_model.pkl"
        original_bytes = artifact_path.read_bytes()
        try:
            bundle = pickle.loads(original_bytes)
            bundle.pop("columns", None)
            artifact_path.write_bytes(pickle.dumps(bundle))

            predict = run_cli("scripts/logreg_cli.py", "predict", "--row-index", "0")
            self.assertNotEqual(predict.returncode, 0)
            self.assertIn("missing feature column list", predict.stderr)
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

    @unittest.skipIf(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is installed")
    def test_compare_models_skips_missing_tensorflow(self):
        compare = subprocess.run(
            [str(PYTHON), "scripts/compare_models.py"],
            cwd=ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(compare.returncode, 0, compare.stderr)
        self.assertIn("TensorFlow is not installed; skipping TensorFlow NN.", compare.stdout)

        report = json.loads((ROOT / "results" / "comparison_report.json").read_text(encoding="utf-8"))
        self.assertFalse(any(row["model"] == "TensorFlow NN" for row in report))

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
            modified = replace_block(
                original,
                "<!-- snapshot-table:start -->",
                "<!-- snapshot-table:end -->",
                "\n".join(
                    [
                        "| Model | Holdout Snapshot | Artifact | Position |",
                        "|---|---|---|---|",
                        "| Logistic Regression | `0.0000 acc` / `0.0000 auc` | `models/logreg_model.pkl` | stale placeholder |",
                        "| SVM | `0.0000 acc` / `0.0000 auc` | `models/svm_model.pkl` | stale placeholder |",
                        "| TensorFlow NN | `0.0000 acc` / `0.0000 auc` | `models/tensorflow_model/` | stale placeholder |",
                    ]
                ),
            )
            modified = replace_block(
                modified,
                "<!-- compare-table:start -->",
                "<!-- compare-table:end -->",
                "\n".join(
                    [
                        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
                        "|---|---:|---:|---:|---:|---:|",
                        "| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |",
                        "| SVM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |",
                        "| TensorFlow NN | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |",
                    ]
                ),
            )
            readme_path.write_text(modified, encoding="utf-8")

            update = run_cli("scripts/update_readme_metrics.py")
            self.assertEqual(update.returncode, 0, update.stderr)

            refreshed = readme_path.read_text(encoding="utf-8")
            report = json.loads((ROOT / "results" / "comparison_report.json").read_text(encoding="utf-8"))
            artifacts = {
                "Logistic Regression": ("models/logreg_model.pkl", "strongest lightweight baseline"),
                "SVM": ("models/svm_model.pkl", "margin-based baseline"),
                "TensorFlow NN": ("models/tensorflow_model/", "neural-network baseline"),
            }

            self.assertNotIn("stale placeholder", refreshed)
            self.assertNotIn("| Logistic Regression | `0.0000 acc` / `0.0000 auc` |", refreshed)
            self.assertNotIn("| Logistic Regression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |", refreshed)
            for row in report:
                artifact, position = artifacts.get(row["model"], ("generated artifact", "deployable candidate"))
                snapshot_line = (
                    f'| {row["model"]} | `{row["accuracy"]:.4f} acc` / `{row["roc_auc"]:.4f} auc` | '
                    f'`{artifact}` | {position} |'
                )
                compare_line = (
                    f'| {row["model"]} | {row["accuracy"]:.4f} | {row["precision"]:.4f} | '
                    f'{row["recall"]:.4f} | {row["f1"]:.4f} | {row["roc_auc"]:.4f} |'
                )
                self.assertIn(snapshot_line, refreshed)
                self.assertIn(compare_line, refreshed)

            self.assertNotIn("| TensorFlow NN | `0.9649 acc` / `0.9917 auc` | `models/tensorflow_model/` | neural-network baseline |", refreshed)
            self.assertNotIn("| TensorFlow NN | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9917 |", refreshed)
        finally:
            readme_path.write_text(original, encoding="utf-8")

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_load_bundle_rejects_corrupt_pickle(self):
        model_utils = load_module(ROOT / "scripts" / "model_utils.py", "model_utils_corrupt_pickle")
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"not a pickle payload")
            artifact_path = Path(handle.name)

        try:
            with self.assertRaises(SystemExit) as exc:
                model_utils.load_bundle(artifact_path)
        finally:
            artifact_path.unlink(missing_ok=True)

        self.assertIn("could not deserialize artifact payload", str(exc.exception))
        self.assertIn("Retrain the model artifact", str(exc.exception))

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_load_bundle_rejects_missing_threshold(self):
        model_utils = load_module(ROOT / "scripts" / "model_utils.py", "model_utils_missing_threshold")
        bundle = {
            "columns": list(model_utils.FEATURE_COLUMNS),
            "metadata": {
                "artifact": "evaluated_holdout_model",
                "artifact_schema_version": 1,
                "split": {"random_state": 42},
            },
        }
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(pickle.dumps(bundle))
            artifact_path = Path(handle.name)

        try:
            with self.assertRaises(SystemExit) as exc:
                model_utils.load_bundle(artifact_path)
        finally:
            artifact_path.unlink(missing_ok=True)

        self.assertIn("missing or invalid decision threshold", str(exc.exception))
        self.assertIn("Retrain the model artifact", str(exc.exception))

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_load_bundle_rejects_wrong_feature_columns(self):
        model_utils = load_module(ROOT / "scripts" / "model_utils.py", "model_utils_wrong_columns")
        bundle = {
            "columns": list(reversed(model_utils.FEATURE_COLUMNS)),
            "metadata": {
                "artifact": "evaluated_holdout_model",
                "artifact_schema_version": 1,
                "threshold": 0.5,
                "split": {"random_state": 42},
            },
        }
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(pickle.dumps(bundle))
            artifact_path = Path(handle.name)

        try:
            with self.assertRaises(SystemExit) as exc:
                model_utils.load_bundle(artifact_path)
        finally:
            artifact_path.unlink(missing_ok=True)

        self.assertIn("unexpected feature column schema", str(exc.exception))
        self.assertIn("Retrain the model artifact", str(exc.exception))

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_require_tensorflow_wraps_runtime_import_failures(self):
        model_utils = load_module(ROOT / "scripts" / "model_utils.py", "model_utils_test")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tensorflow":
                raise RuntimeError("broken runtime")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch.object(model_utils, "suppress_native_stderr", new=contextlib.nullcontext):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(SystemExit) as exc:
                    model_utils.require_tensorflow()

        self.assertIn("TensorFlow could not be imported cleanly", str(exc.exception))
        self.assertIn("RuntimeError: broken runtime", str(exc.exception))

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_compare_models_skips_broken_tensorflow_runtime(self):
        scripts_dir = str(ROOT / "scripts")
        sys.path.insert(0, scripts_dir)
        try:
            compare_models = load_module(ROOT / "scripts" / "compare_models.py", "compare_models_runtime_skip")
        finally:
            sys.path.remove(scripts_dir)

        with mock.patch.object(compare_models.importlib.util, "find_spec", return_value=object()):
            with mock.patch.object(compare_models, "require_tensorflow", side_effect=SystemExit("broken runtime")):
                row, note = compare_models.evaluate_tensorflow(None, None, None, None)

        self.assertIsNone(row)
        self.assertIn("skipped", note.lower())
        self.assertIn("broken runtime", note)

    @unittest.skipUnless(CORE_ML_DEPS_AVAILABLE, "Core ML dependencies are not installed")
    def test_compare_models_propagates_tensorflow_failures(self):
        scripts_dir = str(ROOT / "scripts")
        sys.path.insert(0, scripts_dir)
        try:
            compare_models = load_module(ROOT / "scripts" / "compare_models.py", "compare_models_runtime_fail")
        finally:
            sys.path.remove(scripts_dir)

        with mock.patch.object(compare_models.importlib.util, "find_spec", return_value=object()):
            with mock.patch.object(compare_models, "require_tensorflow", return_value=object()):
                with mock.patch.object(
                    compare_models,
                    "tune_tensorflow_threshold",
                    side_effect=RuntimeError("bad training path"),
                ):
                    with self.assertRaises(RuntimeError) as exc:
                        compare_models.evaluate_tensorflow(None, None, None, None)

        self.assertIn("bad training path", str(exc.exception))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
    def test_tensorflow_train_and_predict(self):
        train = run_train_cli("scripts/tensorflow_cli.py")
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

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
    def test_tensorflow_rejects_artifact_missing_scaler(self):
        run_train_cli("scripts/tensorflow_cli.py")
        artifact_path = ROOT / "models" / "tensorflow_model" / "metadata.pkl"
        original_bytes = artifact_path.read_bytes()
        try:
            bundle = pickle.loads(original_bytes)
            bundle.pop("scaler", None)
            artifact_path.write_bytes(pickle.dumps(bundle))

            predict = run_cli("scripts/tensorflow_cli.py", "predict", "--row-index", "0")
            self.assertNotEqual(predict.returncode, 0)
            self.assertIn("missing scaler payload", predict.stderr)
            self.assertIn("Retrain the model artifact", predict.stderr)
        finally:
            artifact_path.write_bytes(original_bytes)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
    def test_tensorflow_rejects_corrupt_saved_model(self):
        run_train_cli("scripts/tensorflow_cli.py")
        original_bytes = (ROOT / "models" / "tensorflow_model" / "model.keras").read_bytes()
        try:
            (ROOT / "models" / "tensorflow_model" / "model.keras").write_bytes(b"corrupt tensorflow model")

            predict = run_cli("scripts/tensorflow_cli.py", "predict", "--row-index", "0")
            self.assertNotEqual(predict.returncode, 0)
            self.assertIn("could not load TensorFlow model payload", predict.stderr)
            self.assertIn("Retrain the model artifact", predict.stderr)
        finally:
            (ROOT / "models" / "tensorflow_model" / "model.keras").write_bytes(original_bytes)
