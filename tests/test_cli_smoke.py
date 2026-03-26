import importlib.util
import subprocess
import sys
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

        predict = run_cli("scripts/svm_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)

    def test_logreg_train_and_predict(self):
        train = run_cli("scripts/logreg_cli.py", "train")
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("Saved artifact: evaluated_holdout_model", train.stdout)
        self.assertTrue((ROOT / "models" / "logreg_model.pkl").exists())

        predict = run_cli("scripts/logreg_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow") is not None, "TensorFlow is not installed")
    def test_tensorflow_train_and_predict(self):
        train = run_cli("scripts/tensorflow_cli.py", "train")
        self.assertEqual(train.returncode, 0, train.stderr)
        self.assertIn("Saved artifact: evaluated_holdout_model", train.stdout)
        self.assertTrue((ROOT / "models" / "tensorflow_model" / "model.keras").exists())
        self.assertTrue((ROOT / "models" / "tensorflow_model" / "metadata.pkl").exists())

        predict = run_cli("scripts/tensorflow_cli.py", "predict", "--row-index", "0")
        self.assertEqual(predict.returncode, 0, predict.stderr)
        self.assertIn("Predicted diagnosis:", predict.stdout)
