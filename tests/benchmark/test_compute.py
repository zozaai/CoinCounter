import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from benchmark.compute import main
from benchmark.metrics import summarize


class ComputeTests(unittest.TestCase):
    def test_cache_force_and_no_test_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for split, count in [("train", 2), ("test", 9)]:
                folder = dataset / split
                (folder / "images").mkdir(parents=True)
                Image.new("RGB", (4, 4)).save(folder / "images/a.png")
                (folder / "labels.json").write_text(json.dumps({"a.png": count}))
            args = ["--dataset", str(dataset), "--split", "test", "--results", str(root / "results"),
                    "--reports", str(root / "reports")]
            def run(extra=()):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    status = main(args + list(extra))
                self.assertEqual(status, 0)
                return json.loads(next((root / "reports").rglob("summary.json")).read_text())["engines"][0]
            first = run()
            self.assertEqual(first["mae"], 7)
            self.assertEqual(first["cached"], 0)
            self.assertEqual(run()["cached"], 1)
            self.assertEqual(run(["--force"])["cached"], 0)
            prediction = next((root / "results").rglob("*.txt"))
            prediction.write_text("999\n")
            self.assertEqual(run()["cached"], 0)
            self.assertEqual(prediction.read_text(), "2\n")
            Image.new("RGB", (5, 5), "red").save(dataset / "test/images/a.png")
            self.assertEqual(run()["cached"], 0)

    def test_metrics(self):
        result = summarize([{"count": 2, "actual": 2, "seconds": .1, "cached": False},
                            {"count": 5, "actual": 3, "seconds": .3, "cached": True}], 2)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["mae"], 1)
        self.assertAlmostEqual(result["mean_ms"], 200)
