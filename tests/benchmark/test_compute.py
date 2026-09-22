import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
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
            args = ["--dataset", str(dataset), "--split", "test", "--engine", "random_guess", "--results", str(root / "results"),
                    "--reports", str(root / "reports")]
            def run(extra=()):
                output = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    status = main(args + list(extra))
                self.assertEqual(status, 0)
                report = json.loads(next((root / "reports").rglob("summary.json")).read_text())
                summary_path = root / "results" / "test_summary.json"
                self.assertEqual(json.loads(summary_path.read_text()), report)
                self.assertEqual(report["engines"][0]["failed"], 0)
                chart = root / "results" / "test_accuracy_vs_time.html"
                self.assertIn("Plotly.newPlot", chart.read_text())
                self.assertIn(str(summary_path), output.getvalue())
                self.assertIn(str(chart), output.getvalue())
                return report["engines"][0]
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

    def test_failed_engine_still_exports_summary_and_chart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "coin.png"
            Image.new("RGB", (4, 4)).save(image)
            output, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                status = main(["--image", str(image), "--engine", "missing_engine",
                               "--results", str(root / "results"), "--reports", str(root / "reports")])
            self.assertEqual(status, 1)
            self.assertEqual(output.getvalue(), "")
            summary_path = root / "results" / "single_summary.json"
            row = json.loads(summary_path.read_text())["engines"][0]
            self.assertEqual(row["failed"], 1)
            self.assertIsNone(row["accuracy"])
            self.assertIn("No scored models with timing data.",
                          (root / "results" / "single_accuracy_vs_time.html").read_text())
            self.assertIn("Plotly HTML:", errors.getvalue())

    def test_missing_plotly_is_reported_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "coin.png"
            Image.new("RGB", (4, 4)).save(image)
            errors = io.StringIO()
            with patch("benchmark.compute.find_spec", return_value=None), \
                    patch("benchmark.compute.CoinCounter") as counter, \
                    contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                main(["--image", str(image), "--engine", "hough"])
            self.assertEqual(raised.exception.code, 2)
            counter.assert_not_called()
            self.assertIn("pip install -e '.[benchmark]'", errors.getvalue())

    def test_metrics(self):
        result = summarize([{"count": 2, "actual": 2, "seconds": .1, "cached": False},
                            {"count": 5, "actual": 3, "seconds": .3, "cached": True}], 2)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["mae"], 1)
        self.assertAlmostEqual(result["mean_ms"], 200)
