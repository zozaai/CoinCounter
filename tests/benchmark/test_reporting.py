import json
from pathlib import Path
import tempfile
import unittest

from benchmark.metrics import pareto_front
from benchmark.reporting import accuracy_time_figure, save_comparison


def row(engine, mean_ms, accuracy, **changes):
    return {"engine": engine, "mean_ms": mean_ms, "accuracy": accuracy,
            "scored": 4, "completed": 4, "expected": 4, "errors": [],
            "mae": 0.5, "total_seconds": mean_ms * 4 / 1000 if mean_ms is not None else 0,
            "cached": 0, "failed": 0, **changes}


class ReportingTests(unittest.TestCase):
    def test_frontier_handles_dominance_ties_and_input_order(self):
        rows = [row("slow_best", 20, 1), row("dominated", 15, 0.6),
                row("same_time_worse", 10, 0.6), row("middle", 10, 0.8),
                row("same_accuracy_slower", 12, 0.8), row("fast", 1, 0),
                row("middle_tie", 10, 0.8)]
        expected = ["fast", "middle", "middle_tie", "slow_best"]
        self.assertEqual([r["engine"] for r in pareto_front(rows)], expected)
        self.assertEqual([r["engine"] for r in pareto_front(list(reversed(rows)))], expected)
        self.assertEqual(pareto_front([]), [])

    def test_frontier_excludes_missing_invalid_and_incomplete_scores(self):
        complete = row("complete", 10, 0.5)
        rows = [complete, row("failed", None, None, completed=0, scored=0, errors=[{}]),
                row("partial", 1, 1, completed=1, scored=1, errors=[{}]),
                row("unlabelled", 1, 1, scored=1), row("nan", float("nan"), 1),
                row("infinite", 1, float("inf"))]
        self.assertEqual(pareto_front(rows), [complete])

    def test_chart_axes_circles_and_frontier(self):
        rows = [row("best", 20, 1), row("dominated", 15, 0.6), row("middle", 10, 0.8),
                row("fast", 1, 0), row("middle_tie", 10, 0.8),
                row("partial", 0.5, 1, completed=1, scored=1, errors=[{}]),
                row("failed", None, None, completed=0, scored=0, errors=[{}])]
        figure = accuracy_time_figure({"split": "test", "engines": rows})
        frontier, *models = figure.data
        self.assertEqual(frontier.mode, "lines")
        self.assertEqual(list(frontier.x), [1, 10, 20])
        self.assertEqual(list(frontier.y), [0, 0.8, 1])
        self.assertEqual(len(models), 6)
        for model, expected in zip(models, rows):
            self.assertEqual(model.marker.symbol, "circle")
            self.assertEqual(list(model.x), [expected["mean_ms"]])
            self.assertEqual(list(model.y), [expected["accuracy"]])
        self.assertIn("partial", models[-1].name)
        self.assertEqual(figure.layout.xaxis.title.text, "Processing time (Mean ms)")
        self.assertEqual(figure.layout.yaxis.title.text, "Accuracy")
        self.assertEqual(figure.layout.yaxis.tickformat, ".0%")
        self.assertIn("failed", figure.layout.annotations[0].text)

    def test_empty_and_single_model_charts(self):
        figure = accuracy_time_figure({"split": "test", "engines": []})
        self.assertEqual(len(figure.data), 0)
        self.assertTrue(any("No scored models" in a.text for a in figure.layout.annotations))
        figure = accuracy_time_figure({"split": "test", "engines": [row("one", 0, 0)]})
        self.assertEqual(list(figure.data[0].x), [0])
        self.assertEqual(figure.data[1].name, "one")

    def test_export_preserves_metrics_and_embeds_plotly(self):
        summary = {"split": "test", "engines": [row("fast", 2.339389626, 0.0625)]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "nested" / "test"
            save_comparison(folder, summary)
            self.assertEqual(json.loads((folder / "summary.json").read_text()), summary)
            html = (folder / "accuracy_vs_time.html").read_text()
            self.assertIn("<html>", html)
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("plotly.js v", html)
            self.assertNotIn('<script src="', html)
            self.assertIn("fast", html)
