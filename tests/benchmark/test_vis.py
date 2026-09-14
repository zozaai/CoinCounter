import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen
from PIL import Image
from benchmark.compute import main as compute
from benchmark.vis import Viewer, serve


def make_dataset(root):
    dataset = root / "dataset"
    folder = dataset / "test"
    (folder / "images").mkdir(parents=True)
    (dataset / "train" / "images").mkdir(parents=True)
    for name, colour in [("a.png", "white"), ("b.png", "gray")]:
        Image.new("RGB", (40, 40), colour).save(folder / "images" / name)
    (folder / "labels.json").write_text(json.dumps({"a.png": 3, "b.png": 1}))
    Image.new("RGB", (40, 40)).save(dataset / "train/images/c.png")
    (dataset / "train/labels.json").write_text(json.dumps({"c.png": 2}))
    return dataset


class ViewerTests(unittest.TestCase):
    def test_missing_results_are_reported_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            viewer = Viewer(make_dataset(root), "hough", root / "results", "test")
            self.assertEqual(viewer.configurations, {})
            item = viewer.item(0)
            self.assertEqual(item["filename"], "a.png")
            self.assertEqual(item["actual"], 3)
            self.assertIsNone(item["predicted"])
            self.assertIn("No saved results", item["messages"][0])
            self.assertIsInstance(viewer.stage(0, "original"), bytes)
            self.assertIsInstance(viewer.stage(0, "grayscale"), bytes)
            self.assertIn("blur_size", viewer.stage(0, "blur"))
            self.assertIn("no saved prediction", viewer.stage(0, "detections"))

    def test_reads_saved_random_guess_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = compute(["--dataset", str(dataset), "--split", "test", "--engine", "random_guess",
                                  "--results", str(root / "results"), "--reports", str(root / "reports")])
            self.assertEqual(status, 0)
            viewer = Viewer(dataset, "random_guess", root / "results", "test")
            self.assertEqual(len(viewer.configurations), 1)
            item = viewer.item(1)
            self.assertEqual(item["filename"], "b.png")
            self.assertEqual(item["messages"], [])
            self.assertEqual(item["predicted"], 2)
            self.assertEqual(item["error"], 1)
            self.assertGreaterEqual(item["seconds"], 0)
            self.assertEqual(item["stages"], ["original"])
            self.assertEqual(viewer.index_of(dataset / "test/images/b.png"), 1)
            with self.assertRaises(ValueError):
                viewer.index_of("missing.png")

    def test_hough_stages_from_manifest_and_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = make_dataset(root)
            viewer = Viewer(dataset, "hough", root / "results", "test")
            configuration = root / "results/hough/abc"
            configuration.mkdir(parents=True)
            (configuration / "manifest.json").write_text(json.dumps({"parameters": {"blur_size": 5}}))
            path = viewer.items[0][0]
            record = {"count": 1, "seconds": 0.01, "detections": [{"x": 20, "y": 20, "radius": 8}]}
            base = configuration / viewer.dataset_id / "test" / f"a.png-{viewer.image_hash(path)[:12]}.json"
            base.parent.mkdir(parents=True)
            base.write_text(json.dumps(record))
            viewer = Viewer(dataset, "hough", root / "results", "test")
            item = viewer.item(0)
            self.assertEqual(item["stages"], ["original", "grayscale", "blur", "detections"])
            self.assertEqual(item["error"], -2)
            for name in item["stages"]:
                self.assertIsInstance(viewer.stage(0, name), bytes, name)
            overlay = Image.open(io.BytesIO(viewer.stage(0, "detections")))
            self.assertEqual(overlay.getpixel((28, 20))[:2], (255, 40))
            self.assertIn("No saved prediction", viewer.item(1)["messages"][0])

    def test_http_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            viewer = Viewer(make_dataset(root), "hough", root / "results", "test")
            server = serve(viewer, 1)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}"
                page = urlopen(url + "/").read().decode()
                self.assertIn("index: 1", page)
                item = json.loads(urlopen(url + "/api/item?index=0").read())
                self.assertEqual(item["filename"], "a.png")
                with urlopen(url + "/stage/original?index=0") as response:
                    self.assertEqual(response.headers["Content-Type"], "image/png")
                with urlopen(url + "/stage/detections?index=0") as response:
                    self.assertTrue(response.headers["Content-Type"].startswith("text/plain"))
                with self.assertRaises(Exception):
                    urlopen(url + "/api/item?index=5")
            finally:
                server.shutdown()
                server.server_close()
