import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from coincounter import CoinCounter
from coincounter.engines.regression import RegressionEngine
from coincounter.exceptions import ModelLoadingError, UnavailableEngineError

HAS_TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")


class RegressionValidationTests(unittest.TestCase):
    def test_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"torch": None, "torchvision": None}):
            with self.assertRaisesRegex(UnavailableEngineError, "regression"):
                RegressionEngine("missing.pt")

    def test_invalid_parameters(self):
        for parameters in [{"image_size": 16}, {"image_size": 224.0}, {"max_count": 0}, {"max_count": True}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                RegressionEngine("missing.pt", **parameters)


@unittest.skipUnless(HAS_TORCH, "regression extra not installed")
class RegressionTests(unittest.TestCase):
    def test_missing_and_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ModelLoadingError):
                RegressionEngine(Path(directory) / "none.pt")
            corrupt = Path(directory) / "bad.pt"
            corrupt.write_bytes(b"not a checkpoint")
            with self.assertRaises(ModelLoadingError):
                RegressionEngine(corrupt)

    def test_bounded_deterministic_count(self):
        import torch
        from coincounter.engines.regression import build_model
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "model.pt"
            torch.manual_seed(0)
            torch.save(build_model().state_dict(), weights)
            image = np.random.default_rng(0).integers(0, 255, (100, 140, 3), dtype=np.uint8)
            with CoinCounter("regression", {"model_path": str(weights), "image_size": 64, "max_count": 20, "device": "cpu"}) as counter:
                result = counter.run(image)
                self.assertIs(type(result.count), int)
                self.assertTrue(0 <= result.count <= 20)
                self.assertTrue(0 <= result.metadata["raw_count"] <= 20)
                self.assertIsNone(result.confidence)
                self.assertEqual(counter.run(image).count, result.count)
                self.assertEqual(len(result.metadata["weights_sha256"]), 64)
