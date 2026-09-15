import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np

from coincounter import CoinCounter
from coincounter.engines.grounding_dino import GroundingDinoEngine
from coincounter.exceptions import ModelLoadingError, UnavailableEngineError

HAS_DEPS = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")
# The tiny checkpoint is ~660 MB, so the real-model test is opt-in.
RUN_MODEL = os.environ.get("COINCOUNTER_TEST_GROUNDING_DINO") == "1"


class GroundingDinoValidationTests(unittest.TestCase):
    def test_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with self.assertRaisesRegex(UnavailableEngineError, "grounding-dino"):
                GroundingDinoEngine()

    def test_invalid_parameters(self):
        for parameters in [{"box_threshold": 1.5}, {"box_threshold": True}, {"text_threshold": -0.1},
                           {"text_threshold": float("nan")}, {"prompt": ""}, {"model": ""}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                GroundingDinoEngine(**parameters)


@unittest.skipUnless(HAS_DEPS, "grounding-dino extra not installed")
class GroundingDinoLoadingTests(unittest.TestCase):
    def test_missing_model(self):
        with self.assertRaises(ModelLoadingError):
            GroundingDinoEngine(model="/nonexistent/model", local_files_only=True)


@unittest.skipUnless(HAS_DEPS and RUN_MODEL, "set COINCOUNTER_TEST_GROUNDING_DINO=1 to run the real model")
class GroundingDinoModelTests(unittest.TestCase):
    def test_blank_image_and_prompt_normalisation(self):
        with CoinCounter("grounding_dino", {"prompt": "Coin"}) as counter:
            self.assertEqual(counter.engine.prompt, "coin.")
            result = counter.run(np.full((240, 240, 3), 255, dtype=np.uint8))
            self.assertIs(type(result.count), int)
            self.assertEqual(result.count, len(result.detections))
            self.assertIsNone(result.confidence)
            self.assertTrue(all(0 <= d["score"] <= 1 for d in result.detections))
            self.assertIsNotNone(result.metadata["revision"])
