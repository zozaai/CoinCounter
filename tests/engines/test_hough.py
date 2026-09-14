import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from coincounter import CoinCounter
from coincounter.engines.hough import HoughEngine
from coincounter.exceptions import UnavailableEngineError


class HoughValidationTests(unittest.TestCase):
    def test_invalid_parameters(self):
        for parameters in [{"min_radius": 0}, {"max_radius": 2}, {"blur_size": 4},
                           {"dp": .5}, {"edge_threshold": float("nan")},
                           {"accumulator_threshold": -1}, {"min_distance": True}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                HoughEngine(**parameters)

    def test_missing_optional_dependency(self):
        with patch.dict("sys.modules", {"cv2": None}):
            with self.assertRaisesRegex(UnavailableEngineError, "hough"):
                HoughEngine()


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV extra not installed")
class HoughTests(unittest.TestCase):
    def test_blank_image(self):
        with CoinCounter("hough") as counter:
            result = counter.run(np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertEqual(result.count, 0)
        self.assertEqual(result.detections, [])
        self.assertIsNone(result.confidence)

    def test_two_circles_and_reuse(self):
        import cv2
        image = np.zeros((180, 240, 3), dtype=np.uint8)
        for center in [(60, 90), (180, 90)]:
            cv2.circle(image, center, 22, (255, 255, 255), -1)
        with CoinCounter("hough", {"accumulator_threshold": 10}) as counter:
            result = counter.run(image)
            self.assertEqual(result.count, 2)
            self.assertEqual(counter.run(image), result)
            detections = sorted(result.detections, key=lambda d: d["x"])
            for detection, x in zip(detections, [60, 180]):
                self.assertAlmostEqual(detection["x"], x, delta=3)
                self.assertAlmostEqual(detection["y"], 90, delta=3)
                self.assertAlmostEqual(detection["radius"], 22, delta=3)
