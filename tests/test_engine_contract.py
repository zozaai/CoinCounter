import unittest
import numpy as np
from coincounter.engines.random_guess import RandomGuessEngine


class EngineTests(unittest.TestCase):
    def test_order_independent(self):
        engine = RandomGuessEngine([1, 3, 5], [2, 3, 1])
        a = np.zeros((3, 3, 3), dtype=np.uint8)
        b = np.ones((3, 3, 3), dtype=np.uint8)
        original = engine.run(a)
        engine.run(b)
        self.assertEqual(engine.run(a), original)
        self.assertIn(original.count, [1, 3, 5])
        self.assertIsNone(original.confidence)

    def test_invalid_distribution(self):
        for counts, weights in [([], []), ([1], [0]), ([1, 1], [1, 1]), ([-1], [1])]:
            with self.assertRaises(ValueError):
                RandomGuessEngine(counts, weights)
