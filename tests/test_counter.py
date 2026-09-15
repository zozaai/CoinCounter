import unittest
import numpy as np
from coincounter import CoinCounter
from coincounter.exceptions import InferenceError, UnavailableEngineError


class CounterTests(unittest.TestCase):
    def test_lifecycle_and_reuse(self):
        with CoinCounter("random_guess", {"counts": [7], "weights": [1]}) as counter:
            for _ in range(3):
                self.assertEqual(counter.run(np.zeros((4, 4, 3), dtype=np.uint8)).count, 7)
        counter.close()
        with self.assertRaises(InferenceError):
            counter.run(None)

    def test_unimplemented_engine(self):
        with self.assertRaises(UnavailableEngineError):
            CoinCounter("vision_llm")
