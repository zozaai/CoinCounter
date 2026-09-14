import unittest
import numpy as np
from PIL import Image
from coincounter.image import normalize
from coincounter.exceptions import InvalidImageError


class ImageTests(unittest.TestCase):
    def test_pil_grayscale_to_rgb(self):
        self.assertEqual(normalize(Image.new("L", (2, 3))).shape, (3, 2, 3))

    def test_invalid_inputs(self):
        for value in [None, "missing.jpg", np.zeros((2, 2)), np.zeros((0, 2, 3), dtype=np.uint8)]:
            with self.assertRaises(InvalidImageError):
                normalize(value)
