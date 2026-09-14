from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .exceptions import InvalidImageError


def normalize(image):
    """Return a contiguous uint8 RGB array; arrays must already use RGB."""
    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as source:
                return np.array(ImageOps.exif_transpose(source).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.array(ImageOps.exif_transpose(image).convert("RGB"))
        if (isinstance(image, np.ndarray) and image.dtype == np.uint8
                and image.ndim == 3 and image.shape[2] == 3
                and image.shape[0] > 0 and image.shape[1] > 0):
            return np.ascontiguousarray(image)
    except (OSError, ValueError) as exc:
        raise InvalidImageError(str(exc)) from exc
    raise InvalidImageError("Expected an image path, PIL image, or nonempty uint8 RGB array")
