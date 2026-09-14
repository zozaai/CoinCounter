"""Classical circle detection; radii and spacing are in input-image pixels."""
import math

from ..exceptions import UnavailableEngineError
from ..types import CountResult
from .base import BaseEngine


class HoughEngine(BaseEngine):
    def __init__(self, dp=1.0, min_distance=24, min_radius=10, max_radius=30,
                 edge_threshold=200, accumulator_threshold=30, blur_size=5):
        for name, value in {"dp": dp, "min_distance": min_distance,
                            "edge_threshold": edge_threshold,
                            "accumulator_threshold": accumulator_threshold}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if dp < 1:
            raise ValueError("dp must be at least 1")
        if (type(min_radius) is not int or type(max_radius) is not int
                or min_radius < 1 or max_radius < min_radius):
            raise ValueError("Require integer radii: 1 <= min_radius <= max_radius")
        if type(blur_size) is not int or blur_size < 3 or blur_size % 2 == 0:
            raise ValueError("blur_size must be an odd integer >= 3")
        try:
            import cv2
        except ImportError as exc:
            raise UnavailableEngineError("Install the Hough extra: pip install 'coincounter[hough]'") from exc
        self.cv = cv2
        self.parameters = dict(dp=dp, minDist=min_distance, minRadius=min_radius,
                               maxRadius=max_radius, param1=edge_threshold,
                               param2=accumulator_threshold)
        self.blur_size = blur_size

    def run(self, image):
        gray = self.cv.cvtColor(image, self.cv.COLOR_RGB2GRAY)
        gray = self.cv.medianBlur(gray, self.blur_size)
        circles = self.cv.HoughCircles(gray, self.cv.HOUGH_GRADIENT, **self.parameters)
        detections = [] if circles is None else [
            {"x": float(x), "y": float(y), "radius": float(radius)}
            for x, y, radius in circles[0]
        ]
        detections.sort(key=lambda circle: (circle["y"], circle["x"]))
        return CountResult(count=len(detections), detections=detections,
                           metadata={"opencv_version": self.cv.__version__,
                                     "opencv_threads": self.cv.getNumThreads(),
                                     "coordinate_system": "input RGB image pixels"})
