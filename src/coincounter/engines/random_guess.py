"""Seeded empirical-prior baseline; never reads evaluation labels."""
import hashlib
import random

from ..types import CountResult
from .base import BaseEngine


class RandomGuessEngine(BaseEngine):
    def __init__(self, counts, weights, seed=42):
        if (not counts or len(counts) != len(weights)
                or any(type(c) is not int or c < 0 for c in counts)
                or any(type(w) is not int or w <= 0 for w in weights)
                or len(set(counts)) != len(counts)):
            raise ValueError("Provide unique nonnegative counts and positive integer frequencies")
        self.counts = list(counts)
        self.weights = list(weights)
        self.seed = int(seed)

    def run(self, image):
        # Image identity supplies deterministic randomness, not visual evidence.
        # This makes results independent of traversal order and partial cache hits.
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        rng = random.Random(f"{self.seed}:{digest}")
        count = rng.choices(self.counts, weights=self.weights, k=1)[0]
        return CountResult(count=count, metadata={"seed": self.seed, "baseline": "training-count empirical prior"})
