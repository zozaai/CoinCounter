from abc import ABC, abstractmethod


class BaseEngine(ABC):
    @abstractmethod
    def run(self, image):
        """Return CountResult for a normalized RGB array."""

    def close(self):
        """Release resources, if any."""
