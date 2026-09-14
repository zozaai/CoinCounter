from .engines.registry import create_engine
from .exceptions import CoinCounterError, InferenceError
from .image import normalize


class CoinCounter:
    """Reusable counter. Instances are not guaranteed thread-safe."""
    def __init__(self, engine, parameters=None):
        self.engine = create_engine(engine, parameters or {})
        self._closed = False

    def run(self, image):
        if self._closed:
            raise InferenceError("Counter is closed")
        frame = normalize(image)
        try:
            return self.engine.run(frame)
        except CoinCounterError:
            raise
        except Exception as exc:
            raise InferenceError(str(exc)) from exc

    def close(self):
        if not self._closed:
            self.engine.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
