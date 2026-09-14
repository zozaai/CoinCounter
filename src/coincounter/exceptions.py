class CoinCounterError(Exception):
    """Base public error."""


class InvalidImageError(CoinCounterError):
    pass


class UnavailableEngineError(CoinCounterError):
    pass


class ModelLoadingError(CoinCounterError):
    pass


class InferenceError(CoinCounterError):
    pass
