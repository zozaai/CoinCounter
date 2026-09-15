from importlib import import_module

from ..exceptions import UnavailableEngineError

ENGINES = {"random_guess": ("random_guess", "RandomGuessEngine"),
           "hough": ("hough", "HoughEngine"),
           "regression": ("regression", "RegressionEngine")}


def create_engine(name, parameters):
    if name not in ENGINES:
        raise UnavailableEngineError(f"Unknown or unimplemented engine {name!r}; available: {', '.join(ENGINES)}")
    module, class_name = ENGINES[name]
    cls = getattr(import_module(f"coincounter.engines.{module}"), class_name)
    return cls(**parameters)
