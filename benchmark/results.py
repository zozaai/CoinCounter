import json
import os
import tempfile
from pathlib import Path


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as target:
            target.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_prediction(base, record):
    atomic_write(base.with_suffix(".txt"), f"{record['count']}\n")
    write_json(base.with_suffix(".json"), record)
