import hashlib
import json
import math


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_prediction(base, key):
    try:
        value = json.loads(base.with_suffix(".json").read_text())
        count = value["count"]
        seconds = value["seconds"]
        if (value["key"] == key and type(count) is int and count >= 0
                and isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0
                and base.with_suffix(".txt").read_text().strip() == str(count)):
            return value
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None
