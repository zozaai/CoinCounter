import json
from pathlib import Path


def labels_at(root):
    labels = json.loads((Path(root) / "labels.json").read_text())
    if not labels or any(type(v) is not int or v < 0 for v in labels.values()):
        raise ValueError("Labels must map filenames to nonnegative integers")
    for name in labels:
        if Path(name).name != name:
            raise ValueError("Labels must use filenames without directories")
    return labels


def samples(root, split=None, image=None):
    folder = Path(root) / split if split else Path(root)
    if image:
        path = Path(image)
        return [(path, None)]
    labels = labels_at(folder)
    return [(folder / "images" / name, count) for name, count in sorted(labels.items())]
