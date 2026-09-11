#!/usr/bin/env python3
"""Split the labeled 480x480 images into train/val/test.

Usage:
    python3 tools/split_dataset.py [--ratios 0.7 0.15 0.15] [--seed 42] [--move]

Reads dataset/labels.json, then writes:
    dataset/train/images/*.jpg  + dataset/train/labels.json
    dataset/val/...  dataset/test/...
Splits are stratified by coin count so each class is spread across the splits.
"""
import argparse
import json
import os
import random
import shutil
from collections import defaultdict

SPLITS = ("train", "val", "test")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(here, "dataset")
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join(root, "images"))
    ap.add_argument("--labels", default=os.path.join(root, "labels.json"))
    ap.add_argument("--out", default=root)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    args = ap.parse_args()

    with open(args.labels) as fh:
        labels = json.load(fh)
    if not labels:
        raise SystemExit("labels.json is empty - run tools/label_tool.py first")

    missing = [f for f in labels if not os.path.isfile(os.path.join(args.images, f))]
    if missing:
        raise SystemExit(f"{len(missing)} labeled files missing from {args.images}: {missing[:5]}")

    # stratify by coin count
    by_count = defaultdict(list)
    for f, c in labels.items():
        by_count[c].append(f)

    rng = random.Random(args.seed)
    assigned = {s: [] for s in SPLITS}
    r_train, r_val, _ = args.ratios
    for count in sorted(by_count):
        files = sorted(by_count[count])
        rng.shuffle(files)
        n = len(files)
        n_tr = round(n * r_train)
        n_va = round(n * r_val)
        if n >= 3:  # guarantee at least one item per split when possible
            n_tr = max(1, min(n_tr, n - 2))
            n_va = max(1, min(n_va, n - n_tr - 1))
        assigned["train"] += files[:n_tr]
        assigned["val"] += files[n_tr:n_tr + n_va]
        assigned["test"] += files[n_tr + n_va:]

    for split in SPLITS:
        img_dir = os.path.join(args.out, split, "images")
        if os.path.isdir(os.path.join(args.out, split)):
            shutil.rmtree(os.path.join(args.out, split))
        os.makedirs(img_dir, exist_ok=True)
        split_labels = {}
        for f in sorted(assigned[split]):
            src = os.path.join(args.images, f)
            dst = os.path.join(img_dir, f)
            (shutil.move if args.move else shutil.copy2)(src, dst)
            split_labels[f] = labels[f]
        with open(os.path.join(args.out, split, "labels.json"), "w") as fh:
            json.dump(split_labels, fh, indent=2, sort_keys=True)
        counts = sorted(set(split_labels.values()))
        print(f"{split:5s}: {len(split_labels):4d} images  coin counts {counts}")

    total = sum(len(assigned[s]) for s in SPLITS)
    print(f"total: {total} labeled images -> {args.out}")


if __name__ == "__main__":
    main()
