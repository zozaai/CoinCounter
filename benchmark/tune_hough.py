"""Training shortlist then validation selection; never reads the test split."""
import argparse
from itertools import product
from pathlib import Path

from .compute import ROOT
from .dataset import samples
from .cache import file_hash
from .results import write_json
from coincounter import CoinCounter
from coincounter.image import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hough-v1/tuning.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark/hough.yaml")
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)  # Tuning process only; does not change package runtime state.
    data = {split: [(normalize(path), actual) for path, actual in samples(args.dataset, split)]
            for split in ("train", "val")}
    training_subset = data["train"][::4]
    def score(parameters, split, subset=False):
        with CoinCounter("hough", parameters) as counter:
            errors = [abs(counter.run(image).count - actual) for image, actual in
                      (training_subset if subset else data[split])]
        return {"accuracy": errors.count(0) / len(errors), "mae": sum(errors) / len(errors)}
    candidates = []
    for edge, accumulator, radii in product([120, 200], [30, 45, 60, 75], [(10, 30), (12, 40), (15, 50)]):
        parameters = dict(dp=1.0, min_distance=24, min_radius=radii[0], max_radius=radii[1],
                          edge_threshold=edge, accumulator_threshold=accumulator, blur_size=5)
        result = {"parameters": parameters, "train": score(parameters, "train", subset=True)}
        candidates.append(result)
        print(f"Training candidate {len(candidates)}/24: {result['train']}", flush=True)
    candidates.sort(key=lambda r: (-r["train"]["accuracy"], r["train"]["mae"]))
    shortlist = candidates[:5]
    for result in shortlist:
        result["val"] = score(result["parameters"], "val")
    best = min(shortlist, key=lambda r: (-r["val"]["accuracy"], r["val"]["mae"],
                                        -r["train"]["accuracy"], r["train"]["mae"]))
    best["full_train"] = score(best["parameters"], "train")
    write_json(args.output, {"opencv_version": cv2.__version__, "opencv_threads": 1,
                            "selection": "24 candidates on sorted train images [::4]; top 5 on validation; accuracy then MAE",
                            "training_subset_size": len(training_subset),
                            "label_hashes": {s: file_hash(args.dataset / s / "labels.json") for s in data},
                            "selected": best, "candidates": candidates})
    write_json(args.config, best["parameters"])
    print(f"Selected: {best}")


if __name__ == "__main__":
    main()
