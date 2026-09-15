"""Run from the repository: python -m benchmark.compute --dataset dataset --split test."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import time

# Allow repository benchmarking before an editable package installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coincounter import CoinCounter
from .cache import cached_prediction, file_hash, identity
from .dataset import labels_at, samples
from .metrics import summarize
from .reporting import save_summary, table
from .results import save_prediction, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--image", type=Path)
    parser.add_argument("--engine", nargs="+")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs/benchmark")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    parser.add_argument("--force", "--forece", action="store_true")
    parser.add_argument("--sort", choices=["accuracy", "time"], default="accuracy")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.image and args.split:
        parser.error("--image and --split cannot be combined")
    try:
        engines = args.engine or json.loads((args.config_dir / "default.yaml").read_text())["engines"]
        if not engines or any(not isinstance(e, str) or not e.isidentifier() for e in engines):
            raise ValueError("Engine list must contain valid engine names")
        engines = list(dict.fromkeys(engines))
        items = samples(args.dataset, args.split, args.image)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "machine": platform.machine(), "processor": platform.processor(),
                   "numpy": version("numpy"), "pillow": version("Pillow")}
    source_hash = identity({str(p.relative_to(ROOT)): file_hash(p)
                            for folder in [ROOT / "src/coincounter", ROOT / "benchmark"]
                            for p in sorted(folder.rglob("*.py"))})
    dataset_id = identity(str(args.dataset.resolve()))[:12]
    selection = "single" if args.image else args.split or "all"
    rows = []
    for name in engines:
        records, errors = [], []
        loading = 0.0
        config_id = None
        try:
            parameters = json.loads((args.config_dir / f"{name}.yaml").read_text())
            if name == "random_guess":
                histogram = Counter(labels_at(args.dataset / "train").values())
                parameters.update(counts=sorted(histogram), weights=[histogram[c] for c in sorted(histogram)])
            manifest = {"engine": name, "parameters": parameters, "source_hash": source_hash,
                        "environment": environment,
                        "timing_scope": "CoinCounter.run: decode/normalize plus inference; excludes file hashing and persistence"}
            if "model_path" in parameters:
                weights = Path(parameters["model_path"])
                if not weights.is_file():
                    from coincounter.exceptions import ModelLoadingError
                    raise ModelLoadingError(f"Model weights not found: {weights}")
                manifest["weights_sha256"] = file_hash(weights)
            if name == "hough":
                from coincounter.exceptions import UnavailableEngineError
                try:
                    import cv2
                except ImportError as exc:
                    raise UnavailableEngineError("Install the Hough extra: pip install 'coincounter[hough]'") from exc
                manifest["opencv"] = {"version": cv2.__version__, "threads": cv2.getNumThreads()}
            config_id = identity(manifest)[:16]
            destination = args.results / name / config_id
            write_json(destination / "manifest.json", manifest)
            tick = time.perf_counter()
            counter = CoinCounter(name, parameters)
            loading = time.perf_counter() - tick
            with counter:
                for path, actual in items:
                    try:
                        image_hash = file_hash(path)
                        key = identity({"configuration": config_id, "image": image_hash})
                        # Include extension and content identity to prevent filename collisions.
                        base = destination / dataset_id / selection / f"{path.name}-{image_hash[:12]}.prediction"
                        record = None if args.force else cached_prediction(base, key)
                        reused = record is not None
                        if record is None:
                            tick = time.perf_counter()
                            result = counter.run(path)
                            seconds = time.perf_counter() - tick
                            record = {**asdict(result), "key": key, "image": str(path),
                                      "image_hash": image_hash, "seconds": seconds}
                            save_prediction(base, record)
                        records.append({**record, "actual": actual, "cached": reused})
                    except Exception as exc:
                        errors.append({"image": str(path), "error": str(exc)})
        except Exception as exc:
            errors.append({"engine": name, "error": str(exc)})
        row = {"engine": name, "configuration_id": config_id, **summarize(records, len(items)),
               "loading_seconds": loading, "errors": errors}
        rows.append(row)
        if args.image and len(engines) == 1 and records:
            print(records[0]["count"])
        for error in errors:
            print(f"{name}: {error}", file=sys.stderr)
    summary = {"dataset": str(args.dataset.resolve()), "split": selection,
               "environment": environment, "source_hash": source_hash,
               "elapsed_seconds": time.perf_counter() - started, "engines": rows}
    save_summary(args.reports / dataset_id / selection, summary)
    stream = sys.stderr if args.image and len(engines) == 1 else sys.stdout
    if not args.image or len(engines) > 1:
        print(table(rows, args.sort), file=stream)
    print("Timing includes image decoding/normalization and inference; cached timings are recorded measurements.", file=stream)
    for row in rows:
        print(f"{row['engine']} loading: {row['loading_seconds']:.6f}s", file=stream)
    print(f"Command elapsed: {summary['elapsed_seconds']:.6f}s", file=stream)
    return 1 if any(row["errors"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
