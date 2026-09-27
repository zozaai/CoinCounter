# Benchmarking

Run `python -m benchmark.compute --dataset dataset --split test` from the repository.
Use `--engine`, `--force` (alias `--forece`), and `--sort time` as needed.
Configurations currently use the JSON subset of YAML; general YAML syntax is not supported.
The default engine list contains `random_guess` and `hough`. Install `pip install -e '.[benchmark,hough]'`.
Missing OpenCV is reported as an engine failure; the random baseline remains usable alone.

The baseline samples the training-label empirical distribution using seed 42 and a hash
of the normalized image. Pixel content provides identity only, not coin evidence.
Predictions are independent of iteration order and partial cache hits. Test labels are
used only for scoring. Training-set scores are not held-out evaluation.

Timing includes decoding/normalization and inference, excludes hashing and persistence,
and is measured sequentially without a warmup. Cached runs reuse original timings.
Incomplete results sort after completed results and include errors. Each run saves:

- `results/<split>_summary.json`: all table metrics, including `scored`,
  `expected`, `accuracy` (a fraction from 0 to 1), `mae`, `mean_ms`, `total_seconds`,
  `cached`, and `failed`, plus errors, loading time, and run provenance.
- `results/<split>_accuracy_vs_time.html`: an interactive Plotly chart with
  processing time (Mean ms) on the x-axis, accuracy displayed as a percentage on the
  y-axis, and a labeled circle for each model. Hover for the table metrics. A line joins
  Pareto-optimal models in increasing time order: no other eligible model is at least
  as fast and as accurate with a strict improvement in one metric. Colored guide lines
  extend each model to both axes, where ticks and labels show its exact time and accuracy.
- `reports/<dataset-id>/<split>/summary.json` and `summary.csv`: the existing reports.

The command prints the JSON and HTML paths. Open the HTML directly in a browser; Plotly
is embedded, so no internet connection is needed. Partial results appear faded and are
excluded from the frontier; models without accuracy or timing data are listed below the
chart. Only fully scored, error-free runs are eligible for the frontier.

`--results` changes the predictions and comparison output root; `--reports` changes the
JSON/CSV report root. Raw predictions remain in `results/<engine>/<configuration>/`.
Outputs are overwritten for the same dataset selection; preserve a copy to retain an
earlier comparison. The selection is `all` without a split or `single` with `--image`.

For example, a test split writes `results/test_summary.json` and
`results/test_accuracy_vs_time.html`. Regenerate a comparison from a saved summary
without loading any models:

```python
import json
from pathlib import Path
from benchmark.reporting import save_comparison

folder = Path("results")
summary = json.loads((folder / "test_summary.json").read_text())
save_comparison(folder, summary, stem="test")
```

Hough tuning is reproducible with `python -m benchmark.tune_hough`. It evaluates 24
configurations on 21 training images (sorted filenames, every fourth image), evaluates the
five strongest on all 19 validation images, and selects by exact-count accuracy then MAE,
with training scores breaking ties. It also scores the selected configuration on full train.
The test split is never read during tuning. The early permissive-threshold exploratory sweep
was stopped after five training-only candidates because textured backgrounds produced many
false positives; its settings are not part of the final 24-candidate grid.

The selected settings use radii 10–30 px, spacing 24 px, edge threshold 200, accumulator
threshold 30, median kernel 5, and dp 1.0. Hough uses grayscale, median filtering, and
OpenCV HOUGH_GRADIENT, following the [OpenCV tutorial](https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html).
Tuning sets one OpenCV thread within its own process. Evaluation uses the runtime default;
the OpenCV version and thread count enter cache identity and prediction metadata. NumPy
and Pillow versions also enter cache identity. Timings are environment-specific.

Comparison reports for this implementation are saved separately with
`python -m benchmark.compute --dataset dataset --split test --force --reports reports/hough-v1`.

## Regression training

`python -m benchmark.train_regression --dataset dataset --out models/regression` fine-tunes
an ImageNet ResNet-18 (all layers) for 60 epochs on the train split with AdamW, one-cycle
LR peaking at 3e-4, Smooth-L1 loss on the sigmoid-bounded count, and flip / rot90 /
colour-jitter augmentation at 320×320. Each epoch is evaluated on val and the checkpoint
with the best exact accuracy then MAE is saved as `resnet18-320.pt`; `training.json` records
arguments, per-epoch metrics, and the selected epoch. The test split is never read.
Options: `--image-size`, `--max-count`, `--epochs`, `--lr`, `--batch`, `--seed`, `--device`.

Run the comparison with `python -m benchmark.compute --dataset dataset --split test --force
--engine hough random_guess regression --reports reports/regression-v1`. The manifest
includes the weights' SHA-256, so retraining invalidates the cache. Regression is not in
`default.yaml` because it requires the `regression` extra and a trained weights file.
The first MPS inference includes kernel warm-up, which inflates the mean on small splits.

## Grounding DINO

`grounding_dino` needs no training. The `box_threshold` of 0.4 was chosen by sweeping
0.20–0.50 on `train` (82 images) and `val` (19 images) with the tiny checkpoint: exact
accuracy plateaus from 0.30 to 0.45 on both splits and 0.40 and 0.45 tie at 98/101 correct.
The test split was not used. The manifest records the resolved model commit hash and the
transformers version. Expect ~0.5 s per image on an M4 and ~6 s of model loading; the
checkpoint (~660 MB) is fetched into the Hugging Face cache on first use. Run the full
comparison with `python -m benchmark.compute --dataset dataset --split test --force --engine
grounding_dino regression hough random_guess --reports reports/grounding-dino-v1`.
Set `COINCOUNTER_TEST_GROUNDING_DINO=1` to include the real-model unit test.

## Vision LLM

`vision_llm` queries `gemini-3.8-flash` on the Gemini API free tier with the fixed prompt in
`vision_llm.py` at temperature 0; nothing is tuned, so no split is read except for scoring.
Export `GEMINI_API_KEY` before running. The manifest records the API endpoint and requested
model, and each prediction keeps the raw reply and the model the API reports. Timing includes
network latency, model thinking, and retries. The free tier is rate limited, so run one split at
a time; 429 and busy-model 503 responses are retried, honouring `Retry-After`. Any source change
alters the cache identity, so rerunning after an edit queries the API again. On `test` it scored
15/16 (MAE 0.062) at ~8.0 s per image. Run the comparison with
`python -m benchmark.compute --dataset dataset --split test --force --engine vision_llm
grounding_dino regression hough random_guess --reports reports/vision-llm-v1`.
Set `COINCOUNTER_TEST_VISION_LLM=1` (with the key) to include the real-API unit test.

## Viewing saved predictions

`python -m benchmark.vis --dataset dataset --split test --engine hough` serves a local page
(default `http://127.0.0.1:8765`) that pages through the selected images. Use `--image` to
start at a specific file and `--no-browser` to skip opening a tab. The viewer reads
`results/<engine>/<configuration>/` only: it never imports engines, loads models, or runs
inference, so a missing prediction shows a message rather than a fresh count. Detection
overlays come from the saved `detections` list. Grayscale and blur panels are display
reconstructions using Pillow and the manifest `blur_size`; they are not recorded engine
output. Configurations are listed newest first and selectable when more than one exists.
