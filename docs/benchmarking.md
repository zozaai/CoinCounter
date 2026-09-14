# Benchmarking

Run `python -m benchmark.compute --dataset dataset --split test` from the repository.
Use `--engine`, `--force` (alias `--forece`), and `--sort time` as needed.
Configurations currently use the JSON subset of YAML; general YAML syntax is not supported.
The default engine list contains `random_guess` and `hough`. Install `pip install -e '.[hough]'`.
Missing OpenCV is reported as an engine failure; the random baseline remains usable alone.

The baseline samples the training-label empirical distribution using seed 42 and a hash
of the normalized image. Pixel content provides identity only, not coin evidence.
Predictions are independent of iteration order and partial cache hits. Test labels are
used only for scoring. Training-set scores are not held-out evaluation.

Timing includes decoding/normalization and inference, excludes hashing and persistence,
and is measured sequentially without a warmup. Cached runs reuse original timings.
Incomplete results sort after completed results and include errors. JSON/CSV summaries
are saved in reports; raw predictions in results. Reports are overwritten for the same
dataset selection; preserve a copy to retain an earlier comparison. Plotly is future work.

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

## Viewing saved predictions

`python -m benchmark.vis --dataset dataset --split test --engine hough` serves a local page
(default `http://127.0.0.1:8765`) that pages through the selected images. Use `--image` to
start at a specific file and `--no-browser` to skip opening a tab. The viewer reads
`results/<engine>/<configuration>/` only: it never imports engines, loads models, or runs
inference, so a missing prediction shows a message rather than a fresh count. Detection
overlays come from the saved `detections` list. Grayscale and blur panels are display
reconstructions using Pillow and the manifest `blur_size`; they are not recorded engine
output. Configurations are listed newest first and selectable when more than one exists.
