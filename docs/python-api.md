# Python API

Install with `pip install -e .`. Import `CoinCounter` and `CountResult` from `coincounter`.
For `random_guess`, parameters are `counts`, positive integer
`weights`, and optional `seed` (42). Supply the training histogram from README or your own
training data. The runtime package does not read repository datasets.

Install `pip install -e '.[hough]'` for the Hough engine. `CoinCounter("hough")`
uses the selected dataset defaults; override them through `parameters`.
Parameters: `dp=1.0`, `min_distance=24`, `min_radius=10`, `max_radius=30`,
`edge_threshold=200`, `accumulator_threshold=30`, `blur_size=5`.
Radii and minimum spacing are in input pixels, so adjust them for other image scales.
Detections contain `x`, `y`, and `radius` in original image coordinates.
Hough confidence is `None`; the accumulator threshold is not a calibrated probability.

`run` accepts image paths, PIL images, or uint8 RGB arrays. `close` is idempotent;
context-manager cleanup is supported. Results contain count, optional confidence,
optional detections, and metadata. A closed counter rejects inference.
