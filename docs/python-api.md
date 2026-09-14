# Python API

Install with `pip install -e .`. Import `CoinCounter` and `CountResult` from `coincounter`.
The implemented engine is `random_guess`; parameters are `counts`, positive integer
`weights`, and optional `seed` (42). Supply the training histogram from README or your own
training data. The runtime package does not read repository datasets.

`run` accepts image paths, PIL images, or uint8 RGB arrays. `close` is idempotent;
context-manager cleanup is supported. Results contain count, optional confidence,
optional detections, and metadata. A closed counter rejects inference.
