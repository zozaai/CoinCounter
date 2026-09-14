# Benchmarking

Run `python -m benchmark.compute --dataset dataset --split test` from the repository.
Use `--engine`, `--force` (alias `--forece`), and `--sort time` as needed.
Configurations currently use the JSON subset of YAML; general YAML syntax is not supported.
The default engine list contains only `random_guess`.

The baseline samples the training-label empirical distribution using seed 42 and a hash
of the normalized image. Pixel content provides identity only, not coin evidence.
Predictions are independent of iteration order and partial cache hits. Test labels are
used only for scoring. Training-set scores are not held-out evaluation.

Timing includes decoding/normalization and inference, excludes hashing and persistence,
and is measured sequentially without a warmup. Cached runs reuse original timings.
Incomplete results sort after completed results and include errors. JSON/CSV summaries
are saved in reports; raw predictions in results. Reports are overwritten for the same
dataset selection; preserve a copy to retain an earlier comparison. Plotly is future work.
