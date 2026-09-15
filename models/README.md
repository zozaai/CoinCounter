# Model weights

Random guess and Hough require no weights. Large weights are excluded from Git and package
builds; only training logs are versioned.

## regression/resnet18-320.pt

| Field | Value |
|---|---|
| Architecture | torchvision `resnet18`, `fc` replaced by `Linear(512, 1)`; count = `20 * sigmoid(out)` |
| Initialisation | `ResNet18_Weights.IMAGENET1K_V1` (BSD-3, torchvision) |
| Training data | `dataset/train` (82 images), selected on `dataset/val` (19 images) |
| Recipe | `python -m benchmark.train_regression --dataset dataset --out models/regression` (seed 42) |
| Framework | PyTorch 2.14.0 / torchvision 0.29.0, Apple M4 (MPS) |
| Size | 44,786,251 bytes |
| SHA-256 | `1f53a6f10e8a9b5e96fd36a6a7a46d712e81b5f5cae04e1fd8468fecaa07def3` |
| Validation | 78.95% exact accuracy, 0.211 MAE (epoch 52 of 60) |
| Test | 56.25% exact accuracy, 0.438 MAE |

The full epoch history is in [regression/training.json](regression/training.json).
Regenerate the weights with the recipe above; MPS/CUDA kernels are not bit-deterministic,
so a retrained file will have a different checksum and close but not identical metrics.
`benchmark.compute` records the weights' SHA-256 in each run's manifest.
