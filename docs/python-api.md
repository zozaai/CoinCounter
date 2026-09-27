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

Install `pip install -e '.[regression]'` for the regression engine. `CoinCounter("regression",
{"model_path": "models/regression/resnet18-320.pt"})` loads the weights once. Parameters:
`model_path` (required), `image_size=320`, `max_count=20`, `device="auto"` (MPS, then CUDA,
then CPU). The count is `round(max_count * sigmoid(output))`, so it always lies in
`[0, max_count]`. Confidence is `None`; `metadata.raw_count` is the unrounded prediction.
A missing or incompatible file raises `ModelLoadingError`; missing PyTorch raises
`UnavailableEngineError`.

Install `pip install -e '.[grounding-dino]'` for the zero-shot detector.
`CoinCounter("grounding_dino")` downloads `IDEA-Research/grounding-dino-tiny` (~660 MB) on
first use and reuses the Hugging Face cache afterwards. Parameters: `model`, `prompt="coin."`,
`box_threshold=0.4`, `text_threshold=0.25`, `device="auto"`, `revision=None`,
`local_files_only=False`. The count is the number of boxes above `box_threshold`; detections
are `{x1, y1, x2, y2, score}` in input pixels. Confidence is `None`. Loading failures raise
`ModelLoadingError`; missing transformers raises `UnavailableEngineError`.

No extra is needed for `vision_llm`; set `GEMINI_API_KEY` to a free key from
<https://aistudio.google.com/apikey>. `CoinCounter("vision_llm")` sends each image to
`gemini-3.8-flash` on the Gemini API free tier and parses `{"count": n}` from the reply. Parameters: `model`,
`prompt`, `max_image_side=1024`, `timeout=120`, `max_retries=5`. Confidence is `None`;
`metadata` contains `reply`, `resolved_model`, `usage`, and `sent_size`. A missing
key raises `UnavailableEngineError`; HTTP, network, and unreadable replies raise `InferenceError`.

`run` accepts image paths, PIL images, or uint8 RGB arrays. `close` is idempotent;
context-manager cleanup is supported. Results contain count, optional confidence,
optional detections, and metadata. A closed counter rejects inference.
