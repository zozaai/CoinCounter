"""Zero-shot open-vocabulary detection with Grounding DINO; the count is the number of boxes above threshold."""
import math

from ..exceptions import ModelLoadingError, UnavailableEngineError
from ..types import CountResult
from .base import BaseEngine

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def _imports():
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise UnavailableEngineError("Install the Grounding DINO extra: pip install 'coincounter[grounding-dino]'") from exc
    return torch, AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDinoEngine(BaseEngine):
    def __init__(self, model=DEFAULT_MODEL, prompt="coin.", box_threshold=0.4, text_threshold=0.25,
                 device="auto", revision=None, local_files_only=False):
        for name, value in {"box_threshold": box_threshold, "text_threshold": text_threshold}.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number in [0, 1]")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a Hugging Face model id or local path")
        torch, AutoProcessor, AutoModel = _imports()
        self.torch = torch
        # Grounding DINO expects lower-case phrases terminated by a period.
        self.prompt = prompt.strip().lower()
        if not self.prompt.endswith("."):
            self.prompt += "."
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        options = {"revision": revision, "local_files_only": local_files_only}
        try:
            self.processor = AutoProcessor.from_pretrained(model, **options)
            self.model = AutoModel.from_pretrained(model, **options).to(self.device).eval()
        except Exception as exc:
            raise ModelLoadingError(f"Cannot load Grounding DINO model {model!r}: {exc}") from exc
        self.model_id = model
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def run(self, image):
        height, width = image.shape[:2]
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(height, width)])[0]
        detections = [
            {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "score": float(score)}
            for (x1, y1, x2, y2), score in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist())
        ]
        detections.sort(key=lambda box: (box["y1"], box["x1"]))
        return CountResult(count=len(detections), detections=detections,
                           metadata={"model": self.model_id, "revision": self.revision, "prompt": self.prompt,
                                     "box_threshold": self.box_threshold, "text_threshold": self.text_threshold,
                                     "device": str(self.device), "coordinate_system": "input RGB image pixels, xyxy"})

    def close(self):
        self.model = None
        self.processor = None
