"""Learned count regression: ImageNet ResNet-18 with one sigmoid-bounded output in [0, max_count]."""
import hashlib
import math
from pathlib import Path

from ..exceptions import ModelLoadingError, UnavailableEngineError
from ..types import CountResult
from .base import BaseEngine

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _torch():
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise UnavailableEngineError("Install the regression extra: pip install 'coincounter[regression]'") from exc
    return torch, torchvision


def build_model(pretrained=False):
    """ResNet-18 trunk with a single linear output. Shared by training and inference."""
    torch, torchvision = _torch()
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = torch.nn.Linear(net.fc.in_features, 1)
    return net


def bounded_count(logits, max_count):
    """Map raw network output to a count in [0, max_count]."""
    torch, _ = _torch()
    return max_count * torch.sigmoid(logits).squeeze(1)


class RegressionEngine(BaseEngine):
    def __init__(self, model_path, image_size=320, max_count=20, device="auto"):
        if type(image_size) is not int or image_size < 32:
            raise ValueError("image_size must be an integer >= 32")
        if isinstance(max_count, bool) or not isinstance(max_count, (int, float)) or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("max_count must be a finite positive number")
        torch, _ = _torch()
        self.torch = torch
        self.image_size = image_size
        self.max_count = float(max_count)
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadingError(f"Model weights not found: {path}")
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model = build_model(pretrained=False)
            self.model.load_state_dict(state)
        except Exception as exc:
            raise ModelLoadingError(f"Cannot load regression weights from {path}: {exc}") from exc
        self.model.to(self.device).eval()
        self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def preprocess(self, image):
        """uint8 RGB array -> normalized float tensor [1, 3, S, S]."""
        torch = self.torch
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(tensor, size=(self.image_size, self.image_size),
                                                 mode="bilinear", antialias=True, align_corners=False)
        return (tensor - self.mean) / self.std

    def run(self, image):
        with self.torch.inference_mode():
            value = bounded_count(self.model(self.preprocess(image).to(self.device)), self.max_count)
        raw = float(value.item())
        count = int(min(max(round(raw), 0), int(self.max_count)))
        return CountResult(count=count, metadata={"raw_count": raw, "max_count": self.max_count,
                                                  "image_size": self.image_size, "device": str(self.device),
                                                  "weights_sha256": self.weights_sha256,
                                                  "torch_version": self.torch.__version__})

    def close(self):
        self.model = None
