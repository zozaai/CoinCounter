from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CountResult:
    count: int
    confidence: Optional[float] = None
    detections: Optional[list] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.count) is not int or self.count < 0:
            raise ValueError("count must be a nonnegative integer")
