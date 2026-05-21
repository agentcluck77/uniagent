from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioBuffer:
    samples: Any
    sample_rate: int
