from __future__ import annotations

from typing import Any

import numpy as np

from uniagent.speech.errors import SpeechConfigurationError
from uniagent.speech.types import AudioBuffer


class WhisperSTT:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._model: Any = None

    def transcribe(self, audio: AudioBuffer) -> str:
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            return ""
        if self.config["backend"] != "whisper.cpp":
            raise SpeechConfigurationError(f"unsupported STT backend: {self.config['backend']}")

        kwargs: dict[str, Any] = {}
        if self.config["language"]:
            kwargs["language"] = self.config["language"]
        segments = self.load().transcribe(samples, **kwargs)
        return " ".join(str(getattr(segment, "text", "")).strip() for segment in segments).strip()

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        from pywhispercpp.model import Model  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {"print_progress": False, "print_realtime": False}
        if self.config.get("models_dir") is not None:
            kwargs["models_dir"] = self.config["models_dir"]
        self._model = Model(self.config["model"], **kwargs)
        return self._model
