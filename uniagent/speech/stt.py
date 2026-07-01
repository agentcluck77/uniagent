from __future__ import annotations

import threading
from typing import Any

import numpy as np

from uniagent.speech.errors import SpeechConfigurationError
from uniagent.speech.types import AudioBuffer


class WhisperSTT:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._model: Any = None
        self._lock = threading.Lock()

    def transcribe(self, audio: AudioBuffer) -> str:
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            return ""
        if self.config["backend"] != "whisper.cpp":
            raise SpeechConfigurationError(f"unsupported STT backend: {self.config['backend']}")

        kwargs: dict[str, Any] = {}
        if self.config["language"]:
            kwargs["language"] = self.config["language"]
        with self._lock:
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


class MoonshineSTT:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._model: Any = None
        self._lock = threading.Lock()

    def transcribe(self, audio: AudioBuffer) -> str:
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            return ""
        with self._lock:
            transcript = self.load().transcribe_without_streaming(
                samples.tolist(), sample_rate=audio.sample_rate
            )
        return " ".join(line.text for line in transcript.lines).strip()

    def load(self) -> Any:
        if self._model is not None:
            return self._model
        from moonshine_voice import Transcriber  # type: ignore[import-not-found]
        from moonshine_voice.moonshine_api import string_to_model_arch  # type: ignore[import-not-found]

        model_name = self.config.get("model", "medium")
        if not model_name.endswith("-streaming"):
            model_name = f"{model_name}-streaming"
        model_arch = string_to_model_arch(model_name)

        from pathlib import Path
        from moonshine_voice import get_model_for_language  # type: ignore[import-not-found]

        cache_root = Path(self.config["models_dir"]).resolve() if self.config.get("models_dir") else None
        model_path, model_arch = get_model_for_language(
            wanted_language=self.config.get("language", "en"),
            wanted_model_arch=model_arch,
            cache_root=cache_root,
        )
        self._model = Transcriber(model_path=model_path, model_arch=model_arch)
        return self._model


def make_stt(config: dict[str, Any]) -> WhisperSTT | MoonshineSTT:
    backend = config.get("backend", "whisper.cpp")
    if backend == "whisper.cpp":
        return WhisperSTT(config)
    if backend == "moonshine-streaming":
        return MoonshineSTT(config)
    raise SpeechConfigurationError(f"unsupported STT backend: {backend}")
