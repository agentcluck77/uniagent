from __future__ import annotations

from typing import Any

import numpy as np

from uniagent.speech.errors import SpeechConfigurationError
from uniagent.speech.types import AudioBuffer


class SileroVAD:
    def __init__(self, config: dict[str, Any], sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate
        self.enabled = bool(config["enabled"])
        self._model: Any = None
        self._iterator: Any = None
        self._get_speech_timestamps: Any = None

    def reset(self) -> None:
        if self.enabled:
            self.load()
            self._iterator.reset_states()

    def event(self, samples: Any) -> dict | None:
        self.load()
        return self._iterator(self._chunk(samples), return_seconds=True)

    def extract(self, audio: AudioBuffer) -> AudioBuffer:
        if not self.enabled:
            return audio
        self.load()
        timestamps = self._get_speech_timestamps(
            self._chunk(audio.samples),
            self._model,
            threshold=float(self.config["threshold"]),
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=int(self.config["silence_duration_ms"]),
            max_speech_duration_s=float(self.config["max_recording_s"]),
            speech_pad_ms=30,
        )
        if not timestamps:
            return audio

        chunks = [audio.samples[int(item["start"]) : int(item["end"])] for item in timestamps]
        return AudioBuffer(samples=np.concatenate(chunks), sample_rate=audio.sample_rate)

    def load(self) -> None:
        if self._model is not None and self._iterator is not None:
            return
        if self.config["model"] != "silero_v5":
            raise SpeechConfigurationError(f"unsupported VAD model: {self.config['model']}")
        from silero_vad import (  # type: ignore[import-not-found]
            VADIterator,
            get_speech_timestamps,
            load_silero_vad,
        )

        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        self._iterator = VADIterator(
            self._model,
            threshold=float(self.config["threshold"]),
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=int(self.config["silence_duration_ms"]),
            speech_pad_ms=30,
        )

    def _chunk(self, samples: Any) -> Any:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return samples
        return getattr(torch, "from_numpy")(samples)
