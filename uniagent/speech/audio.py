from __future__ import annotations

import math
import queue
import subprocess
import time
from collections import deque
from threading import Event
from typing import Any

import numpy as np

from uniagent.speech.errors import SpeechError
from uniagent.speech.types import AudioBuffer


class AudioIO:
    def __init__(self, config: dict[str, Any], sample_rate: int, tts_playing: Event):
        self.config = config
        self.sample_rate = sample_rate
        self._tts_playing = tts_playing
        self._play_proc: subprocess.Popen | None = None

    def record_utterance(
        self, vad: Any, abort_event: Event | None = None
    ) -> AudioBuffer:
        if not vad.enabled:
            raise SpeechError("VAD must be enabled for live speech recording.")

        import sounddevice as sd  # type: ignore[import-not-found]

        vad.reset()
        audio_queue: queue.Queue[Any] = queue.Queue()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status and not status.input_overflow:
                audio_queue.put(RuntimeError(str(status)))
            else:
                audio_queue.put(indata.copy())

        block_size = 512 if self.sample_rate == 16000 else 256
        max_seconds = float(vad.config["max_recording_s"])
        pre_chunks: deque[Any] = deque(maxlen=max(1, int(0.3 * self.sample_rate / block_size)))
        recorded: list[Any] = []
        started_at: float | None = None

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            device=self.config["input_device"],
            callback=callback,
        ):
            while True:
                if abort_event is not None and abort_event.is_set():
                    break
                try:
                    chunk = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(chunk, BaseException):
                    raise SpeechError(f"audio input failed: {chunk}") from chunk
                if self._tts_playing.is_set():
                    continue

                mono = self.as_float32_mono(chunk)
                event = vad.event(mono)
                if started_at is None:
                    pre_chunks.append(mono)
                    if event and "start" in event:
                        started_at = time.monotonic()
                        recorded.extend(pre_chunks)
                        pre_chunks.clear()
                else:
                    recorded.append(mono)
                    if event and "end" in event:
                        break
                    if time.monotonic() - started_at >= max_seconds:
                        break

        vad.reset()
        samples = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate)

    def play(self, audio: AudioBuffer) -> None:
        samples = self.apply_output_volume(audio.samples)
        already_playing = self._tts_playing.is_set()
        self._tts_playing.set()
        try:
            self._paplay(samples, audio.sample_rate)
        finally:
            if not already_playing:
                self._tts_playing.clear()

    def cancel_play(self) -> None:
        proc = self._play_proc
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._tts_playing.clear()

    def record_until(self, stop_event: Event) -> AudioBuffer:
        import sounddevice as sd  # type: ignore[import-not-found]

        audio_queue: queue.Queue[Any] = queue.Queue()

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info
            if status and not status.input_overflow:
                audio_queue.put(RuntimeError(str(status)))
            else:
                audio_queue.put(indata.copy())

        block_size = 512 if self.sample_rate == 16000 else 256
        recorded: list[Any] = []

        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=block_size,
            channels=1,
            dtype="float32",
            device=self.config["input_device"],
            callback=callback,
        ):
            while stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if isinstance(chunk, BaseException):
                    raise SpeechError(f"audio input failed: {chunk}") from chunk
                if self._tts_playing.is_set():
                    continue
                recorded.append(self.as_float32_mono(chunk))

        samples = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.float32)
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate)

    def prepare_input(self, audio: AudioBuffer | Any) -> AudioBuffer:
        if isinstance(audio, AudioBuffer):
            samples = self.as_float32_mono(audio.samples)
            sample_rate = audio.sample_rate
        else:
            samples = self.as_float32_mono(audio)
            sample_rate = self.sample_rate
        if sample_rate == self.sample_rate:
            return AudioBuffer(samples=samples, sample_rate=sample_rate)
        return self.resample(
            AudioBuffer(samples=samples, sample_rate=sample_rate),
            self.sample_rate,
        )

    def resample(self, audio: AudioBuffer, target_rate: int) -> AudioBuffer:
        if audio.sample_rate == target_rate or audio.samples.size == 0:
            return audio
        duration_s = audio.samples.size / float(audio.sample_rate)
        target_size = max(1, int(round(duration_s * target_rate)))
        source_x = np.linspace(0.0, duration_s, num=audio.samples.size, endpoint=False)
        target_x = np.linspace(0.0, duration_s, num=target_size, endpoint=False)
        samples = np.interp(target_x, source_x, audio.samples).astype(np.float32)
        return AudioBuffer(samples=samples, sample_rate=target_rate)

    def as_float32_mono(self, samples: Any) -> Any:
        array = np.asarray(samples)
        if array.ndim > 1:
            array = array.mean(axis=1)
        if array.dtype.kind in {"i", "u"}:
            max_value = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
            array = array.astype(np.float32) / float(max_value)
        else:
            array = array.astype(np.float32)
        return np.clip(array.reshape(-1), -1.0, 1.0)

    def apply_output_volume(self, samples: Any) -> Any:
        return np.clip(
            np.asarray(samples, dtype=np.float32) * self.config["output_volume"],
            -1.0,
            1.0,
        )

    def builtin_chime(self, frequency: float) -> AudioBuffer:
        sample_rate = 16000
        t = np.arange(int(sample_rate * 0.12), dtype=np.float32) / sample_rate
        envelope = np.linspace(1.0, 0.15, t.size, dtype=np.float32)
        samples = 0.18 * envelope * np.sin(2 * math.pi * frequency * t)
        return AudioBuffer(samples=samples.astype(np.float32), sample_rate=sample_rate)


    def _paplay(self, samples: Any, sample_rate: int) -> None:
        proc = subprocess.Popen(
            ["paplay", "--raw", "--format=float32le", f"--rate={sample_rate}", "--channels=1"],
            stdin=subprocess.PIPE,
        )
        self._play_proc = proc
        try:
            proc.communicate(input=samples.astype("<f4").tobytes())
        except BrokenPipeError:
            pass
        finally:
            self._play_proc = None
