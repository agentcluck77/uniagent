from __future__ import annotations

import queue
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

from uniagent.speech.audio import AudioIO
from uniagent.speech.errors import SpeechConfigurationError
from uniagent.speech.types import AudioBuffer

_MD_PATTERNS = [
    (re.compile(r"```.*?```", re.DOTALL), ""),           # fenced code blocks
    (re.compile(r"`(.+?)`"), r"\1"),                     # inline code
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1"),   # **bold**
    (re.compile(r"\*(.+?)\*", re.DOTALL), r"\1"),        # *italic*
    (re.compile(r"__(.+?)__", re.DOTALL), r"\1"),        # __bold__
    (re.compile(r"_(.+?)_", re.DOTALL), r"\1"),          # _italic_
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),       # # headers
    (re.compile(r"\[(.+?)\]\(.+?\)", re.DOTALL), r"\1"), # [text](url)
    (re.compile(r"^[-*+]\s+", re.MULTILINE), ""),        # - bullets
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),        # 1. numbered lists
    (re.compile(r"^>\s+", re.MULTILINE), ""),            # > blockquotes
]


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MD_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


class PiperTTS:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._voice: Any = None

    def synthesize(self, text: str) -> AudioBuffer:
        chunks = list(self.load().synthesize(_strip_markdown(text), syn_config=self._synthesis_config()))
        if not chunks:
            return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=16000)
        sample_rate = chunks[0].sample_rate
        samples = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        return AudioBuffer(samples=samples, sample_rate=sample_rate)

    def load(self) -> Any:
        if self._voice is not None:
            return self._voice
        if self.config["backend"] != "piper":
            raise SpeechConfigurationError(f"unsupported TTS backend: {self.config['backend']}")

        model_path = self.config["model_path"]
        if not model_path:
            raise SpeechConfigurationError("set tts.model_path to a Piper .onnx voice model")
        if not Path(model_path).exists():
            raise SpeechConfigurationError(f"Piper model file does not exist: {model_path}")

        try:
            from piper import PiperVoice  # type: ignore[import-not-found]
        except ImportError:
            from piper.voice import PiperVoice  # type: ignore[import-not-found]

        self._voice = PiperVoice.load(model_path)
        return self._voice

    def _synthesis_config(self) -> Any:
        from piper.config import SynthesisConfig  # type: ignore[import-not-found]

        voice = self.config["voice"]
        speaker_id = int(voice) if isinstance(voice, str) and voice.isdigit() else voice
        speaker_id = speaker_id if isinstance(speaker_id, int) else None
        speaking_rate = float(self.config["speaking_rate"])
        if speaking_rate <= 0:
            raise SpeechConfigurationError("tts.speaking_rate must be greater than 0")
        return SynthesisConfig(speaker_id=speaker_id, length_scale=1.0 / speaking_rate)


class SpeechOutput:
    def __init__(
        self,
        tts: PiperTTS,
        audio: AudioIO,
        chime_config: dict[str, Any],
        tts_playing: threading.Event,
    ):
        self.tts = tts
        self.audio = audio
        self.chime_config = chime_config
        self._tts_playing = tts_playing
        self._queue: queue.Queue[tuple[str, str | None, threading.Event | None]] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def speak(self, text: str) -> None:
        sentences = self._sentence_parts(text, force=True)[0]
        if not sentences:
            return
        self._tts_playing.set()
        try:
            for sentence in sentences:
                self._play_text(sentence)
            self.play_chime("end")
        finally:
            self._tts_playing.clear()

    def make_on_event(self) -> Any:
        buffer: list[str] = []
        finished = False

        def flush(force: bool = False) -> None:
            text = "".join(buffer)
            sentences, remainder = self._sentence_parts(text, force=force)
            if sentences:
                buffer[:] = [remainder] if remainder else []
                for sentence in sentences:
                    self.enqueue(sentence)

        def on_event(event: dict) -> None:
            nonlocal finished
            event_type = event.get("type")
            if event_type == "assistant_text":
                buffer.append(str(event.get("content") or ""))
                flush(force=True)
                if not finished:
                    self.finish()
                    finished = True
            elif event_type == "token":
                token = str(event.get("content") or "")
                buffer.append(token)
                flush(force="\n" in token)
                if "\n" in token and not finished:
                    self.finish()
                    finished = True

        return on_event

    def enqueue(self, text: str) -> None:
        for sentence in self._sentence_parts(text, force=True)[0]:
            self._enqueue("text", text=sentence)

    def finish(self, wait: bool = False) -> None:
        done = threading.Event() if wait else None
        self._enqueue("finish", done=done)
        if done is not None:
            done.wait()

    def wait(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(("stop", None, None))
            self._thread.join(timeout=5)

    def play_chime(self, kind: str) -> None:
        if self.chime_config["enabled"] and self.chime_config.get(kind) == "builtin":
            self.audio.play(self.audio.builtin_chime(880.0 if kind == "start" else 660.0))

    def _enqueue(
        self,
        command: str,
        text: str | None = None,
        done: threading.Event | None = None,
    ) -> None:
        self._start_worker()
        self._queue.put((command, text, done))

    def _start_worker(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def _worker(self) -> None:
        while True:
            command, text, done = self._queue.get()
            try:
                if command == "stop":
                    return
                if text:
                    self._tts_playing.set()
                    self._play_text(text)
                if command == "finish":
                    self.play_chime("end")
                if done is not None:
                    done.set()
            except BaseException:
                if done is not None:
                    done.set()
            finally:
                if self._queue.empty():
                    self._tts_playing.clear()
                self._queue.task_done()

    def _play_text(self, text: str) -> None:
        self.audio.play(self.tts.synthesize(text))

    @staticmethod
    def _sentence_parts(text: str, force: bool) -> tuple[list[str], str]:
        text = text.strip()
        if not text:
            return [], ""
        sentences: list[str] = []
        last_end = 0
        for match in re.finditer(r".+?[.!?](?:\s+|$)", text, flags=re.DOTALL):
            sentence = match.group(0).strip()
            if sentence:
                sentences.append(sentence)
            last_end = match.end()
        remainder = text[last_end:].strip()
        if force and remainder:
            sentences.append(remainder)
            remainder = ""
        return sentences, remainder
