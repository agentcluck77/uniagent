from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from uniagent.speech.audio import AudioIO
from uniagent.speech.config import DEFAULT_SPEECH_CONFIG, deep_merge
from uniagent.speech.stt import WhisperSTT
from uniagent.speech.tts import PiperTTS, SpeechOutput
from uniagent.speech.vad import SileroVAD


def make_confirm_fn(pipeline: SpeechPipeline) -> Callable[[str], str]:
    def confirm_fn(question: str) -> str:
        pipeline.speak(question)
        return pipeline.listen()

    return confirm_fn


class SpeechPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = deep_merge(DEFAULT_SPEECH_CONFIG, config)
        self.sample_rate = int(self.config["vad"]["sample_rate"])
        self._tts_playing = threading.Event()
        self.vad = SileroVAD(self.config["vad"], self.sample_rate)
        self.audio = AudioIO(self.config["audio"], self.sample_rate, self._tts_playing)
        self.stt = WhisperSTT(self.config["stt"])
        self.tts = PiperTTS(self.config["tts"])
        self.output = SpeechOutput(
            tts=self.tts,
            audio=self.audio,
            chime_config=self.config["chime"],
            tts_playing=self._tts_playing,
        )

    def listen(self) -> str:
        audio = self.audio.record_utterance(self.vad)
        self.output.play_chime("start")
        return self.stt.transcribe(self.audio.prepare_input(audio)).strip()

    def listen_ptt(self, stop_event: threading.Event) -> str:
        audio = self.audio.record_until(stop_event)
        self.output.play_chime("start")
        return self.stt.transcribe(self.audio.prepare_input(audio)).strip()

    def cancel_tts(self) -> None:
        self.output.cancel()

    def speak(self, text: str) -> None:
        self.output.speak(text)

    def make_on_event(self) -> Callable[[dict], None]:
        return self.output.make_on_event()

    def wait_for_tts(self) -> None:
        self.output.wait()

    def close(self) -> None:
        self.output.close()

    def __enter__(self) -> SpeechPipeline:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
