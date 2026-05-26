from __future__ import annotations

from typing import Any

from uniagent.speech.pipeline import SpeechPipeline
from uniagent.speech.types import AudioBuffer


class SpeechTestHarness:
    def __init__(self, pipeline: SpeechPipeline):
        self.pipeline = pipeline

    def synthesize_input(self, text: str) -> AudioBuffer:
        return self.pipeline.tts.synthesize(text)

    def transcribe_audio(self, audio: AudioBuffer | Any, play_chime: bool = True) -> str:
        prepared = self.pipeline.audio.prepare_input(audio)
        speech = self.pipeline.vad.extract(prepared)
        if play_chime:
            self.pipeline.output.play_chime("start")
        return self.pipeline.stt.transcribe(speech).strip()

    def play_audio(self, audio: AudioBuffer) -> None:
        self.pipeline.audio.play(audio)
