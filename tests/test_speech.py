import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from uniagent.speech import DEFAULT_SPEECH_CONFIG, AudioBuffer, SpeechPipeline, load_speech_config
from uniagent.speech.pipeline import make_confirm_fn


class SpeechConfigTests(unittest.TestCase):
    def test_load_speech_config_deep_merges_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.yaml"
            path.write_text("tts:\n  model_path: /tmp/voice.onnx\n", encoding="utf-8")

            config = load_speech_config(str(path))

        self.assertEqual(config["tts"]["model_path"], "/tmp/voice.onnx")
        self.assertEqual(config["vad"]["sample_rate"], DEFAULT_SPEECH_CONFIG["vad"]["sample_rate"])
        self.assertEqual(config["stt"]["backend"], "whisper.cpp")
        self.assertIsNone(config["stt"]["models_dir"])


class SpeechPipelineTests(unittest.TestCase):
    def test_make_confirm_fn_speaks_then_listens(self) -> None:
        pipeline = Mock()
        pipeline.listen.return_value = "yes"

        confirm = make_confirm_fn(pipeline)

        self.assertEqual(confirm("Can I use a tool?"), "yes")
        pipeline.speak.assert_called_once_with("Can I use a tool?")
        pipeline.listen.assert_called_once_with()

    def test_on_event_flushes_streaming_sentences(self) -> None:
        pipeline = SpeechPipeline({"chime": {"enabled": False}})
        spoken: list[str] = []
        finished: list[bool] = []
        pipeline.output.enqueue = spoken.append  # type: ignore[method-assign]
        pipeline.output.finish = lambda wait=False: finished.append(wait)  # type: ignore[method-assign]

        on_event = pipeline.make_on_event()
        on_event({"type": "token", "content": "First sentence. Second"})
        on_event({"type": "token", "content": " sentence"})
        on_event({"type": "token", "content": "\n"})

        self.assertEqual(spoken, ["First sentence.", "Second sentence"])
        self.assertEqual(finished, [False])

    def test_on_event_preserves_remainder_after_multiple_sentences(self) -> None:
        pipeline = SpeechPipeline({"chime": {"enabled": False}})
        spoken: list[str] = []
        pipeline.output.enqueue = spoken.append  # type: ignore[method-assign]
        pipeline.output.finish = lambda wait=False: None  # type: ignore[method-assign]

        on_event = pipeline.make_on_event()
        on_event({"type": "token", "content": "One. Two. Thr"})
        on_event({"type": "token", "content": "ee.\n"})

        self.assertEqual(spoken, ["One.", "Two.", "Three."])

    def test_on_event_handles_non_streaming_assistant_text(self) -> None:
        pipeline = SpeechPipeline({"chime": {"enabled": False}})
        spoken: list[str] = []
        finished: list[bool] = []
        pipeline.output.enqueue = spoken.append  # type: ignore[method-assign]
        pipeline.output.finish = lambda wait=False: finished.append(wait)  # type: ignore[method-assign]

        pipeline.make_on_event()({"type": "assistant_text", "content": "No punctuation needed"})

        self.assertEqual(spoken, ["No punctuation needed"])
        self.assertEqual(finished, [False])

    def test_prepare_input_audio_resamples_to_vad_rate(self) -> None:
        import numpy as np

        pipeline = SpeechPipeline({"vad": {"sample_rate": 16000}, "chime": {"enabled": False}})
        source = np.ones(22050, dtype=np.float32)

        prepared = pipeline.audio.prepare_input(AudioBuffer(source, 22050))

        self.assertEqual(prepared.sample_rate, 16000)
        self.assertEqual(prepared.samples.shape, (16000,))


if __name__ == "__main__":
    unittest.main()
