# Speech

Optional voice interface for `uniagent`:

`mic -> VAD -> STT -> Agent.run() -> streaming TTS -> speakers`

## Setup

1. Install the extra into the project venv: `uv pip install -e ".[speech]"`.
2. Set `tts.model_path` in `config_speech.yaml` to a Piper `.onnx` voice.
3. Optional: set `stt.models_dir` if you want pywhispercpp models stored outside the default cache.

Piper needs the `.onnx` file and its matching `.onnx.json` file next to it.

## Run

- Live mic/speaker loop: `.venv/bin/python examples/speech_basic.py`
- No-mic synthetic harness: `.venv/bin/python scripts/speech_pipeline_harness.py`

The synthetic harness turns typed text into audio, sends it through real VAD/STT, runs the agent, and plays the response through the production TTS path. It tests the end-to-end loop, not ASR quality.

## API

- `SpeechPipeline`: live listening, response TTS, and `on_event` callback support.
- `make_confirm_fn(pipeline)`: voice confirmation callback for HITL tool approval.
- `load_speech_config(path)`: load `config_speech.yaml`.

## Files

- `pipeline.py`: public pipeline.
- `audio.py`: microphone input, speaker playback, resampling, chimes.
- `vad.py`: Silero VAD.
- `stt.py`: pywhispercpp.
- `tts.py`: Piper and sentence-streamed playback queue.
- `config.py`: defaults and config loading.
