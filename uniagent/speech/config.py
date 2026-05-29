from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SPEECH_CONFIG: dict[str, Any] = {
    "vad": {
        "enabled": True,
        "model": "silero_v5",
        "threshold": 0.5,
        "silence_duration_ms": 700,
        "max_recording_s": 30,
        "sample_rate": 16000,
    },
    "stt": {
        "backend": "whisper.cpp",
        "model": "tiny",
        "language": None,
        "models_dir": None,
    },
    "tts": {
        "backend": "piper",
        "model_path": None,
        "voice": None,
        "speaking_rate": 1.0,
    },
    "audio": {
        "input_device": None,
        "output_volume": 1.0,
    },
    "chime": {
        "enabled": True,
        "start": "builtin",
        "end": "builtin",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_speech_config(path: str) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise ValueError("speech config file must contain a YAML mapping")
    return deep_merge(DEFAULT_SPEECH_CONFIG, loaded)
