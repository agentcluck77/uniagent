# pyright: reportUnsupportedDunderAll=false
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from uniagent.tool import tool

DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "backend": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model_name": "gemma4:e2b",
        "api_key": None,
        "timeout": 60,
        "tool_mode": "api",
        "temperature": None,
        "max_tokens": None,
        "seed": None,
        "stream": False,
    },
    "agent": {
        "max_iterations": 10,
        "system_prompt": "You are a helpful assistant.",
        "confirm_tools": False,
        "stop_after_tool_calls": False,
        "tool_call_confirmation": "Tool calls executed.",
        "retry_invalid_tool_calls": False,
        "max_tool_call_retries": 1,
        "history_turns": None,
    },
    "toolrag": {
        "enabled": True,
        "top_k": 5,
        "backend": "tfidf",
        "st_model": None,
        "reselect_window": 3,
        "reselect_roles": ["user", "assistant"],
    },
    "scratchpad": {
        "enabled": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config file must contain a YAML mapping")
    return _deep_merge(DEFAULT_CONFIG, loaded)


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from uniagent.agent import Agent

        return Agent
    raise AttributeError(name)


__all__ = ["Agent", "tool", "load_config"]
