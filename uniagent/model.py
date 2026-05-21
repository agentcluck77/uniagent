import os
import time
from collections.abc import Callable
from typing import Any

from openai import OpenAI


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: int,
        temperature: float | None,
        max_tokens: int | None,
        seed: int | None,
    ):
        self.model_name = model_name
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str, float], None] | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "timeout": self.timeout,
        }
        if tools:
            kwargs["tools"] = tools
        for key, value in {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }.items():
            if value is not None:
                kwargs[key] = value
        if on_token is not None:
            return self._complete_streaming(kwargs, on_token)
        return self._complete_blocking(kwargs)

    def _complete_blocking(self, kwargs: dict[str, Any]) -> dict:
        t0 = time.monotonic()
        completion = self.client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000
        msg = completion.choices[0].message.model_dump()
        usage = completion.usage
        msg["_stats"] = {
            "completion_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": latency_ms,
        }
        return msg

    def _complete_streaming(
        self, kwargs: dict[str, Any], on_token: Callable[[str, float], None]
    ) -> dict:
        content = ""
        tool_call_accum: dict[int, dict] = {}
        chunk_count = 0
        completion_tokens = 0
        t0 = time.monotonic()

        for chunk in self.client.chat.completions.create(**kwargs, stream=True):
            choice = chunk.choices[0] if chunk.choices else None
            if choice is not None:
                delta = choice.delta
                if delta.content:
                    content += delta.content
                    chunk_count += 1
                    elapsed = time.monotonic() - t0
                    tps = round(chunk_count / elapsed, 1) if elapsed > 0 else 0.0
                    on_token(delta.content, tps)
                for tc in delta.tool_calls or []:
                    idx = tc.index
                    if idx not in tool_call_accum:
                        tool_call_accum[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.function:
                        if tc.function.name:
                            tool_call_accum[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_accum[idx]["function"]["arguments"] += tc.function.arguments
            if chunk.usage:
                completion_tokens = chunk.usage.completion_tokens

        if completion_tokens == 0:
            completion_tokens = chunk_count

        latency_ms = (time.monotonic() - t0) * 1000
        msg: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_call_accum:
            msg["tool_calls"] = [tool_call_accum[i] for i in sorted(tool_call_accum)]
        msg["_stats"] = {"completion_tokens": completion_tokens, "latency_ms": latency_ms}
        return msg


def backend_from_config(config: dict) -> OpenAICompatibleBackend:
    model = config["model"]
    backend = model["backend"]
    api_key = model.get("api_key")
    if api_key is None and backend in {"ollama", "llamacpp"}:
        api_key = backend
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("model.api_key is required for openai or set OPENAI_API_KEY")
    return OpenAICompatibleBackend(
        base_url=model["base_url"],
        model_name=model["model_name"],
        api_key=api_key,
        timeout=model["timeout"],
        temperature=model.get("temperature"),
        max_tokens=model.get("max_tokens"),
        seed=model.get("seed"),
    )
