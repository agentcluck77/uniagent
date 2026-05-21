from __future__ import annotations

import json
import unittest
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from uniagent import DEFAULT_CONFIG
from uniagent.agent import Agent
from uniagent.tool import tool


class FakeBackend:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[dict],
        tools: list[dict],
        on_token: Callable[[str, float], None] | None = None,
    ) -> dict:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "streaming": on_token is not None,
            }
        )
        response = deepcopy(self.responses.pop(0))
        for token, tps in response.pop("_emit_tokens", []):
            if on_token is not None:
                on_token(token, tps)
        return response


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


@tool(description="Maybe store an integer")
def maybe_store(value: int | None = None) -> str:
    return str(value)


def make_config() -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    config["toolrag"]["enabled"] = False
    return config


def stats() -> dict[str, int]:
    return {"completion_tokens": 1, "latency_ms": 1}


class AgentLoopTests(unittest.TestCase):
    def test_tool_schema_treats_default_arguments_as_optional(self) -> None:
        schema = maybe_store._schema["function"]["parameters"]

        self.assertEqual(schema["properties"]["value"]["type"], "integer")
        self.assertEqual(schema["required"], [])

    def test_prompt_and_lfm_modes_do_not_append_synthetic_tool_calls(self) -> None:
        cases = {
            "prompt": json.dumps([{"tool": "multiply", "args": {"a": 2, "b": 3}}]),
            "lfm": "<|tool_call_start|>[multiply(a=2, b=3)]<|tool_call_end|>",
        }
        for tool_mode, tool_response in cases.items():
            with self.subTest(tool_mode=tool_mode):
                config = make_config()
                config["model"]["tool_mode"] = tool_mode
                backend = FakeBackend(
                    [
                        {
                            "role": "assistant",
                            "content": tool_response,
                            "_stats": stats(),
                        },
                        {"role": "assistant", "content": "done", "_stats": stats()},
                    ]
                )
                agent = Agent(config, backend, [multiply])

                self.assertEqual(agent.run("multiply 2 by 3"), "done")

                second_call_history = backend.calls[1]["messages"]
                self.assertFalse(any("tool_calls" in msg for msg in second_call_history))
                self.assertEqual(second_call_history[-2]["role"], "assistant")
                self.assertEqual(second_call_history[-2]["content"], tool_response)
                self.assertEqual(second_call_history[-1]["role"], "user")
                self.assertIn("Tool results:", second_call_history[-1]["content"])

    def test_invalid_tool_calls_fail_fast_without_confirmation(self) -> None:
        config = make_config()
        config["agent"]["confirm_tools"] = True
        backend = FakeBackend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_unknown",
                            "function": {"name": "missing_tool", "arguments": "{}"},
                        },
                        {
                            "id": "call_missing_arg",
                            "function": {
                                "name": "multiply",
                                "arguments": json.dumps({"a": 2}),
                            },
                        },
                    ],
                    "_stats": stats(),
                },
            ]
        )
        agent = Agent(config, backend, [multiply])
        confirm_calls: list[str] = []
        events: list[dict[str, Any]] = []

        def confirm(question: str) -> str:
            confirm_calls.append(question)
            return "yes"

        result = agent.run("call invalid tools", events.append, confirm)

        self.assertIn("Error: invalid tool call batch:", result)
        self.assertIn("unknown tool 'missing_tool'", result)
        self.assertIn("missing required argument(s): b", result)
        self.assertEqual(confirm_calls, [])
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(any(event.get("type") == "tool_validation_error" for event in events))

    def test_invalid_tool_calls_retry_whole_batch_when_enabled(self) -> None:
        recorded: list[int] = []

        @tool(description="Record an integer")
        def record(value: int) -> str:
            recorded.append(value)
            return f"recorded {value}"

        config = make_config()
        config["agent"]["retry_invalid_tool_calls"] = True
        config["agent"]["max_tool_call_retries"] = 1
        backend = FakeBackend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_valid_but_not_executed",
                            "function": {
                                "name": "record",
                                "arguments": json.dumps({"value": 1}),
                            },
                        },
                        {
                            "id": "call_missing_arg",
                            "function": {
                                "name": "multiply",
                                "arguments": json.dumps({"a": 2}),
                            },
                        },
                    ],
                    "_stats": stats(),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_record",
                            "function": {
                                "name": "record",
                                "arguments": json.dumps({"value": 2}),
                            },
                        },
                        {
                            "id": "call_multiply",
                            "function": {
                                "name": "multiply",
                                "arguments": json.dumps({"a": 2, "b": 3}),
                            },
                        },
                    ],
                    "_stats": stats(),
                },
                {"role": "assistant", "content": "handled", "_stats": stats()},
            ]
        )
        agent = Agent(config, backend, [multiply, record])

        self.assertEqual(agent.run("call tools"), "handled")
        self.assertEqual(recorded, [2])
        retry_history = backend.calls[1]["messages"]
        self.assertIn("Regenerate the entire batch", retry_history[-1]["content"])
        self.assertFalse(any(message.get("role") == "tool" for message in retry_history))

    def test_tool_call_schema_validation_rejects_extra_args_and_type_mismatches(self) -> None:
        config = make_config()
        backend = FakeBackend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad_schema",
                            "function": {
                                "name": "multiply",
                                "arguments": json.dumps({"a": "2", "b": 3, "unused": True}),
                            },
                        },
                    ],
                    "_stats": stats(),
                },
            ]
        )
        agent = Agent(config, backend, [multiply])

        result = agent.run("multiply badly")

        self.assertIn("unexpected argument(s): unused", result)
        self.assertIn("argument 'a' expected integer, got string", result)

    def test_streaming_final_event_uses_actual_token_emission(self) -> None:
        config = make_config()
        config["model"]["stream"] = True

        no_token_backend = FakeBackend(
            [{"role": "assistant", "content": "silent final", "_stats": stats()}]
        )
        no_token_events: list[dict[str, Any]] = []
        no_token_agent = Agent(config, no_token_backend, [])

        self.assertEqual(no_token_agent.run("hello", no_token_events.append), "silent final")
        self.assertIn(
            {"type": "assistant_text", "content": "silent final"},
            no_token_events,
        )
        self.assertNotIn(
            {"type": "token", "content": "\n", "tok_per_sec": 0.0},
            no_token_events,
        )

        token_backend = FakeBackend(
            [
                {
                    "role": "assistant",
                    "content": "streamed final",
                    "_stats": stats(),
                    "_emit_tokens": [("streamed", 10.0), (" final", 12.0)],
                }
            ]
        )
        token_events: list[dict[str, Any]] = []
        token_agent = Agent(config, token_backend, [])

        self.assertEqual(token_agent.run("hello", token_events.append), "streamed final")
        self.assertEqual(
            [event["content"] for event in token_events if event.get("type") == "token"],
            ["streamed", " final", "\n"],
        )
        self.assertFalse(any(event.get("type") == "assistant_text" for event in token_events))


if __name__ == "__main__":
    unittest.main()
