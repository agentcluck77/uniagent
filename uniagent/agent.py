import json
from collections.abc import Callable
from typing import Any

from uniagent.model import backend_from_config
from uniagent.tool_parsing import parse_tool_calls
from uniagent.tool_validation import (
    tool_call_batch_errors,
    tool_call_retry_prompt,
    validate_tool_call,
)
from uniagent.toolrag import ToolRAG

MAX_ITERATIONS_ERROR = (
    "Error: maximum agent iterations reached before a final response was produced."
)


def _emit(on_event: Callable[[dict], None] | None, event: dict) -> None:
    if on_event is not None:
        try:
            on_event(event)
        except Exception:
            pass


class Agent:
    def __init__(
        self,
        config: dict,
        backend: Any,
        tools: list[Callable],
        toolrag: ToolRAG | None = None,
        state_fn: Callable[[], str] | None = None,
    ):
        self.config = config
        self.backend = backend
        self.tools = self._eligible_tools(config, tools)
        self.toolrag = toolrag
        self.state_fn = state_fn
        self._tool_map = {fn.__name__: fn for fn in self.tools}
        self._history: list[list[dict]] = []

    @classmethod
    def from_config(
        cls,
        config: dict,
        tools: list[Callable],
        *,
        state_fn: Callable[[], str] | None = None,
    ) -> "Agent":
        backend = backend_from_config(config)
        tools = cls._eligible_tools(config, tools)
        toolrag = None
        if config["toolrag"]["enabled"]:
            toolrag = ToolRAG(
                tools,
                backend=config["toolrag"]["backend"],
                st_model=config["toolrag"]["st_model"],
            )
        return cls(config=config, backend=backend, tools=tools, toolrag=toolrag, state_fn=state_fn)

    def clear_history(self) -> None:
        self._history.clear()

    def run(
        self,
        user_message: str,
        on_event: Callable[[dict], None] | None = None,
        confirm_fn: Callable[[str], str] | None = None,
    ) -> str:
        history_turns = self.config["agent"].get("history_turns")
        prefix = self._slice_history(history_turns)
        history: list[dict] = (
            [{"role": "system", "content": self._build_system_prompt()}]
            + prefix
            + [{"role": "user", "content": user_message}]
        )
        prefix_len = len(prefix)

        scratchpad = {} if self.config["scratchpad"]["enabled"] else None
        session_tool_calls = 0
        invalid_tool_call_retries = 0
        result = MAX_ITERATIONS_ERROR

        for iteration in range(self.config["agent"]["max_iterations"]):
            active_tools = self._active_tools(history)
            tool_schemas = [fn._schema for fn in active_tools]
            response = self._complete(history, tool_schemas, on_event)
            backend_stats = response.pop("_stats", {})

            tokens = backend_stats.get("completion_tokens", 0)
            latency_ms = backend_stats.get("latency_ms", 0)
            tok_per_sec = round(tokens / latency_ms * 1000, 1) if latency_ms > 0 else 0.0
            _emit(
                on_event,
                {
                    "type": "stats",
                    "tok_per_sec": tok_per_sec,
                    "completion_tokens": tokens,
                    "latency_ms": round(latency_ms),
                    "active_tools": [fn.__name__ for fn in active_tools],
                    "history_depth": len(history),
                    "iteration": iteration + 1,
                    "session_tool_calls": session_tool_calls,
                },
            )

            if not response.get("tool_calls"):
                content = response.get("content") or ""
                history.append({"role": "assistant", "content": content})
                if response.get("_streamed_tokens", False):
                    _emit(on_event, {"type": "token", "content": "\n", "tok_per_sec": 0.0})
                else:
                    _emit(on_event, {"type": "assistant_text", "content": content})
                result = content
                break

            validation_errors = tool_call_batch_errors(
                response["tool_calls"],
                self._tool_map,
                scratchpad,
            )
            if validation_errors:
                _emit(
                    on_event,
                    {
                        "type": "tool_validation_error",
                        "errors": validation_errors,
                        "retry": self._can_retry_invalid_tool_calls(invalid_tool_call_retries),
                    },
                )
                if self._can_retry_invalid_tool_calls(invalid_tool_call_retries):
                    invalid_tool_call_retries += 1
                    history.append(self._invalid_tool_call_history_message(response))
                    history.append(
                        {
                            "role": "user",
                            "content": tool_call_retry_prompt(
                                validation_errors,
                                tool_schemas,
                                self.config["model"]["tool_mode"],
                            ),
                        }
                    )
                    continue
                message = "Error: invalid tool call batch: " + "; ".join(validation_errors)
                _emit(on_event, {"type": "assistant_text", "content": message})
                result = message
                break

            history.append(self._assistant_history_message(response))
            results = [
                (
                    tc["function"]["name"],
                    self._execute_tool_call(tc, scratchpad, on_event, confirm_fn),
                )
                for tc in response["tool_calls"]
            ]
            session_tool_calls += len(results)
            self._append_results(history, results)

            if scratchpad is not None:
                _emit(on_event, {"type": "scratchpad", "state": dict(scratchpad)})
        else:
            _emit(on_event, {"type": "assistant_text", "content": MAX_ITERATIONS_ERROR})

        if history_turns is not None:
            self._history.append(history[1 + prefix_len :])

        return result

    def _build_system_prompt(self) -> str:
        base = self.config["agent"]["system_prompt"]
        if self.state_fn is not None:
            state_str = self.state_fn()
            if state_str:
                return f"{base}\n\n{state_str}"
        return base

    def _slice_history(self, history_turns: int | None) -> list[dict]:
        if history_turns is None or not self._history:
            return []
        turns = self._history if history_turns == 0 else self._history[-history_turns:]
        return [msg for turn in turns for msg in turn]

    def _complete(
        self,
        history: list[dict],
        tool_schemas: list[dict],
        on_event: Callable[[dict], None] | None = None,
    ) -> dict:
        tool_mode = self.config["model"]["tool_mode"]
        stream = self.config["model"].get("stream", False)

        if tool_mode == "api":
            streamed_tokens = False

            def on_token(tok: str, tps: float) -> None:
                nonlocal streamed_tokens
                streamed_tokens = True
                _emit(on_event, {"type": "token", "content": tok, "tok_per_sec": tps})

            response = self.backend.complete(
                history,
                tool_schemas,
                on_token=on_token if stream else None,
            )
            response["_streamed_tokens"] = streamed_tokens
            return response

        history[0] = {"role": "system", "content": self._prompt_system(tool_schemas)}

        if stream:
            token_buf: list[str] = []
            tps_last: list[float] = [0.0]

            def buf_token(tok: str, tps: float) -> None:
                token_buf.append(tok)
                tps_last[0] = tps

            raw = self.backend.complete(history, [], on_token=buf_token)
        else:
            raw = self.backend.complete(history, [])

        stats = raw.pop("_stats", {})
        content = raw.get("content") or ""
        calls = self._parse_prompt_tool_calls(content)

        if calls is None:
            streamed_tokens = False
            if stream and token_buf:
                for tok in token_buf:
                    _emit(on_event, {"type": "token", "content": tok, "tok_per_sec": tps_last[0]})
                streamed_tokens = True
            return {
                "role": "assistant",
                "content": content,
                "_stats": stats,
                "_streamed_tokens": streamed_tokens,
            }

        return {
            "role": "assistant",
            "content": content,
            "_stats": stats,
            "_streamed_tokens": False,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "function": {
                        "name": c["tool"],
                        "arguments": json.dumps(c.get("args", {})),
                    },
                }
                for i, c in enumerate(calls)
            ],
        }

    def _assistant_history_message(self, response: dict) -> dict:
        if self.config["model"]["tool_mode"] == "api":
            return {key: value for key, value in response.items() if not key.startswith("_")}
        return {"role": "assistant", "content": response.get("content") or ""}

    def _invalid_tool_call_history_message(self, response: dict) -> dict:
        content = response.get("content")
        if content:
            return {"role": "assistant", "content": content}
        tool_calls = [
            {
                "name": (tc.get("function") or {}).get("name", ""),
                "arguments": (tc.get("function") or {}).get("arguments", ""),
            }
            for tc in response.get("tool_calls", [])
        ]
        return {
            "role": "assistant",
            "content": "Invalid tool call batch: "
            + json.dumps(tool_calls, ensure_ascii=False, sort_keys=True),
        }

    def _append_results(self, history: list[dict], results: list[tuple[str, dict]]) -> None:
        if self.config["model"]["tool_mode"] == "api":
            for _, msg in results:
                history.append(msg)
        else:
            lines = "\n".join(f"{name} → {msg['content']}" for name, msg in results)
            history.append({"role": "user", "content": f"Tool results:\n{lines}"})

    def _active_tools(self, history: list[dict]) -> list[Callable]:
        top_k = self.config["toolrag"]["top_k"]
        if self.toolrag is None or len(self.tools) <= top_k:
            return self.tools
        window = self.config["toolrag"]["reselect_window"]
        roles = set(self.config["toolrag"]["reselect_roles"])
        recent = [m for m in history if m["role"] in roles][-window:]
        query = " ".join(m["content"] for m in recent if m.get("content"))
        if not query:
            return self.tools
        return self.toolrag.retrieve(query, top_k)

    def _prompt_system(self, tool_schemas: list[dict]) -> str:
        base = self._build_system_prompt()
        if self.config["model"]["tool_mode"] == "lfm":
            tools_json = json.dumps(
                [schema["function"] for schema in tool_schemas],
                indent=2,
            )
            return (
                f"{base}\n\n"
                f"List of tools: {tools_json}\n\n"
                "To call tools, respond with Liquid LFM tool-call syntax only:\n"
                "<|tool_call_start|>[tool_name(arg_name=value)]<|tool_call_end|>\n"
                "Use Python literals for strings, integers, floats, and booleans. "
                "Use True and False for boolean values. "
                "When no tool is needed, respond in plain text."
            )
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            f"{base}\n\n"
            f"You have access to the following tools:\n{tools_json}\n\n"
            "To call tools, respond ONLY with a JSON array:\n"
            '[{"tool": "tool_name", "args": {"param": "value"}}, ...]\n\n'
            "To give a final answer without calling tools, respond in plain text."
        )

    def _parse_prompt_tool_calls(self, content: str) -> list[dict] | None:
        calls = parse_tool_calls(content, self.config["model"]["tool_mode"])
        return calls or None

    @staticmethod
    def _eligible_tools(config: dict, tools: list[Callable]) -> list[Callable]:
        if config["scratchpad"]["enabled"]:
            return tools
        return [fn for fn in tools if not fn._needs_scratchpad]

    def _classify_approval(self, user_text: str) -> bool:
        try:
            messages = [
                {"role": "system", "content": "You are a binary classifier."},
                {
                    "role": "user",
                    "content": (
                        f"The user was asked to approve or deny running a tool call.\n"
                        f'Their response was: "{user_text}"\n'
                        'Did they approve? Reply with only "yes" or "no".'
                    ),
                },
            ]
            response = self.backend.complete(messages, [])
            response.pop("_stats", None)
            return "yes" in (response.get("content") or "").lower()
        except Exception:
            return False

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        scratchpad: dict | None,
        on_event: Callable[[dict], None] | None = None,
        confirm_fn: Callable[[str], str] | None = None,
    ) -> dict:
        tool_call_id = tool_call["id"]
        fn_name = tool_call["function"]["name"]

        def finish(result: Any) -> dict:
            _emit(
                on_event,
                {
                    "type": "tool_result",
                    "name": fn_name,
                    "tool_call_id": tool_call_id,
                    "content": str(result),
                },
            )
            return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}

        validation = validate_tool_call(tool_call, self._tool_map, scratchpad)

        _emit(
            on_event,
            {
                "type": "tool_call",
                "name": fn_name,
                "args": validation.args,
                "tool_call_id": tool_call_id,
            },
        )
        if validation.errors:
            return finish("Error: " + "; ".join(validation.errors))
        fn = validation.fn
        assert fn is not None

        if confirm_fn is not None and self.config["agent"]["confirm_tools"]:
            args_str = ", ".join(f"{k}={v}" for k, v in validation.args.items())
            question = f"Can I use the {fn_name} function with inputs {args_str}?"
            user_text = confirm_fn(question)
            if not self._classify_approval(user_text):
                return finish(f"denied by user: {user_text}")

        try:
            if fn._needs_scratchpad:
                result = fn(**validation.args, scratchpad=scratchpad)
            else:
                result = fn(**validation.args)
        except Exception as error:
            result = f"Error: {error}"
        return finish(result)

    def _can_retry_invalid_tool_calls(self, retries_used: int) -> bool:
        agent_config = self.config["agent"]
        return bool(agent_config.get("retry_invalid_tool_calls")) and retries_used < int(
            agent_config.get("max_tool_call_retries", 1)
        )
