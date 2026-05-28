from __future__ import annotations

import ast
import json
import re
from typing import Any


def parse_tool_calls(content: str, tool_mode: str) -> list[dict[str, Any]]:
    parsers = [
        _parse_json_tool_calls,
        _parse_lfm_tool_calls,
        _parse_functiongemma_tool_calls,
    ]
    if tool_mode == "lfm":
        parsers = [_parse_lfm_tool_calls, _parse_json_tool_calls, _parse_functiongemma_tool_calls]
    elif tool_mode == "functiongemma":
        parsers = [_parse_functiongemma_tool_calls, _parse_json_tool_calls, _parse_lfm_tool_calls]

    seen = set()
    calls: list[dict[str, Any]] = []
    for parser in parsers:
        for call in parser(content):
            key = json.dumps(call, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                calls.append(call)
        if calls:
            return calls
    return []


def parse_response_tool_calls(
    response: dict[str, Any],
    tool_mode: str,
) -> list[dict[str, Any]]:
    api_calls = parse_api_tool_calls(response)
    if api_calls:
        return api_calls
    return parse_tool_calls(response.get("content") or "", tool_mode)


def parse_api_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for tool_call in response.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        name = function.get("name")
        raw_args = function.get("arguments") or {}
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        else:
            args = raw_args
        if isinstance(name, str) and isinstance(args, dict):
            calls.append({"tool": name, "args": args})
    return calls


def _parse_json_tool_calls(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    candidates = []
    if text.startswith(("[", "{")):
        candidates.append(text)
    candidates.extend(re.findall(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL))
    candidates.extend(_balanced_json_fragments(text, "[", "]"))
    candidates.extend(_balanced_json_fragments(text, "{", "}"))

    calls: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            normalised = _normalise_json_tool_calls(json.loads(candidate))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if normalised:
            calls.extend(normalised)
    return calls


def _normalise_json_tool_calls(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        if any(key in parsed for key in ("tool", "name")):
            parsed = [parsed]
        else:
            parsed = parsed.get("tool_calls") or parsed.get("calls") or []
    if not isinstance(parsed, list):
        return []

    calls = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("tool") or item.get("name")
        args = item.get("args")
        if args is None:
            args = item.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if args is None:
            args = item.get("parameters")
        if args is None:
            args = {
                key: value
                for key, value in item.items()
                if key not in {"tool", "name", "arguments", "parameters", "args"}
            }
        if isinstance(name, str) and isinstance(args, dict):
            calls.append({"tool": name, "args": args})
    return calls


def _balanced_json_fragments(text: str, open_char: str, close_char: str) -> list[str]:
    fragments = []
    stack = 0
    start = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_char:
            if stack == 0:
                start = index
            stack += 1
        elif char == close_char and stack:
            stack -= 1
            if stack == 0 and start is not None:
                fragments.append(text[start : index + 1])
    return fragments


def _parse_lfm_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    for match in re.findall(
        r"<\|tool_call_start\|>\[(.*?)\]<\|tool_call_end\|>", content, re.DOTALL
    ):
        try:
            tree = ast.parse(f"[{match.strip()}]", mode="eval")
            if not isinstance(tree.body, ast.List):
                continue
            for item in tree.body.elts:
                if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name):
                    continue
                calls.append(
                    {
                        "tool": item.func.id,
                        "args": {
                            kw.arg: ast.literal_eval(kw.value)
                            for kw in item.keywords
                            if kw.arg is not None
                        },
                    }
                )
        except (SyntaxError, ValueError, TypeError):
            # Ignore malformed LFM fragments and keep scanning for valid tool calls.
            continue
    return calls


def _parse_functiongemma_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    pattern = r"<start_function_call>\s*call:([A-Za-z_][A-Za-z0-9_]*)\{(.*?)\}<end_function_call>"
    for name, body in re.findall(pattern, content, re.DOTALL):
        args = {}
        for arg_name, value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*):<escape>(.*?)<escape>", body, re.DOTALL
        ):
            args[arg_name] = value
        calls.append({"tool": name, "args": args})
    return calls
