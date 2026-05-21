from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallValidation:
    tool_call_id: str
    name: str
    args: dict[str, Any]
    errors: list[str]
    fn: Callable | None


def tool_call_batch_errors(
    tool_calls: list[dict[str, Any]],
    tool_map: dict[str, Callable],
    scratchpad: dict | None,
) -> list[str]:
    errors: list[str] = []
    for index, tool_call in enumerate(tool_calls):
        validation = validate_tool_call(tool_call, tool_map, scratchpad)
        errors.extend(
            f"{validation.tool_call_id or f'call_{index}'} {validation.name}: {error}"
            for error in validation.errors
        )
    return errors


def validate_tool_call(
    tool_call: dict[str, Any],
    tool_map: dict[str, Callable],
    scratchpad: dict | None,
) -> ToolCallValidation:
    tool_call_id = str(tool_call.get("id") or "")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ToolCallValidation(tool_call_id, "<missing>", {}, ["missing function"], None)

    name = str(function.get("name") or "")
    raw_arguments = function.get("arguments") or "{}"
    args: dict[str, Any] = {}
    errors: list[str] = []
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        if isinstance(parsed, dict):
            args = parsed
        else:
            errors.append("arguments must be a JSON object")
    except json.JSONDecodeError as error:
        errors.append(f"could not parse arguments as JSON: {error}")

    fn = tool_map.get(name)
    if fn is None:
        errors.append(f"unknown tool '{name}'")
        return ToolCallValidation(tool_call_id, name, args, errors, None)

    parameters = fn._schema["function"]["parameters"]
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    extra = [key for key in args if key not in properties]
    missing = [key for key in required if key not in args]
    if extra:
        errors.append(f"unexpected argument(s): {', '.join(extra)}")
    if missing:
        errors.append(f"missing required argument(s): {', '.join(missing)}")
    for key, value in args.items():
        spec = properties.get(key)
        if spec is None:
            continue
        expected_type = spec.get("type")
        if expected_type and not matches_json_type(value, expected_type):
            errors.append(
                f"argument '{key}' expected {expected_type}, got {json_value_type(value)}"
            )
    if fn._needs_scratchpad and scratchpad is None:
        errors.append("scratchpad is disabled for this agent")
    return ToolCallValidation(tool_call_id, name, args, errors, fn)


def tool_call_retry_prompt(
    validation_errors: list[str],
    tool_schemas: list[dict],
    tool_mode: str,
) -> str:
    if tool_mode == "lfm":
        format_hint = (
            "Return only Liquid LFM tool-call syntax: "
            "<|tool_call_start|>[tool_name(arg_name=value)]<|tool_call_end|>."
        )
    elif tool_mode == "api":
        format_hint = "Use the provided tool-call interface."
    else:
        format_hint = 'Return only a JSON array: [{"tool":"tool_name","args":{"param":"value"}}].'
    return (
        "Your previous tool-call batch failed schema validation. "
        "Regenerate the entire batch; none of the previous calls were executed.\n"
        f"Validation errors: {json.dumps(validation_errors, sort_keys=True)}\n"
        f"Available tool schemas: {json.dumps(tool_schemas, sort_keys=True)}\n"
        "Use exact tool names and argument names, include every required argument, "
        "do not include extra arguments, and match the JSON types in the schema. "
        f"{format_hint}"
    )


def matches_json_type(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def json_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    return type(value).__name__
