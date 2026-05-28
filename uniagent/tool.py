import inspect
from collections.abc import Callable
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _json_type(annotation: Any) -> str:
    if annotation in TYPE_MAP:
        return TYPE_MAP[annotation]
    if get_origin(annotation) in {Union, UnionType}:
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none_args) == 1:
            return _json_type(non_none_args[0])
    raise TypeError(f"unsupported tool parameter type: {annotation!r}")


def _build_schema(fn: Callable, description: str) -> dict[str, Any]:
    hints = get_type_hints(fn)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(fn).parameters.items():
        if name == "scratchpad":
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"unsupported tool parameter kind for {name}")
        properties[name] = {"type": _json_type(hints.get(name, Any))}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool(description: str, schema: dict[str, Any] | None = None):
    def decorate(fn: Callable) -> Callable:
        hints = get_type_hints(fn)
        fn._description = description
        fn._schema = schema if schema is not None else _build_schema(fn, description)
        fn._needs_scratchpad = hints.get("scratchpad") == dict
        return fn

    return decorate
