# Requires Ollama running locally: ollama run gemma4:e2b
# Requires TUI extra: uv pip install -e ".[tui]"

from uniagent import Agent, load_config, tool
from uniagent.tui import run_tui


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


@tool(description="Store a value in the scratchpad")
def remember(key: str, value: str, scratchpad: dict) -> str:
    scratchpad[key] = value
    return f"stored {key}"


if __name__ == "__main__":
    config = load_config("config.yaml")
    agent = Agent.from_config(config, tools=[multiply, remember])
    run_tui(agent)
