# Requires Ollama running locally: ollama run gemma4:e2b

import argparse

from uniagent import Agent, load_config, tool


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


@tool(description="Store a value in the scratchpad")
def remember(key: str, value: str, scratchpad: dict) -> str:
    scratchpad[key] = value
    return f"stored {key}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tui", action="store_true", help="Launch the Textual debug TUI")
    args = parser.parse_args()

    config = load_config("config.yaml")
    agent = Agent.from_config(config, tools=[multiply, remember])
    if args.tui:
        from uniagent.tui import run_tui

        run_tui(agent)
        return
    result = agent.run("What is 6 times 9? Remember the result as 'answer'.")
    print(result)


if __name__ == "__main__":
    main()
