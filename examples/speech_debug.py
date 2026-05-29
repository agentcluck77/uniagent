# Requires speech extra: uv pip install -e ".[speech]"
# Pass --config for agent/model settings (e.g. uniagent/config.yaml).
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.

import argparse

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, load_speech_config
from uniagent.tui import run_speech_tui


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Agent/model config YAML")
    parser.add_argument("--speech-config", default="config_speech.yaml", help="Speech config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    speech_config = load_speech_config(args.speech_config)
    agent = Agent.from_config(config, tools=[multiply])

    with SpeechPipeline(speech_config) as pipeline:
        run_speech_tui(agent, pipeline)


if __name__ == "__main__":
    main()
