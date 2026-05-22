# Requires Ollama running locally: ollama run gemma4:e2b
# Requires speech extra: uv pip install -e ".[speech]"
# Agent/model settings come from ../wnt/agents/config_agent_speech.yaml.
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, load_speech_config
from uniagent.tui import run_speech_tui


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


def main() -> None:
    config = load_config("../wnt/agents/config_agent_speech.yaml")
    speech_config = load_speech_config("config_speech.yaml")
    agent = Agent.from_config(config, tools=[multiply])

    with SpeechPipeline(speech_config) as pipeline:
        run_speech_tui(agent, pipeline)


if __name__ == "__main__":
    main()
