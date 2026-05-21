# Requires Ollama running locally: ollama run gemma4:e2b
# Requires speech extra: uv pip install -e ".[speech]"
# Agent/model settings come from ../wnt/agents/config_agent_speech.yaml.
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, load_speech_config, make_confirm_fn


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


def main() -> None:
    config = load_config("../wnt/agents/config_agent_speech.yaml")
    speech_config = load_speech_config("config_speech.yaml")
    agent = Agent.from_config(config, tools=[multiply])

    with SpeechPipeline(speech_config) as pipeline:
        confirm_fn = make_confirm_fn(pipeline)
        while True:
            text = pipeline.listen()
            if text.lower() in {"quit", "exit", "stop"}:
                break
            agent.run(text, on_event=pipeline.make_on_event(), confirm_fn=confirm_fn)
            pipeline.wait_for_tts()


if __name__ == "__main__":
    main()
