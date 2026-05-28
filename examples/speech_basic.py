# Requires speech extra: uv pip install -e ".[speech]"
# Pass --config for agent/model settings (e.g. wnt/agents/configs/gemma4_e2b_wnt_v6_q4km.yaml).
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.

import argparse

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, load_speech_config, make_confirm_fn


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
        confirm_fn = make_confirm_fn(pipeline)
        while True:
            text = pipeline.listen()
            if text.lower() in {"quit", "exit", "stop"}:
                break
            agent.run(text, on_event=pipeline.make_on_event(), confirm_fn=confirm_fn)
            pipeline.wait_for_tts()


if __name__ == "__main__":
    main()
