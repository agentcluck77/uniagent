# Requires speech extra: uv pip install -e ".[speech]"
# Pass --config for agent/model settings (e.g. uniagent/config.yaml).
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.
#
# This harness avoids live microphone input. It synthesizes typed text to audio,
# runs that audio through the same VAD + STT path, then speaks the agent response.

import argparse

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, SpeechTestHarness, load_speech_config


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
        harness = SpeechTestHarness(pipeline)
        while True:
            typed = input("typed input> ").strip()
            if typed.lower() in {"quit", "exit", "stop"}:
                break
            if not typed:
                continue

            input_audio = harness.synthesize_input(typed)
            harness.play_audio(input_audio)
            transcript = harness.transcribe_audio(input_audio)
            print(f"stt transcript> {transcript}")

            response = agent.run(transcript, on_event=pipeline.make_on_event())
            pipeline.wait_for_tts()
            print(f"agent text> {response}")


if __name__ == "__main__":
    main()
