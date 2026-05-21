# Requires Ollama running locally: ollama run gemma4:e2b
# Requires speech extra: uv pip install -e ".[speech]"
# Agent/model settings come from ../wnt/agents/config_agent_speech.yaml.
# Requires config_speech.yaml tts.model_path to point at a Piper .onnx voice.
#
# This harness avoids live microphone input. It synthesizes typed text to audio,
# runs that audio through the same VAD + STT path, then speaks the agent response.

from speech_harness import SpeechTestHarness

from uniagent import Agent, load_config, tool
from uniagent.speech import SpeechPipeline, load_speech_config


@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)


def main() -> None:
    config = load_config("../wnt/agents/config_agent_speech.yaml")
    speech_config = load_speech_config("config_speech.yaml")
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
