# uniagent

Minimal Python agent framework with decorator-defined tools, OpenAI-compatible model backends, optional ToolRAG, per-run scratchpad, and an optional Textual debug TUI.

## Install

```bash
uv venv
uv pip install -e .
uv pip install ruff pyright
```

For the debug TUI:

```bash
uv pip install -e ".[tui]"
```

For speech:

```bash
uv pip install -e ".[speech]"
```

## Configure

Edit `config.yaml`:

```yaml
model:
  backend: ollama
  base_url: http://localhost:11434/v1
  model_name: gemma4:e2b
  api_key: null
```

For Ollama, make sure the model is available:

```bash
ollama pull gemma4:e2b
ollama run gemma4:e2b
```

## Basic Usage

```python
from uniagent import Agent, load_config, tool

@tool(description="Multiply two integers")
def multiply(a: int, b: int) -> str:
    return str(a * b)

@tool(description="Store a value in the scratchpad")
def remember(key: str, value: str, scratchpad: dict) -> str:
    scratchpad[key] = value
    return f"stored {key}"

config = load_config("config.yaml")
agent = Agent.from_config(config, tools=[multiply, remember])

print(agent.run("What is 6 times 9? Remember the result as 'answer'."))
```

Run the included example:

```bash
.venv/bin/python examples/basic.py
```

## Debug TUI

```python
from uniagent.tui import run_tui

run_tui(agent)
```

Press `q` to quit.

## Speech

Download models into `uniagent/models/` (run from the repo root):

```bash
# Piper TTS voice (~60 MB)
mkdir -p uniagent/models/piper
curl -L -o uniagent/models/piper/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o uniagent/models/piper/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json

# Whisper STT model (~75 MB)
mkdir -p uniagent/models/whisper
curl -L -o uniagent/models/whisper/ggml-tiny.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin
```

`config_speech.yaml` points to these paths by default. Piper requires the `.onnx` and its
matching `.onnx.json` in the same directory. Browse other voices at
`https://huggingface.co/rhasspy/piper-voices`.

Then run:

```bash
.venv/bin/python examples/speech_basic.py
```

To test the speech stack without a live microphone, use the typed-input harness:

```bash
.venv/bin/python scripts/speech_pipeline_harness.py
```

That harness loads WNT agent/model settings from `../wnt/agents/config_agent_speech.yaml` and
speech-device settings from `config_speech.yaml`, so it can test the same WNT model through STT/TTS.
