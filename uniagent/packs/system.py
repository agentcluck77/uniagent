import subprocess
import urllib.request
from pathlib import Path

from uniagent.tool import tool


@tool(description="Run a shell command and return stdout and stderr")
def shell_run(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout + result.stderr
    except Exception as error:
        return f"Error: {error}"


@tool(description="Read a file path and return contents as a string")
def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as error:
        return f"Error: {error}"


@tool(description="Write string content to a file path")
def write_file(path: str, content: str) -> str:
    try:
        Path(path).write_text(content, encoding="utf-8")
        return "ok"
    except Exception as error:
        return f"Error: {error}"


@tool(description="HTTP GET a URL and return the response body")
def http_get(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as error:
        return f"Error: {error}"
