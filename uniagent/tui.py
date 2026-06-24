import threading
from collections.abc import Callable
from pprint import pformat
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Input, Log, Static

CONFIRM_TIMEOUT_S = 120.0


class DebugApp(App):
    BINDINGS = [("q", "quit", "Quit")]
    CSS = """
    #main { height: 1fr; }
    #history { width: 1fr; height: 1fr; }
    #right {
        width: 36;
        height: 1fr;
        border-left: solid $primary;
    }
    #diagnostics {
        height: 1fr;
        padding: 0 1;
        border-bottom: solid $primary-darken-1;
    }
    #scratchpad {
        height: auto;
        min-height: 4;
        padding: 0 1;
    }
    #input { dock: bottom; }
    """

    def __init__(self, agent: Any):
        super().__init__()
        self._agent = agent
        self._agent_lock = threading.Lock()
        self._session_tokens = 0
        self._session_latency_ms = 0.0
        self._session_tool_calls = 0
        self._confirm_pending = False
        self._confirm_event = threading.Event()
        self._confirm_response = ""
        self._confirm_timeout_s = CONFIRM_TIMEOUT_S
        self._streaming_turn = False
        self._last_live_tps = 0.0
        self._last_stats: dict | None = None

    def on_mount(self) -> None:
        self.query_one("#input", Input).focus()
        self._refresh_diagnostics()

    def on_unmount(self) -> None:
        self._resolve_confirmation("no: TUI closed")

    def action_quit(self) -> None:
        self._resolve_confirmation("no: TUI closed")
        self.exit()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            yield Log(id="history", auto_scroll=True)
            with Vertical(id="right"):
                yield Static("", id="diagnostics")
                yield Static("Scratchpad\n(empty)", id="scratchpad")
        yield Input(placeholder="Message...", id="input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        event.input.clear()
        if not message:
            return
        if message.startswith("/"):
            if self._confirm_pending:
                self._log("[error] Cannot run a command while waiting for confirmation")
                return
            self._handle_slash_command(message)
            return
        if self._confirm_pending:
            self._resolve_confirmation(message)
            return
        self._log(f"[user] {message}")
        self._run_agent(message)

    def _handle_slash_command(self, command: str) -> None:
        if command == "/clear":
            self._do_clear(full=True)
        elif command == "/clearchat":
            self._do_clear(full=False)
        else:
            self._log(f"[error] Unknown command: {command}")

    @work(thread=True)
    def _do_clear(self, full: bool) -> None:
        with self._agent_lock:
            self._agent.clear_history()
            if full:
                if hasattr(self._agent, "clear_state"):
                    self._agent.clear_state()
                self.call_from_thread(self._reset_session_stats)
            else:
                self.call_from_thread(self.query_one("#history", Log).clear)

    def _reset_session_stats(self) -> None:
        self._session_tokens = 0
        self._session_latency_ms = 0.0
        self._session_tool_calls = 0
        self._last_stats = None
        self.query_one("#history", Log).clear()
        self.query_one("#scratchpad", Static).update("Scratchpad\n(empty)")
        self._refresh_diagnostics()

    @work(thread=True)
    def _run_agent(self, message: str) -> None:
        confirm_fn: Callable[[str], str] | None = None
        if self._agent.config["agent"]["confirm_tools"]:
            confirm_fn = self._make_confirm_fn()
        with self._agent_lock:
            try:
                self._agent.run(message, on_event=self._handle_event, confirm_fn=confirm_fn)
            except Exception as error:
                self.call_from_thread(self._log, f"[error] {error}")

    def _make_confirm_fn(self) -> Callable[[str], str]:
        def tui_confirm_fn(question: str) -> str:
            self._confirm_event.clear()
            try:
                self.call_from_thread(self._begin_confirmation, question)
            except RuntimeError:
                return "no: TUI unavailable"
            if self._confirm_event.wait(timeout=self._confirm_timeout_s):
                return self._confirm_response
            self._resolve_confirmation("no: confirmation timed out")
            try:
                self.call_from_thread(
                    self._log,
                    "[assistant] Confirmation timed out; denying tool call.",
                )
            except RuntimeError:
                pass
            return self._confirm_response

        return tui_confirm_fn

    def _begin_confirmation(self, question: str) -> None:
        self._confirm_response = "no: confirmation cancelled"
        self._confirm_pending = True
        self._log(f"[assistant] {question} (approve/deny, timeout {self._confirm_timeout_s:.0f}s)")

    def _resolve_confirmation(self, response: str) -> None:
        if not self._confirm_pending and self._confirm_event.is_set():
            return
        self._confirm_response = response
        self._confirm_pending = False
        self._confirm_event.set()

    def _handle_event(self, event: dict) -> None:
        self.call_from_thread(self._dispatch, event)

    def _dispatch(self, event: dict) -> None:
        t = event.get("type")
        log = self.query_one("#history", Log)
        if t == "token":
            tok = event.get("content", "")
            if not self._streaming_turn:
                self._streaming_turn = True
                log.write("[assistant] ")
            if tok == "\n":
                log.write("\n")
                self._streaming_turn = False
            else:
                log.write(tok)
            self._last_live_tps = event.get("tok_per_sec", 0.0)
            self._refresh_diagnostics()
        elif t == "tool_call":
            self._log(f"[tool->] {event.get('name')}({pformat(event.get('args', {}))})")
        elif t == "tool_result":
            self._log(f"[<-tool] {event.get('name')}: {event.get('content')}")
        elif t == "tool_validation_error":
            self._log(f"[tool validation] {pformat(event.get('errors', []))}")
        elif t == "scratchpad":
            self.query_one("#scratchpad", Static).update(
                "Scratchpad\n" + pformat(event.get("state", {}))
            )
        elif t == "assistant_text":
            if not self._streaming_turn:
                self._log(f"[assistant] {event.get('content')}")
        elif t == "stats":
            self._streaming_turn = False
            self._session_tokens += event.get("completion_tokens", 0)
            self._session_latency_ms += event.get("latency_ms", 0)
            self._session_tool_calls = event.get("session_tool_calls", 0)
            self._refresh_diagnostics(event)

    def _extra_diagnostics_lines(self) -> list[str]:
        return []

    def _refresh_diagnostics(self, stats: dict | None = None) -> None:
        if stats is not None:
            self._last_stats = stats
        cfg = self._agent.config
        m = cfg["model"]
        stats = self._last_stats
        avg_tps = (
            round(self._session_tokens / self._session_latency_ms * 1000, 1)
            if self._session_latency_ms > 0
            else 0.0
        )
        lines = [
            "Diagnostics",
            "─" * 24,
            f"Model:    {self._short(m['model_name'])}",
            f"Provider: {m['backend']}",
            f"Mode:     {m['tool_mode']}",
            f"Prompt:   {self._short(cfg['agent']['system_prompt'], head=True)}",
            "",
            f"Live:     {self._last_live_tps:.1f} tok/s",
            f"Avg:      {avg_tps:.1f} tok/s",
            f"Tokens:   {self._session_tokens:,}",
            f"Calls:    {self._session_tool_calls}",
        ]

        if stats:
            lines += [
                f"Last:     {stats.get('tok_per_sec', 0):.1f} tok/s",
                f"History:  {stats.get('history_depth', 0)} msgs",
                f"Iter:     {stats.get('iteration', 0)}",
                "",
                "Tools exposed:",
            ]
            for name in stats.get("active_tools", []):
                lines.append(f"  • {name}")

        lines += self._extra_diagnostics_lines()
        self.query_one("#diagnostics", Static).update("\n".join(lines))

    @staticmethod
    def _short(text: str, head: bool = False) -> str:
        if len(text) <= 28:
            return text
        return text[:25] + "..." if head else "..." + text[-25:]

    def _log(self, line: str) -> None:
        self.query_one("#history", Log).write_line(line)


class SpeechDebugApp(DebugApp):
    BINDINGS = DebugApp.BINDINGS + [("ctrl+s", "cycle_mode", "Cycle input mode")]

    def __init__(self, agent: Any, pipeline: Any):
        super().__init__(agent)
        self._pipeline = pipeline
        self._mode = "text"
        self._speech_state = "idle"
        self._recording = False
        self._ptt_stop_event: threading.Event | None = None
        self._harness: Any = None

    def on_mount(self) -> None:
        super().on_mount()

    def action_cycle_mode(self) -> None:
        _MODES = ("text", "speech", "sim")
        self._mode = _MODES[(_MODES.index(self._mode) + 1) % 3]
        self._log(f"[mode] {self._mode} (Ctrl+S)")
        self._apply_mode()
        self._refresh_diagnostics()

    def _apply_mode(self) -> None:
        inp = self.query_one("#input", Input)
        if self._mode == "speech":
            inp.disabled = True
            inp.placeholder = f"speech — {self._speech_state}"
        else:
            inp.disabled = False
            inp.placeholder = "Message..."
            inp.focus()

    def key_space(self) -> None:
        if self._mode == "speech":
            self._handle_ptt_toggle()

    def _handle_ptt_toggle(self) -> None:
        if self._speech_state == "recording":
            if self._ptt_stop_event is not None:
                self._ptt_stop_event.clear()
            self._set_speech_state("responding")
        elif self._speech_state == "speaking":
            self._pipeline.cancel_tts()
            self._start_ptt()
        elif self._speech_state == "idle":
            self._start_ptt()

    def _start_ptt(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        self._ptt_stop_event = stop_event
        self._set_speech_state("recording")
        threading.Thread(target=self._ptt_loop, args=(stop_event,), daemon=True).start()

    def _ptt_loop(self, stop_event: threading.Event) -> None:
        try:
            text = self._pipeline.listen_ptt(stop_event)
        except Exception as exc:
            self.call_from_thread(self._log, f"[speech error] {exc}")
            self._set_speech_state("idle")
            return
        if not text:
            self._set_speech_state("idle")
            return
        self.call_from_thread(self._log, f"[user] {text}")
        self._set_speech_state("responding")
        confirm_fn = self._make_confirm_fn() if self._agent.config["agent"]["confirm_tools"] else None
        with self._agent_lock:
            try:
                self._agent.run(text, on_event=self._speech_on_event(), confirm_fn=confirm_fn)
            except Exception as exc:
                self.call_from_thread(self._log, f"[error] {exc}")
        self._set_speech_state("speaking")
        self._pipeline.wait_for_tts()
        self._set_speech_state("idle")

    def _get_harness(self) -> Any:
        if self._harness is None:
            from uniagent.speech import SpeechTestHarness

            self._harness = SpeechTestHarness(self._pipeline)
        return self._harness

    @work(thread=True)
    def _run_agent(self, message: str) -> None:
        confirm_fn = self._make_confirm_fn() if self._agent.config["agent"]["confirm_tools"] else None
        if self._mode == "sim":
            harness = self._get_harness()
            try:
                input_audio = harness.synthesize_input(message)
                harness.play_audio(input_audio)
                transcript = harness.transcribe_audio(input_audio, play_chime=False)
            except Exception as exc:
                self.call_from_thread(self._log, f"[sim error] {exc}")
                return
            self.call_from_thread(self._log, f"[stt] {transcript}")
            message = transcript

        with self._agent_lock:
            try:
                self._agent.run(message, on_event=self._speech_on_event(), confirm_fn=confirm_fn)
            except Exception as exc:
                self.call_from_thread(self._log, f"[error] {exc}")
        self._pipeline.wait_for_tts()

    def _speech_on_event(self) -> Callable[[dict], None]:
        tts_on_event = self._pipeline.make_on_event()

        def combined_on_event(event: dict, _tts: Callable = tts_on_event) -> None:
            self._handle_event(event)
            _tts(event)

        return combined_on_event

    def _set_speech_state(self, state: str) -> None:
        self._speech_state = state
        try:
            self.call_from_thread(self._refresh_speech_ui)
        except RuntimeError:
            self._refresh_speech_ui()

    def _refresh_speech_ui(self) -> None:
        self._refresh_diagnostics()
        if self._mode == "speech":
            self.query_one("#input", Input).placeholder = f"speech — {self._speech_state}"

    def _extra_diagnostics_lines(self) -> list[str]:
        lines = ["", f"Mode:     {self._mode} (Ctrl+S)"]
        if self._mode == "speech":
            lines.append(f"State:    {self._speech_state}")
        return lines


def run_tui(agent: Any) -> None:
    DebugApp(agent).run()


def run_speech_tui(agent: Any, pipeline: Any) -> None:
    SpeechDebugApp(agent, pipeline).run()


__all__ = ["run_tui", "run_speech_tui"]
