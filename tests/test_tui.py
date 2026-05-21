from __future__ import annotations

import time
import unittest
from typing import Any, cast

try:
    from uniagent.tui import DebugApp
except ImportError:
    DebugApp = None  # type: ignore[assignment]


class FakeAgent:
    config = {
        "agent": {"confirm_tools": True},
        "model": {
            "backend": "ollama",
            "model_name": "gemma4:e2b",
            "tool_mode": "api",
        },
    }


@unittest.skipIf(DebugApp is None, "textual is not installed")
class TUIConfirmationTests(unittest.TestCase):
    def test_confirmation_times_out_with_denial_string(self) -> None:
        assert DebugApp is not None
        app = DebugApp(FakeAgent())
        app._confirm_timeout_s = 0.01
        logs: list[str] = []
        cast(Any, app)._log = logs.append
        cast(Any, app).call_from_thread = lambda fn, *args: fn(*args)

        started = time.monotonic()
        response = app._make_confirm_fn()("Can I use multiply?")

        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(response, "no: confirmation timed out")
        self.assertFalse(app._confirm_pending)
        self.assertTrue(app._confirm_event.is_set())
        self.assertTrue(any("timed out" in line for line in logs))

    def test_unmount_cancels_pending_confirmation(self) -> None:
        assert DebugApp is not None
        app = DebugApp(FakeAgent())
        app._confirm_pending = True
        app._confirm_event.clear()

        app.on_unmount()

        self.assertEqual(app._confirm_response, "no: TUI closed")
        self.assertFalse(app._confirm_pending)
        self.assertTrue(app._confirm_event.is_set())


if __name__ == "__main__":
    unittest.main()
