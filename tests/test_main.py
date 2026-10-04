from __future__ import annotations

import asyncio
import signal
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src import main as app


class MainShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        stack = ExitStack()
        self.addCleanup(stack.close)
        cfg = SimpleNamespace(
            tiktok=SimpleNamespace(username="test", session_id="", tt_target_idc=""),
            tts=SimpleNamespace(
                model="test", voice="test", language_code="a", speed=1,
                followers_only=False,
            ),
            moderation=SimpleNamespace(enabled=False),
            audio=SimpleNamespace(output_device=None),
            db=SimpleNamespace(path=":memory:"),
            logging=SimpleNamespace(level="INFO"),
        )
        stack.enter_context(patch.object(app.Config, "load", return_value=cfg))
        stack.enter_context(patch.object(app, "_setup_logging"))
        self.cleanup_calls = []
        for name, methods in (
            ("TikTokBridge", ("start", "shutdown")),
            ("KokoroTTSEngine", ("load", "warmup", "close")),
            ("EventLog", ("open", "close")),
            ("Pipeline", ("run", "drain")),
        ):
            instance = MagicMock()
            for method in methods:
                setattr(instance, method, AsyncMock())
            stack.enter_context(patch.object(app, name, return_value=instance))
            setattr(self, name, instance)
        stack.enter_context(patch.object(app, "AudioPlayer"))
        for mock, label in (
            (self.Pipeline.drain, "drain"),
            (self.KokoroTTSEngine.close, "tts.close"),
            (self.TikTokBridge.shutdown, "bridge.shutdown"),
            (self.EventLog.close, "log.close"),
        ):
            async def record(label=label):
                self.cleanup_calls.append(label)

            mock.side_effect = record
        self.signals = {}

        def register(sig, callback):
            self.signals[sig] = callback

        stack.enter_context(patch.object(
            asyncio.get_running_loop(), "add_signal_handler", side_effect=register,
        ))

    def assert_cleaned_up(self) -> None:
        self.assertEqual(
            self.cleanup_calls,
            ["drain", "tts.close", "bridge.shutdown", "log.close"],
        )
        self.assertEqual(asyncio.all_tasks(), {asyncio.current_task()})

    async def test_runner_exception_returns_one_after_cleanup(self) -> None:
        self.Pipeline.run.side_effect = RuntimeError("messages failed")
        with self.assertLogs(app.log, level="ERROR") as captured:
            code = await app._amain(Path("config.yaml"))
        self.assertEqual(code, 1)
        self.assertIn("pipeline runner crashed", captured.output[0])
        self.assertIn("messages failed", captured.output[0])
        self.assert_cleaned_up()

    async def test_signal_stop_returns_zero_and_cancels_runner(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig):
                self.cleanup_calls.clear()
                cancelled = asyncio.Event()

                async def run():
                    self.signals[sig]()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()

                self.Pipeline.run.side_effect = run
                self.assertEqual(await app._amain(Path("config.yaml")), 0)
                self.assertTrue(cancelled.is_set())
                self.assert_cleaned_up()

    async def test_runner_failure_during_stop_still_returns_one(self) -> None:
        async def run():
            self.signals[signal.SIGTERM]()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise RuntimeError("runner shutdown failed")

        self.Pipeline.run.side_effect = run
        with self.assertLogs(app.log, level="ERROR"):
            self.assertEqual(await app._amain(Path("config.yaml")), 1)
        self.assert_cleaned_up()

    async def test_normal_runner_completion_returns_zero(self) -> None:
        self.assertEqual(await app._amain(Path("config.yaml")), 0)
        self.assert_cleaned_up()

    async def test_outer_cancellation_propagates_after_cleanup(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def run():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.Pipeline.run.side_effect = run
        task = asyncio.create_task(app._amain(Path("config.yaml")))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cancelled.is_set())
        self.assert_cleaned_up()


class MainExitTests(unittest.TestCase):
    def test_main_preserves_exit_code(self) -> None:
        for code in (0, 1):
            with self.subTest(code=code), patch("sys.argv", ["tiktok-tts-mod"]), \
                    patch.object(app, "_amain", new_callable=AsyncMock, return_value=code):
                self.assertEqual(app.main(), code)

    def test_keyboard_interrupt_returns_130(self) -> None:
        def interrupt(coro):
            coro.close()
            raise KeyboardInterrupt

        with patch("sys.argv", ["tiktok-tts-mod"]), \
                patch.object(app.asyncio, "run", side_effect=interrupt):
            self.assertEqual(app.main(), 130)
