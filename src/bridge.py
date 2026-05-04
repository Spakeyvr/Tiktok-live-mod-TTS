from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    id: str
    user_id: str
    unique_id: str
    nickname: str
    comment: str
    ts_ms: int


class TikTokBridge:
    """Subprocess bridge to the Node.js TikTok-Live-Connector script.

    NDJSON over stdin/stdout. Chat events arrive as ChatMessage on the queue;
    moderation commands are sent via mute()/block().
    """

    def __init__(
        self,
        username: str,
        session_id: str = "",
        tt_target_idc: str = "",
        script_path: str | Path = "bridge/tiktok_bridge.js",
        node_bin: str = "node",
    ) -> None:
        self.username = username
        self.session_id = session_id
        self.tt_target_idc = tt_target_idc
        self.script_path = Path(script_path).resolve()
        self.node_bin = node_bin
        self._proc: asyncio.subprocess.Process | None = None
        self._messages: asyncio.Queue[ChatMessage] = asyncio.Queue()
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()

    async def start(self) -> None:
        if shutil.which(self.node_bin) is None:
            raise RuntimeError(
                f"'{self.node_bin}' not found on PATH; install Node 18+ and run `npm install`."
            )
        if not self.script_path.exists():
            raise FileNotFoundError(self.script_path)

        args = [self.node_bin, str(self.script_path), "--user", self.username]
        if self.session_id:
            args += ["--sessionid", self.session_id]
        if self.tt_target_idc:
            args += ["--ttargetidc", self.tt_target_idc]

        log.info("starting TikTok bridge: %s", " ".join(args))
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        await asyncio.wait_for(self._ready.wait(), timeout=30)

    async def _read_stdout(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    log.warning("bridge: non-JSON line: %r", line)
                    continue
                etype = event.get("type")
                if etype == "ready":
                    log.info("bridge ready (room=%s)", event.get("roomId"))
                    self._ready.set()
                elif etype == "chat":
                    msg = ChatMessage(
                        id=event.get("id", ""),
                        user_id=event.get("userId", ""),
                        unique_id=event.get("uniqueId", ""),
                        nickname=event.get("nickname", ""),
                        comment=event.get("comment", ""),
                        ts_ms=int(event.get("ts", 0)),
                    )
                    await self._messages.put(msg)
                elif etype == "ack":
                    log.info(
                        "bridge ack cmd=%s user=%s ok=%s reason=%s",
                        event.get("cmd"),
                        event.get("userId"),
                        event.get("ok"),
                        event.get("reason", ""),
                    )
                elif etype == "disconnected":
                    log.warning("bridge disconnected: %s", event.get("reason", ""))
                elif etype == "error":
                    log.error("bridge error: %s", event.get("message"))
                else:
                    log.debug("bridge event: %s", event)
        finally:
            self._closed.set()

    async def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        while True:
            line = await self._proc.stderr.readline()
            if not line:
                break
            log.debug("bridge stderr: %s", line.decode("utf-8", "replace").rstrip())

    async def messages(self) -> AsyncIterator[ChatMessage]:
        while not (self._closed.is_set() and self._messages.empty()):
            try:
                yield await asyncio.wait_for(self._messages.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    async def _send(self, payload: dict) -> None:
        if not self._proc or not self._proc.stdin or self._proc.stdin.is_closing():
            raise RuntimeError("bridge not running")
        self._proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def mute(self, user_id: str, duration_seconds: int) -> None:
        await self._send(
            {"cmd": "mute", "userId": user_id, "durationSeconds": duration_seconds}
        )

    async def block(self, user_id: str) -> None:
        await self._send({"cmd": "block", "userId": user_id})

    async def shutdown(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                await self._send({"cmd": "shutdown"})
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
