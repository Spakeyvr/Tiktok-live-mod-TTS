from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL NOT NULL,
    msg_id          TEXT,
    user_id         TEXT,
    unique_id       TEXT,
    nickname        TEXT,
    raw_comment     TEXT,
    normalized      TEXT,
    transcript      TEXT,
    safe            INTEGER NOT NULL,
    punishment      TEXT,
    reason          TEXT,
    llm_raw         TEXT,
    action_ok       INTEGER
);
CREATE INDEX IF NOT EXISTS events_user ON events(user_id);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    async def open(self) -> None:
        await asyncio.to_thread(self._open_blocking)

    def _open_blocking(self) -> None:
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    async def log(
        self,
        *,
        msg_id: str,
        user_id: str,
        unique_id: str,
        nickname: str,
        raw_comment: str,
        normalized: str,
        transcript: str,
        safe: bool,
        punishment: str | None,
        reason: str,
        llm_raw: str,
        action_ok: bool | None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._insert,
                msg_id,
                user_id,
                unique_id,
                nickname,
                raw_comment,
                normalized,
                transcript,
                safe,
                punishment,
                reason,
                llm_raw,
                action_ok,
            )

    def _insert(self, *vals) -> None:
        assert self._conn is not None
        (
            msg_id,
            user_id,
            unique_id,
            nickname,
            raw_comment,
            normalized,
            transcript,
            safe,
            punishment,
            reason,
            llm_raw,
            action_ok,
        ) = vals
        self._conn.execute(
            """INSERT INTO events
               (ts, msg_id, user_id, unique_id, nickname, raw_comment, normalized,
                transcript, safe, punishment, reason, llm_raw, action_ok)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                msg_id,
                user_id,
                unique_id,
                nickname,
                raw_comment,
                normalized,
                transcript,
                1 if safe else 0,
                punishment,
                reason,
                llm_raw,
                None if action_ok is None else (1 if action_ok else 0),
            ),
        )
        self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None
