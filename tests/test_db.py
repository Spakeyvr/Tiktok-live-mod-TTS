from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.db import EventLog


class EventLogTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_and_insert_match_direct_audio_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.db"
            event_log = EventLog(path)
            await event_log.open()
            try:
                await event_log.log(
                    msg_id="1",
                    user_id="u1",
                    unique_id="viewer",
                    nickname="Viewer",
                    raw_comment="hello",
                    normalized="hello",
                    safe=True,
                    punishment=None,
                    reason="allowed",
                    llm_raw='{"safe":true}',
                    action_ok=None,
                )
            finally:
                await event_log.close()

            with sqlite3.connect(path) as connection:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(events)")]
                row = connection.execute(
                    "SELECT msg_id, normalized, safe, punishment, reason FROM events"
                ).fetchone()

            self.assertEqual(
                columns,
                [
                    "id",
                    "ts",
                    "msg_id",
                    "user_id",
                    "unique_id",
                    "nickname",
                    "raw_comment",
                    "normalized",
                    "safe",
                    "punishment",
                    "reason",
                    "llm_raw",
                    "action_ok",
                ],
            )
            self.assertEqual(row, ("1", "hello", 1, None, "allowed"))


if __name__ == "__main__":
    unittest.main()
