from __future__ import annotations

import unittest

from src.moderator import TikTokModerator


class Bridge:
    def __init__(self) -> None:
        self.calls = []

    async def mute(self, *args):
        self.calls.append(("mute", *args))

    async def block(self, *args):
        self.calls.append(("block", *args))


class ModeratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_punishment_is_refused(self) -> None:
        bridge = Bridge()
        moderator = TikTokModerator(bridge, 300)
        self.assertFalse(await moderator.punish("u", "skip"))
        self.assertEqual(bridge.calls, [])

    async def test_block_is_dispatched(self) -> None:
        bridge = Bridge()
        moderator = TikTokModerator(bridge, 300)
        self.assertTrue(await moderator.punish("u", "block"))
        self.assertEqual(bridge.calls, [("block", "u")])
