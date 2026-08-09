from __future__ import annotations

import logging
from typing import Protocol

from .bridge import TikTokBridge

log = logging.getLogger(__name__)


class Moderator(Protocol):
    async def punish(self, user_id: str, punishment: str) -> bool: ...


class TikTokModerator:
    """Issues mute/block via the Node bridge.

    Returns True if the command was dispatched. Whether TikTok actually
    applied the action is reported asynchronously via the bridge's `ack`
    events (logged by TikTokBridge).
    """

    def __init__(self, bridge: TikTokBridge, mute_duration_seconds: int) -> None:
        self.bridge = bridge
        self.mute_duration_seconds = mute_duration_seconds

    async def punish(self, user_id: str, punishment: str) -> bool:
        if not user_id:
            log.warning("cannot punish: missing user_id (punishment=%s)", punishment)
            return False
        try:
            if punishment == "mute":
                await self.bridge.mute(user_id, self.mute_duration_seconds)
            elif punishment == "block":
                await self.bridge.block(user_id)
            else:
                log.error("refusing unknown punishment %r", punishment)
                return False
            return True
        except Exception as e:
            log.error("moderation dispatch failed: %s", e)
            return False
