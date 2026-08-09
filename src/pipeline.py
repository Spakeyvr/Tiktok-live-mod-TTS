from __future__ import annotations

import asyncio
import logging

from .audio import AudioPlayer
from .bridge import ChatMessage, TikTokBridge
from .db import EventLog
from .llm_filter import AudioFilter
from .moderator import Moderator
from .normalizer import Normalizer
from .tts import TTSEngine

log = logging.getLogger(__name__)


class Pipeline:
    """Per-message pipeline:
        normalize -> TTS (buffer) -> Gemma 4 audio filter -> moderate / play.

    Messages are processed concurrently up to `max_concurrency`. Audio
    playback is serialized inside AudioPlayer so safe clips don't overlap.
    """

    def __init__(
        self,
        bridge: TikTokBridge,
        normalizer: Normalizer,
        tts: TTSEngine,
        llm: AudioFilter | None,
        moderator: Moderator | None,
        player: AudioPlayer,
        event_log: EventLog,
        max_concurrency: int = 2,
        moderation_enabled: bool = True,
        followers_only: bool = False,
    ) -> None:
        self.bridge = bridge
        self.normalizer = normalizer
        self.tts = tts
        self.llm = llm
        self.moderator = moderator
        self.player = player
        self.event_log = event_log
        self.moderation_enabled = moderation_enabled
        self.followers_only = followers_only
        self._sem = asyncio.Semaphore(max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    async def run(self) -> None:
        async for msg in self.bridge.messages():
            task = asyncio.create_task(self._handle(msg))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _handle(self, msg: ChatMessage) -> None:
        async with self._sem:
            try:
                await self._process(msg)
            except Exception:
                log.exception("pipeline crashed on msg %s", msg.id)

    async def _process(self, msg: ChatMessage) -> None:
        if self.followers_only and not msg.is_follower:
            log.debug("skip non-follower %s", msg.unique_id)
            return

        normalized = self.normalizer.normalize(msg.comment)
        if not normalized:
            log.debug("skip empty after normalize: %s", msg.id)
            return

        log.info("[%s] %s: %r -> %r", msg.id, msg.unique_id, msg.comment, normalized)

        pcm, sample_rate = await self.tts.synthesize(normalized)
        if pcm.size == 0:
            log.debug("tts produced empty audio: %s", msg.id)
            return

        if not self.moderation_enabled:
            await self.event_log.log(
                msg_id=msg.id,
                user_id=msg.user_id,
                unique_id=msg.unique_id,
                nickname=msg.nickname,
                raw_comment=msg.comment,
                normalized=normalized,
                safe=True,
                punishment=None,
                reason="moderation disabled",
                llm_raw="",
                action_ok=None,
            )
            await self.player.play(pcm, sample_rate)
            return

        assert self.llm is not None and self.moderator is not None
        decision = await self.llm.check_audio(pcm, sample_rate)
        log.info(
            "[%s] decision safe=%s punishment=%s reason=%s",
            msg.id,
            decision.safe,
            decision.punishment,
            decision.reason,
        )

        action_ok: bool | None = None
        if not decision.safe:
            action_ok = await self.moderator.punish(msg.user_id, "block")

        await self.event_log.log(
            msg_id=msg.id,
            user_id=msg.user_id,
            unique_id=msg.unique_id,
            nickname=msg.nickname,
            raw_comment=msg.comment,
            normalized=normalized,
            safe=decision.safe,
            punishment=decision.punishment,
            reason=decision.reason,
            llm_raw=decision.raw,
            action_ok=action_ok,
        )

        if decision.safe:
            await self.player.play(pcm, sample_rate)
        # blocked audio is dropped; safe audio is played.
