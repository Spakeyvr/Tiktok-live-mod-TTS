from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import warnings
from pathlib import Path

# Suppress HuggingFace Hub's unauthenticated-request UserWarning.
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*", category=UserWarning)

from .audio import AudioPlayer
from .bridge import TikTokBridge
from .config import Config
from .db import EventLog
from .llm_filter import Gemma4AudioFilter
from .moderator import TikTokModerator
from .normalizer import HomoglyphNormalizer
from .pipeline import Pipeline
from .tts import KokoroTTSEngine

log = logging.getLogger("tiktok-tts-mod")


def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Root logger at WARNING so third-party model libraries don't flood the console.
    logging.basicConfig(level=logging.WARNING, format=fmt)

    # Our own loggers at the user-configured level.
    for name in ("tiktok-tts-mod", "src"):
        logging.getLogger(name).setLevel(lvl)

    # Silence chatty third-party loggers that aren't useful at runtime.
    for noisy in (
        "httpx",
        "huggingface_hub",
        "huggingface_hub.utils._http",
        "phonemizer",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)


async def _amain(cfg_path: Path) -> int:
    cfg = Config.load(cfg_path)
    _setup_logging(cfg.logging.level)

    bridge = TikTokBridge(
        username=cfg.tiktok.username,
        session_id=cfg.tiktok.session_id,
        tt_target_idc=cfg.tiktok.tt_target_idc,
    )
    normalizer = HomoglyphNormalizer()
    tts = KokoroTTSEngine(
        model=cfg.tts.model,
        voice=cfg.tts.voice,
        language_code=cfg.tts.language_code,
        speed=cfg.tts.speed,
    )
    if cfg.moderation.enabled:
        llm = Gemma4AudioFilter(
            model=cfg.gemma.model,
            adapter_path=cfg.gemma.adapter_path,
            revision=cfg.gemma.revision,
            device=cfg.gemma.device,
            max_new_tokens=cfg.gemma.max_new_tokens,
        )
        llm.validate_configuration()
        moderator = TikTokModerator(bridge, cfg.moderation.mute_duration_seconds)
    else:
        log.info("moderation disabled: TTS-only mode")
        llm = None
        moderator = None
    player = AudioPlayer(output_device=cfg.audio.output_device)
    event_log = EventLog(cfg.db.path)
    await event_log.open()

    pipeline = Pipeline(
        bridge=bridge,
        normalizer=normalizer,
        tts=tts,
        llm=llm,
        moderator=moderator,
        player=player,
        event_log=event_log,
        moderation_enabled=cfg.moderation.enabled,
        followers_only=cfg.tts.followers_only,
    )

    stop = asyncio.Event()

    def _signal_stop(*_):
        log.info("shutdown signal received")
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_stop)
        except NotImplementedError:
            # Windows: signal handlers via add_signal_handler aren't supported;
            # rely on KeyboardInterrupt bubbling up to asyncio.run.
            pass

    runner: asyncio.Task | None = None
    waiter: asyncio.Task | None = None
    bridge_started = False
    exit_code = 0
    try:
        # Fail before joining the live if weights/configuration are unavailable,
        # and absorb the one-time MLX compilation cost during startup.
        await tts.load()
        await tts.warmup()
        await bridge.start()
        bridge_started = True

        runner = asyncio.create_task(pipeline.run())
        waiter = asyncio.create_task(stop.wait())
        await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        log.info("shutting down...")
        if runner is not None:
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("pipeline runner crashed")
                # Finish cleanup, but report the fatal runner failure to the caller.
                exit_code = 1
        if waiter is not None:
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                pass

        await pipeline.drain()
        await tts.close()
        if bridge_started:
            await bridge.shutdown()
        await event_log.close()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="TikTok Live TTS moderation pipeline")
    parser.add_argument("-c", "--config", default="config.yaml", type=Path)
    args = parser.parse_args()
    try:
        return asyncio.run(_amain(args.config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
