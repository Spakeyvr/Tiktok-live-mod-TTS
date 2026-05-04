from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import warnings
from pathlib import Path

# Suppress PyTorch's FutureWarning about the deprecated weight_norm API used
# internally by Kokoro — we can't fix third-party model code.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
# Suppress HuggingFace Hub's unauthenticated-request UserWarning.
warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*", category=UserWarning)

from .audio import AudioPlayer
from .bridge import TikTokBridge
from .config import Config
from .db import EventLog
from .llm_filter import OpenAICompatibleFilter
from .moderator import TikTokModerator
from .normalizer import HomoglyphNormalizer
from .pipeline import Pipeline
from .stt import FasterWhisperSTT
from .tts import RealtimeTTSEngine

log = logging.getLogger("tiktok-tts-mod")


def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    # Root logger at WARNING so third-party libraries that log to the root
    # (RealtimeTTS, NLTK, httpx) don't flood the console.
    logging.basicConfig(level=logging.WARNING, format=fmt)

    # Our own loggers at the user-configured level.
    for name in ("tiktok-tts-mod", "src"):
        logging.getLogger(name).setLevel(lvl)

    # Silence chatty third-party loggers that aren't useful at runtime.
    for noisy in ("httpx", "huggingface_hub", "huggingface_hub.utils._http"):
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
    tts = RealtimeTTSEngine(
        engine_name=cfg.tts.engine,
        voice=cfg.tts.voice,
        piper_model=cfg.tts.piper_model,
        sample_rate=cfg.tts.sample_rate,
    )
    if cfg.moderation.enabled:
        stt = FasterWhisperSTT(
            model=cfg.stt.model,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
        )
        llm = OpenAICompatibleFilter(
            base_url=cfg.llm.base_url,
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            request_timeout=cfg.llm.request_timeout,
            local_blocklist=cfg.moderation.blocklist,
            think=cfg.llm.think,
        )
        moderator = TikTokModerator(bridge, cfg.moderation.mute_duration_seconds)
    else:
        log.info("moderation disabled: TTS-only mode")
        stt = None
        llm = None
        moderator = None
    player = AudioPlayer(output_device=cfg.audio.output_device)
    event_log = EventLog(cfg.db.path)
    await event_log.open()

    pipeline = Pipeline(
        bridge=bridge,
        normalizer=normalizer,
        tts=tts,
        stt=stt,
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

    await bridge.start()

    runner = asyncio.create_task(pipeline.run())
    waiter = asyncio.create_task(stop.wait())
    done, _pending = await asyncio.wait(
        {runner, waiter}, return_when=asyncio.FIRST_COMPLETED
    )

    log.info("shutting down...")
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass
    except Exception:
        log.exception("pipeline runner crashed")

    await pipeline.drain()
    await tts.close()
    await bridge.shutdown()
    await event_log.close()
    return 0


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
