"""
TTS smoke test. Synthesizes a phrase and plays it through the same
AudioPlayer the live pipeline uses.

Usage:
    python -m test_tts
    python -m test_tts -c config.yaml -t "hello from kokoro"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from src.audio import AudioPlayer
from src.config import Config
from src.tts import RealtimeTTSEngine


async def _amain(cfg_path: Path, text: str, voice_override: str | None) -> int:
    cfg = Config.load(cfg_path)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("tts-test")

    voice = voice_override or cfg.tts.voice
    tts = RealtimeTTSEngine(
        engine_name=cfg.tts.engine,
        voice=voice,
        piper_model=cfg.tts.piper_model,
        sample_rate=cfg.tts.sample_rate,
    )
    player = AudioPlayer(output_device=cfg.audio.output_device)

    try:
        log.info("synthesizing %r via %s/%s", text, cfg.tts.engine, voice)
        pcm, sr = await tts.synthesize(text)
        log.info("got %d samples @ %d Hz (%.2fs)", pcm.size, sr, pcm.size / max(sr, 1))
        if pcm.size == 0:
            log.error("TTS produced empty audio")
            return 1
        log.info("playing...")
        await player.play(pcm, sr)
        log.info("done")
    finally:
        await tts.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS smoke test")
    parser.add_argument("-c", "--config", default="config.yaml", type=Path)
    parser.add_argument(
        "-t",
        "--text",
        default="Hello, this is a TikTok TTS smoke test. If you can hear me, the pipeline is wired up correctly.",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default=None,
        help="Override tts.voice from config (e.g. bm_george, af_bella).",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_amain(args.config, args.text, args.voice))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
