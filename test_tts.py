"""
TTS smoke test. Synthesizes a phrase and plays it through the same
AudioPlayer the live pipeline uses.

Usage:
    python3 -m test_tts
    python3 -m test_tts -c config.yaml -t "hello from Kokoro"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from src.audio import AudioPlayer
from src.config import Config
from src.tts import KOKORO_TTS_VOICES, KokoroTTSEngine


async def _amain(
    cfg_path: Path,
    text: str,
    voice_override: str | None,
    *,
    play: bool,
) -> int:
    cfg = Config.load(cfg_path)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    log = logging.getLogger("tts-test")

    voice = voice_override or cfg.tts.voice
    tts = KokoroTTSEngine(
        model=cfg.tts.model,
        voice=voice,
        language_code=cfg.tts.language_code,
        speed=cfg.tts.speed,
    )
    player = AudioPlayer(output_device=cfg.audio.output_device)

    try:
        load_started = time.perf_counter()
        await tts.load()
        log.info(
            "loaded %s in %.2fs",
            cfg.tts.model,
            time.perf_counter() - load_started,
        )
        log.info("warming Kokoro...")
        await tts.warmup()
        log.info(
            "synthesizing %r with voice=%s speed=%g",
            text,
            voice,
            cfg.tts.speed,
        )
        generation_started = time.perf_counter()
        pcm, sr = await tts.synthesize(text)
        generation_seconds = time.perf_counter() - generation_started
        audio_seconds = pcm.size / max(sr, 1)
        real_time_factor = generation_seconds / max(audio_seconds, 0.001)
        log.info(
            "got %d samples @ %d Hz (%.2fs audio) in %.2fs, RTF=%.3f",
            pcm.size,
            sr,
            audio_seconds,
            generation_seconds,
            real_time_factor,
        )
        if pcm.size == 0:
            log.error("TTS produced empty audio")
            return 1
        if play:
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
        choices=sorted(KOKORO_TTS_VOICES),
        help="Override tts.voice from config.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Synthesize and report timing without playing the generated audio.",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(
            _amain(args.config, args.text, args.voice, play=not args.no_play)
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
