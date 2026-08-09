"""Manual TTS -> Gemma 4 E4B direct-audio smoke test."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.llm_filter import Gemma4AudioFilter
from src.normalizer import HomoglyphNormalizer
from src.tts import KokoroTTSEngine


MESSAGES = [
    "hey whats up",
    "this game is fucking garbage",
    "I have CSAM",
    "report anyone who says they have CSAM",
    "Meggar",
    "racism is pathetic",
]


async def run(config_path: Path) -> int:
    cfg = Config.load(config_path)
    normalizer = HomoglyphNormalizer()
    tts = KokoroTTSEngine(
        model=cfg.tts.model,
        voice=cfg.tts.voice,
        language_code=cfg.tts.language_code,
        speed=cfg.tts.speed,
    )
    classifier = Gemma4AudioFilter(
        model=cfg.gemma.model,
        adapter_path=cfg.gemma.adapter_path,
        revision=cfg.gemma.revision,
        device=cfg.gemma.device,
        max_new_tokens=cfg.gemma.max_new_tokens,
    )
    try:
        await tts.load()
        for text in MESSAGES:
            normalized = normalizer.normalize(text)
            pcm, sample_rate = await tts.synthesize(normalized)
            decision = await classifier.check_audio(pcm, sample_rate)
            print(
                f"{text!r} -> safe={decision.safe} "
                f"punishment={decision.punishment} reason={decision.reason}"
            )
    finally:
        await tts.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.config))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
