"""
End-to-end pipeline smoke test.

Usage:
    python test_pipeline.py
    python test_pipeline.py --config config.yaml

Each message goes through:
    raw text
      -> normalize()      unicode / homoglyph collapse
      -> synthesize()     TTS -> (pcm, sample_rate) buffer, NOT played
      -> transcribe()     Distil-Whisper STT -> spoken transcript
      -> check()          local LLM safety decision -> FilterDecision
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Ensure UTF-8 output on Windows regardless of console code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.normalizer import HomoglyphNormalizer
from src.tts import RealtimeTTSEngine
from src.stt import FasterWhisperSTT
from src.llm_filter import OpenAICompatibleFilter, FilterDecision

# ---------------------------------------------------------------------------
# Module-level singletons (initialised in main before tests run)
# ---------------------------------------------------------------------------

_normalizer: HomoglyphNormalizer | None = None
_tts: RealtimeTTSEngine | None = None
_stt: FasterWhisperSTT | None = None
_llm: OpenAICompatibleFilter | None = None


class Audio(NamedTuple):
    pcm: np.ndarray
    sample_rate: int


def normalize(text: str) -> str:
    assert _normalizer is not None, "call init() first"
    return _normalizer.normalize(text)


async def synthesize(text: str) -> Audio:
    assert _tts is not None, "call init() first"
    pcm, sample_rate = await _tts.synthesize(text)
    return Audio(pcm, sample_rate)


async def transcribe(audio: Audio) -> str:
    assert _stt is not None, "call init() first"
    return await _stt.transcribe(audio.pcm, audio.sample_rate)


async def check(transcript: str) -> FilterDecision:
    assert _llm is not None, "call init() first"
    return await _llm.check(transcript)


# ---------------------------------------------------------------------------
# Test messages
# ---------------------------------------------------------------------------

test_messages = [
    "hey whats up",
    "n1gg3r",
    "уou аre ѕtuрid",   # Cyrillic lookalikes
    "f u c k y o u",
    "totally normal message",
    "kys",
    "Pho q migour",
    "I support cake a k",
    "I hate migours",
    "are you a migour?",
    "fuck you I'm gonna dox you bitch",
    "you should kill yourself now you fat fucking n1gg3r bitch",
    "a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a",
    "what do you think about michael jackson?",
    "yo how u doing?",
    "so many people saying slurs in here man",
]

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
ORANGE = "\033[33;1m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def _verdict(decision: FilterDecision) -> str:
    if decision.safe:
        return f"{GREEN}SAFE{RESET}"
    if decision.punishment == "skip":
        return f"{ORANGE}SKIP{RESET}"
    color = RED if decision.punishment == "block" else YELLOW
    return f"{color}UNSAFE  punishment={decision.punishment}{RESET}"


async def test() -> None:
    width = max(len(m) for m in test_messages) + 4

    print(f"\n{'─' * 72}")
    print(f"  {'INPUT':<{width}}  {'NORMALIZED':<{width}}  TRANSCRIPT  →  VERDICT")
    print(f"{'─' * 72}")

    for msg in test_messages:
        normed = normalize(msg)
        audio  = await synthesize(normed)
        spoken = await transcribe(audio)
        result = await check(spoken)

        verdict_str = _verdict(result)
        reason_str  = f"  {DIM}({result.reason}){RESET}" if result.reason else ""
        print(
            f"  {msg!r:<{width}}"
            f"  {DIM}{normed!r:<{width}}{RESET}"
            f"  {spoken!r:<30}"
            f"  {verdict_str}{reason_str}"
        )

    print(f"{'─' * 72}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main(cfg_path: Path) -> None:
    global _normalizer, _tts, _stt, _llm

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Surface our own progress messages only
    logging.getLogger("src.stt").setLevel(logging.INFO)
    logging.getLogger("src.tts").setLevel(logging.INFO)

    cfg = Config.load(cfg_path)

    print(f"{BOLD}Initialising pipeline…{RESET}")
    _normalizer = HomoglyphNormalizer()
    _tts = RealtimeTTSEngine(
        engine_name=cfg.tts.engine,
        voice=cfg.tts.voice,
        piper_model=cfg.tts.piper_model,
        sample_rate=cfg.tts.sample_rate,
    )
    _stt = FasterWhisperSTT(
        model=cfg.stt.model,
        device=cfg.stt.device,
        compute_type=cfg.stt.compute_type,
    )
    _llm = OpenAICompatibleFilter(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
        temperature=cfg.llm.temperature,
        request_timeout=cfg.llm.request_timeout,
        local_blocklist=cfg.moderation.blocklist,
        think=cfg.llm.think,
    )
    print(f"LLM endpoint : {cfg.llm.base_url}  model={cfg.llm.model}")
    print(f"TTS engine   : {cfg.tts.engine}  voice={cfg.tts.voice}")
    print(f"STT model    : {cfg.stt.model}\n")

    await test()

    await _tts.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline smoke test")
    parser.add_argument("-c", "--config", default="config.yaml", type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.config))
