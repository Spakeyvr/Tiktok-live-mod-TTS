from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TikTokCfg:
    username: str
    session_id: str = ""
    tt_target_idc: str = ""


@dataclass
class TTSCfg:
    engine: str = "kokoro"
    voice: str = "af_sarah"
    piper_model: str = ""
    sample_rate: int = 24000


@dataclass
class STTCfg:
    model: str = "distil-whisper/distil-small.en"
    device: str = "auto"
    compute_type: str = "int8"


@dataclass
class LLMCfg:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    model: str = "qwen3.5:4b"
    temperature: float = 0.0
    request_timeout: int = 30
    # Qwen3 thinking mode: False disables <think> reasoning via extra_body.
    think: bool = False


@dataclass
class ModerationCfg:
    enabled: bool = True
    mute_duration_seconds: int = 300
    blocklist: list[str] = field(default_factory=list)


@dataclass
class AudioCfg:
    output_device: int | None = None


@dataclass
class DBCfg:
    path: str = "moderation.db"


@dataclass
class LoggingCfg:
    level: str = "INFO"


@dataclass
class Config:
    tiktok: TikTokCfg
    tts: TTSCfg
    stt: STTCfg
    llm: LLMCfg
    moderation: ModerationCfg
    audio: AudioCfg
    db: DBCfg
    logging: LoggingCfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            tiktok=TikTokCfg(**raw.get("tiktok", {})),
            tts=TTSCfg(**raw.get("tts", {})),
            stt=STTCfg(**raw.get("stt", {})),
            llm=LLMCfg(**raw.get("llm", {})),
            moderation=ModerationCfg(
                **{
                    "enabled": raw.get("moderation", {}).get("enabled", True),
                    "mute_duration_seconds": raw.get("moderation", {}).get(
                        "mute_duration_seconds", 300
                    ),
                    "blocklist": [
                        s for s in raw.get("moderation", {}).get("blocklist", []) if s
                    ],
                }
            ),
            audio=AudioCfg(**raw.get("audio", {})),
            db=DBCfg(**raw.get("db", {})),
            logging=LoggingCfg(**raw.get("logging", {})),
        )
