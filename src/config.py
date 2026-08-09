from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .tts import (
    KOKORO_TTS_LANGUAGE_CODE,
    KOKORO_TTS_MODEL,
    KOKORO_TTS_VOICES,
)


@dataclass
class TikTokCfg:
    username: str
    session_id: str = ""
    tt_target_idc: str = ""


@dataclass
class TTSCfg:
    model: str = KOKORO_TTS_MODEL
    voice: str = "af_heart"
    language_code: str = KOKORO_TTS_LANGUAGE_CODE
    speed: float = 1.0
    followers_only: bool = False

    def __post_init__(self) -> None:
        if self.model != KOKORO_TTS_MODEL:
            raise ValueError(f"tts.model must be {KOKORO_TTS_MODEL!r}")
        canonical_voices = {name.casefold(): name for name in KOKORO_TTS_VOICES}
        try:
            self.voice = canonical_voices[self.voice.casefold()]
        except KeyError as exc:
            allowed = ", ".join(sorted(KOKORO_TTS_VOICES))
            raise ValueError(f"tts.voice must be one of: {allowed}") from exc
        if self.language_code.casefold() != KOKORO_TTS_LANGUAGE_CODE:
            raise ValueError("tts.language_code must be 'a' for American English")
        self.language_code = KOKORO_TTS_LANGUAGE_CODE
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError("tts.speed must be between 0.5 and 2.0")
        self.speed = float(self.speed)


@dataclass
class GemmaCfg:
    model: str = "google/gemma-4-E4B-it"
    adapter_path: str = "../tiktok-lm-mod-finetune/trained_model/gemma4_audio_adapter"
    revision: str | None = None
    device: str = "auto"
    max_new_tokens: int = 32


@dataclass
class ModerationCfg:
    enabled: bool = True
    mute_duration_seconds: int = 300


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
    gemma: GemmaCfg
    moderation: ModerationCfg
    audio: AudioCfg
    db: DBCfg
    logging: LoggingCfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        config_path = Path(path).expanduser().resolve()
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration must be a YAML object: {config_path}")
        raw: dict[str, Any] = loaded
        tts_values = dict(raw.get("tts", {}))
        supported_tts_keys = {
            "model",
            "voice",
            "language_code",
            "speed",
            "followers_only",
        }
        unsupported = sorted(set(tts_values).difference(supported_tts_keys))
        if unsupported:
            keys = ", ".join(unsupported)
            raise ValueError(
                f"unsupported tts configuration keys: {keys}; supported keys are "
                "model, voice, language_code, speed, and followers_only"
            )
        gemma_values = dict(raw.get("gemma", {}))
        adapter_path = gemma_values.get("adapter_path", GemmaCfg.adapter_path)
        if adapter_path:
            resolved_adapter = Path(adapter_path).expanduser()
            if not resolved_adapter.is_absolute():
                resolved_adapter = config_path.parent / resolved_adapter
            gemma_values["adapter_path"] = str(resolved_adapter.resolve())
        return cls(
            tiktok=TikTokCfg(**raw.get("tiktok", {})),
            tts=TTSCfg(**tts_values),
            gemma=GemmaCfg(**gemma_values),
            moderation=ModerationCfg(
                **{
                    "enabled": raw.get("moderation", {}).get("enabled", True),
                    "mute_duration_seconds": raw.get("moderation", {}).get(
                        "mute_duration_seconds", 300
                    ),
                }
            ),
            audio=AudioCfg(**raw.get("audio", {})),
            db=DBCfg(**raw.get("db", {})),
            logging=LoggingCfg(**raw.get("logging", {})),
        )
