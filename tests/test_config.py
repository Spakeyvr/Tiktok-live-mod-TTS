from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import Config
from src.tts import KOKORO_TTS_MODEL


class ConfigTests(unittest.TestCase):
    def load_yaml(self, source: str) -> Config:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(source, encoding="utf-8")
            return Config.load(path)

    def test_kokoro_defaults(self) -> None:
        config = self.load_yaml("tiktok:\n  username: streamer\n")
        self.assertEqual(config.tts.model, KOKORO_TTS_MODEL)
        self.assertEqual(config.tts.voice, "af_heart")
        self.assertEqual(config.tts.language_code, "a")
        self.assertEqual(config.tts.speed, 1.0)

    def test_accepts_supported_voice_case_insensitively(self) -> None:
        config = self.load_yaml(
            "tiktok:\n  username: streamer\ntts:\n  voice: AF_BELLA\n  speed: 1.1\n"
        )
        self.assertEqual(config.tts.voice, "af_bella")
        self.assertEqual(config.tts.speed, 1.1)

    def test_rejects_retired_tts_keys_with_migration_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported tts configuration keys"):
            self.load_yaml(
                "tiktok:\n  username: streamer\ntts:\n  speaker: old_voice\n  language: English\n"
            )

    def test_rejects_other_checkpoint_voice_language_or_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "tts.model"):
            self.load_yaml(
                "tiktok:\n  username: streamer\ntts:\n  model: another/model\n"
            )
        with self.assertRaisesRegex(ValueError, "tts.voice"):
            self.load_yaml(
                "tiktok:\n  username: streamer\ntts:\n  voice: Aiden\n"
            )
        with self.assertRaisesRegex(ValueError, "tts.language_code"):
            self.load_yaml(
                "tiktok:\n  username: streamer\ntts:\n  language_code: b\n"
            )
        with self.assertRaisesRegex(ValueError, "tts.speed"):
            self.load_yaml(
                "tiktok:\n  username: streamer\ntts:\n  speed: 0.4\n"
            )
