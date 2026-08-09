from __future__ import annotations

import asyncio
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from src.tts import KOKORO_TTS_MODEL, KokoroTTSEngine


class FakeKokoroModel:
    def __init__(self, results=None, delay: float = 0.0) -> None:
        self.results = results or [
            SimpleNamespace(
                audio=np.array([0.1, -0.2], dtype=np.float32),
                sample_rate=24_000,
            )
        ]
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def generate(self, **kwargs):
        def results():
            with self._lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                if self.delay:
                    time.sleep(self.delay)
                self.calls.append(kwargs)
                yield from self.results
            finally:
                with self._lock:
                    self.active -= 1

        return results()


class KokoroTTSEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_once_and_uses_fixed_generation_contract(self) -> None:
        model = FakeKokoroModel()
        loaded = []

        def loader(model_name):
            loaded.append(model_name)
            return model

        engine = KokoroTTSEngine(model_loader=loader)
        await engine.load()
        pcm, sample_rate = await engine.synthesize("hello")

        self.assertEqual(loaded, [KOKORO_TTS_MODEL])
        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(pcm.dtype, np.float32)
        self.assertTrue(pcm.flags.c_contiguous)
        self.assertEqual(
            model.calls,
            [
                {
                    "text": "hello",
                    "voice": "af_heart",
                    "speed": 1.0,
                    "lang_code": "a",
                }
            ],
        )

    async def test_concatenates_segments(self) -> None:
        model = FakeKokoroModel(
            [
                SimpleNamespace(audio=np.array([0.1, 0.2]), sample_rate=24_000),
                SimpleNamespace(audio=np.array([0.3]), sample_rate=24_000),
            ]
        )
        engine = KokoroTTSEngine(model_loader=lambda _: model)
        pcm, _ = await engine.synthesize("hello")
        np.testing.assert_allclose(pcm, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    async def test_serializes_concurrent_generation(self) -> None:
        model = FakeKokoroModel(delay=0.03)
        engine = KokoroTTSEngine(model_loader=lambda _: model)
        await asyncio.gather(engine.synthesize("one"), engine.synthesize("two"))
        self.assertEqual(model.max_active, 1)

    async def test_empty_text_does_not_load_model(self) -> None:
        loaded = []
        engine = KokoroTTSEngine(model_loader=lambda name: loaded.append(name))
        pcm, sample_rate = await engine.synthesize("  ")
        self.assertEqual(loaded, [])
        self.assertEqual(pcm.size, 0)
        self.assertEqual(sample_rate, 24_000)

    async def test_rejects_nonfinite_audio(self) -> None:
        model = FakeKokoroModel(
            [SimpleNamespace(audio=np.array([np.nan]), sample_rate=24_000)]
        )
        engine = KokoroTTSEngine(model_loader=lambda _: model)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            await engine.synthesize("hello")

    async def test_rejects_audio_beyond_moderation_limit(self) -> None:
        model = FakeKokoroModel(
            [SimpleNamespace(audio=np.ones(25, dtype=np.float32), sample_rate=10)]
        )
        engine = KokoroTTSEngine(
            max_audio_seconds=2.0,
            model_loader=lambda _: model,
        )
        with self.assertRaisesRegex(ValueError, "moderation limit"):
            await engine.synthesize("too long")

    def test_rejects_other_models_voices_languages_and_speeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the Kokoro"):
            KokoroTTSEngine(model="some/other-model")
        with self.assertRaisesRegex(ValueError, "unsupported American English"):
            KokoroTTSEngine(voice="Aiden")
        with self.assertRaisesRegex(ValueError, "language_code='a'.*only"):
            KokoroTTSEngine(language_code="b")
        with self.assertRaisesRegex(ValueError, "speed"):
            KokoroTTSEngine(speed=2.1)
