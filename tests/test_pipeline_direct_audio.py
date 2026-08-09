from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.bridge import ChatMessage
from src.llm_filter import FilterDecision
from src.pipeline import Pipeline


class FakeNormalizer:
    def normalize(self, text: str) -> str:
        return text


class FakeTTS:
    sample_rate = 24_000

    def __init__(self) -> None:
        self.pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    async def synthesize(self, text: str):
        return self.pcm, self.sample_rate


class FakeFilter:
    def __init__(self, decision: FilterDecision) -> None:
        self.decision = decision
        self.received = None

    async def check_audio(self, pcm, sample_rate):
        self.received = (pcm, sample_rate)
        return self.decision


class Recorder:
    def __init__(self) -> None:
        self.calls = []

    async def play(self, *args):
        self.calls.append(args)

    async def punish(self, *args):
        self.calls.append(args)
        return True

    async def log(self, **kwargs):
        self.calls.append(kwargs)


class DirectAudioPipelineTests(unittest.IsolatedAsyncioTestCase):
    def build(self, decision: FilterDecision):
        tts = FakeTTS()
        classifier = FakeFilter(decision)
        player = Recorder()
        moderator = Recorder()
        event_log = Recorder()
        pipeline = Pipeline(
            bridge=SimpleNamespace(),
            normalizer=FakeNormalizer(),
            tts=tts,
            llm=classifier,
            moderator=moderator,
            player=player,
            event_log=event_log,
        )
        return pipeline, tts, classifier, player, moderator, event_log

    async def test_safe_audio_is_classified_and_same_pcm_is_played(self) -> None:
        pipeline, tts, classifier, player, moderator, _ = self.build(FilterDecision(True, None))
        await pipeline._process(ChatMessage("1", "u", "name", "n", "hello", 0))
        self.assertIs(classifier.received[0], tts.pcm)
        self.assertEqual(classifier.received[1], 24_000)
        self.assertIs(player.calls[0][0], tts.pcm)
        self.assertEqual(moderator.calls, [])

    async def test_unsafe_audio_can_only_block(self) -> None:
        pipeline, _, _, player, moderator, event_log = self.build(FilterDecision(False, "block"))
        await pipeline._process(ChatMessage("1", "u", "name", "n", "bad", 0))
        self.assertEqual(player.calls, [])
        self.assertEqual(moderator.calls, [("u", "block")])
        self.assertEqual(event_log.calls[0]["safe"], False)
        self.assertEqual(event_log.calls[0]["punishment"], "block")
