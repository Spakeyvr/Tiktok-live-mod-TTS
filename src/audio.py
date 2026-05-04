from __future__ import annotations

import asyncio
import logging

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioPlayer:
    """Serialized blocking playback via sounddevice on a worker thread.

    Using a single asyncio lock ensures TTS clips don't overlap when
    several safe messages arrive back-to-back.
    """

    def __init__(self, output_device: int | None = None) -> None:
        self.output_device = output_device
        self._lock = asyncio.Lock()

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        if pcm.size == 0:
            return
        async with self._lock:
            await asyncio.to_thread(self._play_blocking, pcm, sample_rate)

    def _play_blocking(self, pcm: np.ndarray, sample_rate: int) -> None:
        try:
            sd.play(pcm, samplerate=sample_rate, device=self.output_device)
            sd.wait()
        except Exception as e:
            log.warning("audio playback failed: %s", e)
