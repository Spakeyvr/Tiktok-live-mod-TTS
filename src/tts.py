from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class TTSEngine(Protocol):
    sample_rate: int

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (mono float32 PCM in [-1, 1], sample_rate)."""
        ...

    async def close(self) -> None: ...


class RealtimeTTSEngine:
    """RealtimeTTS wrapper that synthesizes to a buffer (no playback).

    We use the engine's `play(muted=True, output_wavfile=...)` path: it runs
    the full synthesis pipeline silently and writes a wav we read back. This
    keeps every backend (Kokoro, Piper) on the same code path without us
    needing to know engine-internal stream APIs.
    """

    def __init__(
        self,
        engine_name: str = "kokoro",
        voice: str = "af_sarah",
        piper_model: str = "",
        sample_rate: int = 24000,
    ) -> None:
        self.engine_name = engine_name.lower()
        self.voice = voice
        self.piper_model = piper_model
        self.sample_rate = sample_rate
        self._engine = None
        self._stream = None
        self._lock = asyncio.Lock()

    def _build_engine(self):
        if self.engine_name == "kokoro":
            from RealtimeTTS import KokoroEngine  # type: ignore

            return KokoroEngine(voice=self.voice)
        if self.engine_name == "piper":
            from RealtimeTTS import PiperEngine, PiperVoice  # type: ignore

            if not self.piper_model:
                raise ValueError("piper_model path required for piper engine")
            return PiperEngine(piper_path="piper", voice=PiperVoice(self.piper_model))
        raise ValueError(f"unknown TTS engine: {self.engine_name}")

    def _ensure_stream(self):
        if self._stream is not None:
            return
        from RealtimeTTS import TextToAudioStream  # type: ignore

        self._engine = self._build_engine()
        self._stream = TextToAudioStream(self._engine, muted=True, level=logging.WARNING)

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, int]:
        self._ensure_stream()
        assert self._stream is not None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            wav_path = Path(tf.name)
        try:
            self._stream.feed(text)
            self._stream.play(
                muted=True,
                output_wavfile=str(wav_path),
                log_synthesized_text=False,
            )
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return data, int(sr)
        finally:
            try:
                wav_path.unlink()
            except OSError:
                pass

    async def close(self) -> None:
        if self._stream is not None:
            try:
                await asyncio.to_thread(self._stream.stop)
            except Exception:
                pass
        self._stream = None
        self._engine = None


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Helper for callers (e.g., STT) that want a wav blob."""
    buf = io.BytesIO()
    pcm16 = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()
