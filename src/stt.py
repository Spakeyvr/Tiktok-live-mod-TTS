from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class STTEngine(Protocol):
    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str: ...


class FasterWhisperSTT:
    """Distil-Whisper via faster-whisper.

    Loads lazily on first transcribe so the rest of the pipeline can boot
    while the model downloads.
    """

    def __init__(
        self,
        model: str = "Systran/faster-distil-whisper-small.en",
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):
        from faster_whisper import WhisperModel  # type: ignore

        device = self.device
        compute_type = self.compute_type

        if device == "auto":
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # ctranslate2 on CPU doesn't support int8_float16; fall back to int8.
        if device == "cpu" and compute_type == "int8_float16":
            compute_type = "int8"

        log.info(
            "loading faster-whisper model=%s device=%s compute=%s",
            self.model_name,
            device,
            compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_name, device=device, compute_type=compute_type
            )
        except RuntimeError as e:
            if device != "cpu":
                log.warning("CUDA load failed (%s); falling back to CPU", e)
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
            else:
                raise

    async def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        if pcm.size == 0:
            return ""
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_blocking, pcm, sample_rate)

    def _transcribe_blocking(self, pcm: np.ndarray, sample_rate: int) -> str:
        if self._model is None:
            self._load()
        assert self._model is not None
        if sample_rate != 16000:
            # faster-whisper requires 16k mono float32; resample with simple
            # linear interpolation to avoid pulling in scipy/librosa.
            pcm = _resample_linear(pcm, sample_rate, 16000)
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
        segments, _info = self._model.transcribe(
            pcm,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    duration = x.shape[0] / src_sr
    new_len = int(round(duration * dst_sr))
    if new_len <= 1:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, x.shape[0] - 1, new_len, dtype=np.float64)
    lo = np.floor(src_idx).astype(np.int64)
    hi = np.minimum(lo + 1, x.shape[0] - 1)
    frac = (src_idx - lo).astype(np.float32)
    return (x[lo] * (1.0 - frac) + x[hi] * frac).astype(np.float32)
