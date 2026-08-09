from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

KOKORO_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
KOKORO_TTS_SAMPLE_RATE = 24_000
KOKORO_TTS_LANGUAGE_CODE = "a"
KOKORO_TTS_VOICES = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    }
)


class TTSEngine(Protocol):
    sample_rate: int

    async def load(self) -> None: ...

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (mono float32 PCM in [-1, 1], sample_rate)."""
        ...

    async def close(self) -> None: ...


def _load_kokoro_model(model_name: str) -> Any:
    try:
        from mlx_audio.tts.utils import load_model

        # MLX-Audio imports these lazily on the first generation. Import them
        # here so a broken installation fails during startup with a useful
        # message instead of on the first live comment.
        importlib.import_module("misaki.en")
        importlib.import_module("misaki.espeak")
    except ImportError as exc:  # pragma: no cover - exercised by the real smoke test
        raise RuntimeError(
            "Kokoro requires MLX-Audio and misaki[en]; install requirements.txt "
            "on Apple Silicon"
        ) from exc
    return load_model(model_name)


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx
        mx.clear_cache()
    except (ImportError, RuntimeError) as exc:
        # Cleanup must never mask the original startup/inference failure. A
        # sandboxed or headless process can import MLX without having Metal.
        log.debug("could not clear MLX cache during shutdown: %s", exc)
        return


def _audio_to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return np.array(audio, dtype=np.float32, copy=True)

    # MLX evaluates lazily. Materialize the result before it crosses back from
    # the worker thread, then take an owned NumPy copy.
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - only possible with a broken install
        raise RuntimeError("generated MLX audio cannot be materialized") from exc
    mx.eval(audio)
    return np.array(audio, dtype=np.float32, copy=True)


def _mono_audio(audio: Any) -> np.ndarray:
    pcm = np.squeeze(_audio_to_numpy(audio))
    if pcm.ndim == 0:
        pcm = pcm.reshape(1)
    elif pcm.ndim == 2 and 1 in pcm.shape:
        pcm = pcm.reshape(-1)
    if pcm.ndim != 1:
        raise ValueError(f"Kokoro returned non-mono audio with shape {pcm.shape}")
    if not np.isfinite(pcm).all():
        raise ValueError("Kokoro returned non-finite audio samples")
    return np.clip(pcm, -1.0, 1.0).astype(np.float32, copy=False)


class KokoroTTSEngine:
    """Kokoro 82M BF16 engine for American English on Apple Silicon."""

    sample_rate = KOKORO_TTS_SAMPLE_RATE

    def __init__(
        self,
        *,
        model: str = KOKORO_TTS_MODEL,
        voice: str = "af_heart",
        language_code: str = KOKORO_TTS_LANGUAGE_CODE,
        speed: float = 1.0,
        max_audio_seconds: float = 30.0,
        model_loader: Callable[[str], Any] = _load_kokoro_model,
    ) -> None:
        if model != KOKORO_TTS_MODEL:
            raise ValueError(
                f"only the Kokoro 82M BF16 model is supported: {KOKORO_TTS_MODEL}"
            )
        canonical_voices = {name.casefold(): name for name in KOKORO_TTS_VOICES}
        try:
            voice = canonical_voices[voice.casefold()]
        except KeyError as exc:
            allowed = ", ".join(sorted(KOKORO_TTS_VOICES))
            raise ValueError(
                f"unsupported American English Kokoro voice {voice!r}; use {allowed}"
            ) from exc
        if language_code.casefold() != KOKORO_TTS_LANGUAGE_CODE:
            raise ValueError("this deployment supports Kokoro language_code='a' only")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("Kokoro speed must be between 0.5 and 2.0")
        if max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be positive")

        self.model_name = model
        self.voice = voice
        self.language_code = KOKORO_TTS_LANGUAGE_CODE
        self.speed = float(speed)
        self.max_audio_seconds = float(max_audio_seconds)
        self._model_loader = model_loader
        self._model: Any = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        log.info("loading Kokoro model=%s", self.model_name)
        self._model = self._model_loader(self.model_name)
        log.info(
            "loaded Kokoro model=%s voice=%s language_code=%s speed=%g",
            self.model_name,
            self.voice,
            self.language_code,
            self.speed,
        )

    async def warmup(self) -> None:
        pcm, sample_rate = await self.synthesize("System ready.")
        log.info(
            "Kokoro warmup complete: %d samples @ %d Hz",
            pcm.size,
            sample_rate,
        )

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if not text.strip():
            return np.zeros(0, dtype=np.float32), self.sample_rate
        await self.load()
        async with self._inference_lock:
            return await asyncio.to_thread(self._synthesize_blocking, text)

    def _synthesize_blocking(self, text: str) -> tuple[np.ndarray, int]:
        if self._model is None:
            raise RuntimeError("Kokoro is not loaded")

        generated: Iterable[Any] = self._model.generate(
            text=text,
            voice=self.voice,
            speed=self.speed,
            lang_code=self.language_code,
        )
        chunks: list[np.ndarray] = []
        sample_rates: set[int] = set()
        for result in generated:
            audio = getattr(result, "audio", None)
            if audio is None:
                raise ValueError("Kokoro returned a result without audio")
            chunk = _mono_audio(audio)
            if chunk.size:
                chunks.append(chunk)
            result_rate = getattr(result, "sample_rate", None)
            if result_rate is not None:
                sample_rates.add(int(result_rate))

        if not chunks:
            raise ValueError("Kokoro produced empty audio")
        if len(sample_rates) > 1:
            raise ValueError(f"Kokoro returned inconsistent sample rates: {sample_rates}")

        sample_rate = sample_rates.pop() if sample_rates else self.sample_rate
        if sample_rate <= 0:
            raise ValueError(f"Kokoro returned invalid sample rate {sample_rate}")
        pcm = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
        duration = pcm.size / sample_rate
        if duration > self.max_audio_seconds:
            raise ValueError(
                f"Kokoro audio exceeds the {self.max_audio_seconds:g}-second "
                f"moderation limit ({duration:.2f}s)"
            )
        return pcm, sample_rate

    async def close(self) -> None:
        async with self._inference_lock:
            self._model = None
            await asyncio.to_thread(_clear_mlx_cache)
