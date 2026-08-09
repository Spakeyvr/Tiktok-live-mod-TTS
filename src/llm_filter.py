from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FilterDecision:
    safe: bool
    punishment: str | None
    reason: str = ""
    raw: str = ""


POLICY_PROMPT = """Listen to the following TTS audio and classify it under this exact policy.

Block only:
1. A clear claim, request, or offer to possess, access, obtain, trade, sell, send, or share child sexual abuse material (CSAM), or an unambiguous equivalent.
2. Clear racism: racial or ethnic slurs, racist abuse/dehumanization/exclusion, or a deliberate spoken near-slur meant to sound racist. Judge sound rather than spelling; for example, an intentional N-word-like pronunciation such as "Meggar" is racist evasion.

Everything else is safe, including profanity, insults, harassment, spam, threats, doxxing, self-harm language, illegal activity, adult sexual language, non-racial identity hostility, and borderline cases. Neutral reporting, condemnation, education, or discussion of CSAM or racism is safe unless the speaker actually makes a prohibited claim/request/offer or uses racist audio as abuse or bait.

Return only {"safe": true} or {"safe": false, "punishment": "block"}. No explanation."""


class AudioFilter(Protocol):
    async def check_audio(self, pcm: np.ndarray, sample_rate: int) -> FilterDecision: ...


_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


def parse_decision(raw: str) -> FilterDecision:
    """Accept only the exact block schema. Any uncertainty fails open."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        return FilterDecision(True, None, "unparseable response; allowed", raw)
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return FilterDecision(True, None, "invalid JSON; allowed", raw)
    if value == {"safe": False, "punishment": "block"}:
        return FilterDecision(False, "block", raw=raw)
    if value == {"safe": True}:
        return FilterDecision(True, None, raw=raw)
    return FilterDecision(True, None, "unexpected schema; allowed", raw)


def resample_mono(pcm: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    audio = np.asarray(pcm, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = audio.reshape(-1)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if audio.size == 0 or source_rate == target_rate:
        return audio
    target_size = max(1, round(audio.size * target_rate / source_rate))
    source_points = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_points = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(target_points, source_points, audio).astype(np.float32)


class Gemma4AudioFilter:
    """Local Gemma 4 E4B classifier that consumes the synthesized PCM directly."""

    def __init__(
        self,
        *,
        model: str = "google/gemma-4-E4B-it",
        adapter_path: str = "",
        revision: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 32,
    ) -> None:
        self.model_name = model
        self.adapter_path = Path(adapter_path).expanduser() if adapter_path else None
        self.revision = revision or None
        self.requested_device = device
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._load_error: Exception | None = None
        self._manifest: dict[str, Any] | None = None

    def validate_configuration(self) -> None:
        if self.adapter_path is None:
            raise ValueError("a trained Gemma 4 direct-audio adapter is required")
        adapter = self.adapter_path.resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"missing Gemma 4 adapter: {adapter}")
        manifest_path = adapter / "training_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing adapter provenance manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("architecture") != "gemma-4-direct-audio-lora":
            raise ValueError(f"adapter is not a Gemma 4 direct-audio adapter: {adapter}")
        if manifest.get("model") != self.model_name:
            raise ValueError(
                f"adapter base model is {manifest.get('model')!r}, not {self.model_name!r}"
            )
        adapter_weights = adapter / "adapter_model.safetensors"
        if not adapter_weights.is_file():
            raise FileNotFoundError(f"missing adapter weights: {adapter_weights}")
        if manifest.get("adapter_sha256") != sha256_file(adapter_weights):
            raise ValueError(f"adapter weights do not match their provenance manifest: {adapter}")
        policy_hash = hashlib.sha256(POLICY_PROMPT.encode("utf-8")).hexdigest()
        if manifest.get("policy_sha256") != policy_hash:
            raise ValueError(f"runtime policy does not match the adapter's training policy: {adapter}")
        trained_revision = manifest.get("revision")
        if self.revision and trained_revision and self.revision != trained_revision:
            raise ValueError(
                f"adapter base revision is {trained_revision!r}, not {self.revision!r}"
            )
        self._manifest = manifest

    async def load(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        async with self._load_lock:
            if self._model is None:
                try:
                    await asyncio.to_thread(self._load_blocking)
                except Exception as exc:
                    self._load_error = exc
                    raise

    def _load_blocking(self) -> None:
        self.validate_configuration()
        assert self._manifest is not None
        effective_revision = self.revision or self._manifest.get("revision")
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("install the Gemma 4 runtime requirements") from exc

        if self.requested_device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.requested_device
        if device == "mps":
            dtype = torch.bfloat16 if torch.backends.mps.is_macos_or_newer(14, 0) else torch.float16
        elif device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        processor = AutoProcessor.from_pretrained(self.model_name, revision=effective_revision)
        model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            revision=effective_revision,
            dtype=dtype,
            attn_implementation="eager" if device == "mps" else None,
        )
        if self.adapter_path:
            adapter = self.adapter_path.resolve()
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.to(device).eval()
        self._torch = torch
        self._processor = processor
        self._model = model
        self._device = device
        log.info("loaded %s on %s", self.model_name, device)

    async def check_audio(self, pcm: np.ndarray, sample_rate: int) -> FilterDecision:
        if np.asarray(pcm).size == 0:
            return FilterDecision(True, None, "empty audio")
        try:
            await self.load()
            async with self._inference_lock:
                return await asyncio.to_thread(self._check_blocking, pcm, sample_rate)
        except Exception as exc:
            log.warning("Gemma 4 audio classification failed (%s); allowing audio.", exc)
            return FilterDecision(True, None, f"model_error: {exc}")

    def _check_blocking(self, pcm: np.ndarray, sample_rate: int) -> FilterDecision:
        processor = self._processor
        torch = self._torch
        target_rate = int(processor.feature_extractor.sampling_rate)
        audio = resample_mono(pcm, sample_rate, target_rate)
        duration = audio.size / target_rate
        if duration > 30.0:
            raise ValueError(f"audio exceeds Gemma 4's 30-second limit ({duration:.2f}s)")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": POLICY_PROMPT},
                    {"type": "audio"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=prompt,
            audio=audio,
            sampling_rate=target_rate,
            return_tensors="pt",
            return_mm_token_type_ids=True,
        ).to(self._device)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        raw = processor.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
        return parse_decision(raw)
