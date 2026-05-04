from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol

from openai import AsyncOpenAI

log = logging.getLogger(__name__)


Punishment = Literal["mute", "block", "skip"]


@dataclass
class FilterDecision:
    safe: bool
    punishment: Punishment | None  # None when safe
    reason: str = ""             # local diagnostic; not sent to TikTok
    raw: str = ""                # raw model output


SYSTEM_PROMPT = """You are a strict real-time content safety filter for a live-streaming chat-to-TTS pipeline.

You receive a transcript of what a TTS engine *actually said out loud* after reading a viewer's chat message. The viewer may have used homoglyphs, leetspeak, spacing tricks, or phonetic spellings to bypass filters; the TTS+STT round-trip exposes the spoken form.

Decide whether the spoken transcript contains any of:
- slurs targeting protected characteristics (race, ethnicity, religion, gender, sexuality, disability)
- hate speech, dehumanization, or incitement
- explicit attempts to bypass moderation (e.g. coded slurs)
- explicit threats of doxxing
- similar words to slurs which carry over a bad meaning
- content that resembles the above in a way that suggests the user is trying to be harmful, even if the exact words aren't on a blocklist
- spam meant to be annoying or disruptive (e.g. repeated characters, long strings of nonsense, etc.)

Severity — choose the lowest level that fits:
- "skip"  — drop the TTS audio silently, no action taken against the user. Use for borderline/crude content that shouldn't be read aloud but doesn't warrant punishment (e.g. spam)
- "mute"  — silence the user for 5 minutes. Use for clear slurs, harassment, hate speech.
- "block" — permanent mute. Use only for explicit threats of violence, doxxing, or severe escalation.

Do not punish:
- non-harmful content (e.g. casual profanity, inoffensive reclaimed slurs, non-hateful uses of hateful words)
- others critizing or making fun of hateful content (e.g. "I hate [slur]", "X is a [slur]", "so many people saying slurs")

To decide punishment:
- "safe": true means the message is clean and gets sent.
- "safe": false means the message is not clean and gets declined.
- "punishment": "skip" means the message is not clean but not too severe, so it gets declined but no punishment.
- "punishment": "mute" means the user should be muted for 5 minutes.
- "punishment": "block" means the user should be permanently blocked.

Output STRICTLY a single JSON object on one line, no prose, no markdown, no code fences:
{"safe": true} when clean.
{"safe": false, "punishment": "mute"} or {"safe": false, "punishment": "block"} or {"safe": false, "punishment": "skip"} when not.

Do not include any other keys. Do not explain. Do not apologize. JSON only. /no_think"""


class LLMFilter(Protocol):
    async def check(self, transcript: str) -> FilterDecision: ...


class OpenAICompatibleFilter:
    """Works with any OpenAI-compatible endpoint (Ollama, LM Studio, oMLX, ...).

    For Qwen3.5 on Ollama, pass think=False so the model skips its reasoning
    phase entirely. The parameter goes as a top-level body key via extra_body;
    it is silently ignored by backends that don't understand it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        request_timeout: int = 30,
        local_blocklist: Iterable[str] = (),
        think: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.think = think
        self.client = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "not-needed", timeout=request_timeout
        )
        self.local_blocklist = [b.lower() for b in local_blocklist if b]

    async def check(self, transcript: str) -> FilterDecision:
        if not transcript.strip():
            return FilterDecision(safe=True, punishment=None, reason="empty transcript")

        lowered = transcript.lower()
        for term in self.local_blocklist:
            if term in lowered:
                return FilterDecision(
                    safe=False,
                    punishment="mute",
                    reason=f"local blocklist match: {term!r}",
                )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=50,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Transcript: {transcript}"},
                    {"role": "assistant", "content": "{"},
                ],
                extra_body={"think": self.think},
            )
        except Exception as e:
            log.warning("LLM call failed (%s); failing closed with mute.", e)
            return FilterDecision(
                safe=False, punishment="mute", reason=f"llm_error: {e}"
            )

        raw = _strip_think_tags(resp.choices[0].message.content or "").strip()
        # Restore the prefilled "{" if the model didn't repeat it.
        if raw and not raw.startswith("{"):
            raw = "{" + raw
        decision = _parse_decision(raw)
        decision.raw = raw
        return decision


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks that Qwen3.5 may emit even with think=false.

    Also strips an unterminated <think> tag to end-of-string, which happens
    when max_tokens cuts off the model mid-reasoning.
    """
    text = _THINK_RE.sub("", text)
    return _THINK_OPEN_RE.sub("", text)


def _parse_decision(raw: str) -> FilterDecision:
    if not raw:
        return FilterDecision(
            safe=False, punishment="mute", reason="empty llm response"
        )
    text = raw.strip()
    # Tolerate fenced output even though we asked for none.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            return FilterDecision(
                safe=False, punishment="mute", reason=f"unparseable: {raw!r}"
            )
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return FilterDecision(
                safe=False, punishment="mute", reason=f"unparseable: {raw!r}"
            )

    safe = bool(obj.get("safe"))
    if safe:
        return FilterDecision(safe=True, punishment=None)
    punishment = obj.get("punishment")
    if punishment not in ("mute", "block", "skip"):
        punishment = "mute"
    return FilterDecision(safe=False, punishment=punishment)
