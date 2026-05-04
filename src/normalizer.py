from __future__ import annotations

import re
import unicodedata
from typing import Protocol

try:
    from confusables import normalize as confusables_normalize  # type: ignore
except ImportError:  # pragma: no cover
    confusables_normalize = None


class Normalizer(Protocol):
    def normalize(self, text: str) -> str: ...


_ZERO_WIDTH = re.compile(
    "[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]"
)
_WHITESPACE = re.compile(r"\s+")

# Common leet-speak digit/symbol → letter substitutions applied per character.
_LEET: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
    "!": "i",
    "+": "t",
}
# Only de-leet tokens that are *purely* leet (no real word is all-digits).
# A token qualifies if every character is either a letter or a leet digit.
_LEET_TOKEN = re.compile(r"\b([a-zA-Z0-9@$!+]+)\b")


def _deleet(token: str) -> str:
    """Replace leet digits with letters only in mixed letter+digit tokens.

    A pure number like '100' is left alone; 'n1gg3r' (has real letters) is
    decoded because the substitution is meaningful in context.
    """
    if not any(c.isalpha() for c in token):
        return token  # pure digits / symbols — leave unchanged
    if not any(c in _LEET for c in token):
        return token
    return "".join(_LEET.get(c, c) for c in token)


class HomoglyphNormalizer:
    """NFKC + zero-width strip + confusables homoglyph collapse + leet de-coding.

    Pass order:
      1. NFKC unicode normalisation
      2. zero-width / invisible character removal
      3. confusables homoglyph collapse (Cyrillic/Greek look-alikes → Latin)
      4. leet-speak digit substitution (0→o, 1→i, 3→e, …)
    """

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = _ZERO_WIDTH.sub("", text)
        if confusables_normalize is not None:
            candidates = confusables_normalize(text)
            if candidates:
                text = candidates[0]
        text = _LEET_TOKEN.sub(lambda m: _deleet(m.group(1)), text)
        text = _WHITESPACE.sub(" ", text).strip()
        return text
