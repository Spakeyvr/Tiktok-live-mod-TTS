from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.llm_filter import (
    POLICY_PROMPT,
    Gemma4AudioFilter,
    parse_decision,
    resample_mono,
    sha256_file,
)


class DecisionParserTests(unittest.TestCase):
    def test_exact_block_is_the_only_unsafe_schema(self) -> None:
        decision = parse_decision('{"safe": false, "punishment": "block"}')
        self.assertFalse(decision.safe)
        self.assertEqual(decision.punishment, "block")

    def test_exact_safe_is_safe(self) -> None:
        self.assertTrue(parse_decision('{"safe": true}').safe)

    def test_old_mute_and_skip_labels_fail_open(self) -> None:
        self.assertTrue(parse_decision('{"safe": false, "punishment": "mute"}').safe)
        self.assertTrue(parse_decision('{"safe": false, "punishment": "skip"}').safe)

    def test_extra_keys_fail_open(self) -> None:
        self.assertTrue(parse_decision('{"safe": false, "punishment": "block", "score": 1}').safe)

    def test_malformed_output_fails_open(self) -> None:
        self.assertTrue(parse_decision("definitely unsafe").safe)
        self.assertTrue(parse_decision("{broken").safe)


class ResamplingTests(unittest.TestCase):
    def test_resamples_to_expected_length(self) -> None:
        source = np.linspace(-1, 1, 24_000, dtype=np.float32)
        output = resample_mono(source, 24_000, 16_000)
        self.assertEqual(output.shape, (16_000,))
        self.assertEqual(output.dtype, np.float32)

    def test_rejects_invalid_rate(self) -> None:
        with self.assertRaises(ValueError):
            resample_mono(np.ones(10), 0, 16_000)


class AdapterValidationTests(unittest.TestCase):
    def test_adapter_is_bound_to_model_and_weight_hash(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            adapter = Path(raw)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"adapter")
            (adapter / "training_manifest.json").write_text(
                json.dumps(
                    {
                        "architecture": "gemma-4-direct-audio-lora",
                        "model": "google/gemma-4-E4B-it",
                        "revision": "abc",
                        "adapter_sha256": sha256_file(weights),
                        "policy_sha256": hashlib.sha256(POLICY_PROMPT.encode("utf-8")).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            classifier = Gemma4AudioFilter(
                model="google/gemma-4-E4B-it", adapter_path=str(adapter)
            )
            classifier.validate_configuration()
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "provenance"):
                classifier.validate_configuration()
