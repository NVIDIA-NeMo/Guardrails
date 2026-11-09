# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the pure functions in content_safety_parity.py.

These tests do not call any live LLM or provider services.
"""

import base64
import os
import tempfile

import pytest

from benchmark.content_safety_parity.content_safety_parity import (
    Sample,
    Verdict,
    _build_content,
    _extract_rail_outcome,
    _stratified_sample,
    compute_metrics,
    parse_verdict,
)


class TestParseVerdict:
    def test_safe_verdict(self):
        raw = "User Safety: safe\nSafety Categories: None"
        v = parse_verdict(raw, latency_ms=42.0)
        assert v.is_safe is True
        assert v.error is False
        assert v.latency_ms == 42.0

    def test_unsafe_verdict_with_categories(self):
        raw = "User Safety: unsafe\nSafety Categories: Hate/Identity Hate, Threat"
        v = parse_verdict(raw)
        assert v.is_safe is False
        assert v.error is False
        assert "Hate/Identity Hate" in v.categories
        assert "Threat" in v.categories

    def test_missing_verdict_line_is_error(self):
        raw = "Some truncated output without the verdict line"
        v = parse_verdict(raw)
        assert v.error is True

    def test_empty_response_is_error(self):
        v = parse_verdict("")
        assert v.error is True

    def test_categories_split_correctly(self):
        # Category names may contain slashes — split on comma only, not slash
        raw = "User Safety: unsafe\nSafety Categories: Criminal Planning/Confessions, PII/Privacy"
        v = parse_verdict(raw)
        assert len(v.categories) == 2
        assert "Criminal Planning/Confessions" in v.categories
        assert "PII/Privacy" in v.categories

    def test_safe_with_no_categories(self):
        raw = "User Safety: safe"
        v = parse_verdict(raw)
        assert v.is_safe is True
        assert v.categories == []
        assert v.error is False

    def test_raw_stored_verbatim(self):
        raw = "User Safety: safe\nSafety Categories: None"
        v = parse_verdict(raw)
        assert v.raw == raw

    def test_truncated_verdict_line_is_error(self):
        # A response truncated after "User Safety:" with no value must be an error,
        # not an unsafe verdict.
        raw = "User Safety:"
        v = parse_verdict(raw)
        assert v.error is True
        assert v.is_safe is False

    def test_safe_verdict_capital_s(self):
        # NIM sometimes returns "Safe" with a capital S — the production parser
        # handles this case and must still return is_safe=True.
        raw = "User Safety: Safe\nSafety Categories: None"
        v = parse_verdict(raw, latency_ms=10.0)
        assert v.is_safe is True
        assert v.error is False

    def test_unsafe_verdict_capital_u(self):
        raw = "User Safety: Unsafe\nSafety Categories: Violence"
        v = parse_verdict(raw)
        assert v.is_safe is False
        assert v.error is False
        assert "Violence" in v.categories


class TestComputeMetrics:
    def _make_pair(self, gt: str, pred_safe: bool) -> tuple:
        return Sample(text="x", ground_truth=gt), Verdict(is_safe=pred_safe, raw="")

    def test_perfect_classifier(self):
        pairs = [
            self._make_pair("unsafe", False),
            self._make_pair("unsafe", False),
            self._make_pair("safe", True),
        ]
        samples, verdicts = zip(*pairs)
        m = compute_metrics(list(samples), list(verdicts))
        assert m["tp"] == 2
        assert m["tn"] == 1
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["f1"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_all_false_negatives(self):
        pairs = [self._make_pair("unsafe", True) for _ in range(3)]
        samples, verdicts = zip(*pairs)
        m = compute_metrics(list(samples), list(verdicts))
        assert m["fn"] == 3
        assert m["tp"] == 0
        assert m["f1"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_errored_verdicts_excluded(self):
        s = Sample(text="x", ground_truth="unsafe")
        v_ok = Verdict(is_safe=False, raw="", error=False)
        v_err = Verdict(is_safe=True, raw="ERROR", error=True)
        m = compute_metrics([s, s], [v_ok, v_err])
        assert m["errors"] == 1
        assert m["total"] == 1
        assert m["tp"] == 1

    def test_empty_input(self):
        m = compute_metrics([], [])
        assert m["total"] == 0
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)

    def test_precision_recall_f1_balanced(self):
        # 2 TP, 1 FP, 1 FN → precision=2/3, recall=2/3, F1=2/3
        pairs = [
            self._make_pair("unsafe", False),
            self._make_pair("unsafe", False),
            self._make_pair("safe", False),  # FP
            self._make_pair("unsafe", True),  # FN
        ]
        samples, verdicts = zip(*pairs)
        m = compute_metrics(list(samples), list(verdicts))
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3)


class TestStratifiedSample:
    def _samples(self, n_safe: int, n_unsafe: int) -> list:
        safe = [Sample(text=f"s{i}", ground_truth="safe") for i in range(n_safe)]
        unsafe = [Sample(text=f"u{i}", ground_truth="unsafe") for i in range(n_unsafe)]
        return safe + unsafe

    def test_no_sampling_when_n_exceeds_total(self):
        samples = self._samples(5, 5)
        result = _stratified_sample(samples, n=100)
        assert len(result) == 10

    def test_no_sampling_when_n_zero(self):
        samples = self._samples(5, 5)
        result = _stratified_sample(samples, n=0)
        assert len(result) == 10

    def test_balanced_output(self):
        samples = self._samples(100, 100)
        result = _stratified_sample(samples, n=40)
        safe_n = sum(1 for s in result if s.ground_truth == "safe")
        unsafe_n = sum(1 for s in result if s.ground_truth == "unsafe")
        assert safe_n == 20
        assert unsafe_n == 20

    def test_reproducible_with_seed(self):
        samples = self._samples(50, 50)
        r1 = _stratified_sample(samples, n=20, seed=7)
        r2 = _stratified_sample(samples, n=20, seed=7)
        assert [s.text for s in r1] == [s.text for s in r2]

    def test_different_seeds_differ(self):
        samples = self._samples(50, 50)
        r1 = _stratified_sample(samples, n=20, seed=1)
        r2 = _stratified_sample(samples, n=20, seed=2)
        # Very unlikely to be identical with different seeds
        assert [s.text for s in r1] != [s.text for s in r2]

    def test_skewed_dataset_capped_at_available(self):
        # 3 safe, 100 unsafe → request 20; safe capped at 3, unsafe fills remaining 17
        samples = self._samples(3, 100)
        result = _stratified_sample(samples, n=20)
        safe_n = sum(1 for s in result if s.ground_truth == "safe")
        unsafe_n = sum(1 for s in result if s.ground_truth == "unsafe")
        assert safe_n == 3
        assert unsafe_n == 17

    def test_skewed_dataset_reverse_capped_at_available(self):
        # 100 safe, 3 unsafe → request 20; unsafe capped at 3, safe fills remaining 17
        samples = self._samples(100, 3)
        result = _stratified_sample(samples, n=20)
        safe_n = sum(1 for s in result if s.ground_truth == "safe")
        unsafe_n = sum(1 for s in result if s.ground_truth == "unsafe")
        assert safe_n == 17
        assert unsafe_n == 3
        assert len(result) == 20


class TestExtractRailOutcome:
    """Tests for _extract_rail_outcome — walks the generate_async log structure."""

    def _make_result(self, return_value=None, n_rails: int = 1):
        """Build a minimal fake GenerationResponse-like object."""

        class FakeAction:
            def __init__(self, rv):
                self.return_value = rv

        class FakeRail:
            def __init__(self, rv):
                self.executed_actions = [FakeAction(rv)]

        class FakeLog:
            def __init__(self, rv):
                self.activated_rails = [FakeRail(rv) for _ in range(n_rails)]

        class FakeResult:
            def __init__(self, rv):
                self.log = FakeLog(rv)

        return FakeResult(return_value)

    def test_returns_none_when_no_log(self):
        class NoLog:
            pass

        assert _extract_rail_outcome(NoLog()) is None

    def test_returns_none_when_activated_rails_empty(self):
        class EmptyLog:
            activated_rails = []

        class Result:
            log = EmptyLog()

        assert _extract_rail_outcome(Result()) is None

    def test_returns_none_when_no_rail_outcome_in_actions(self):
        # return_value is a plain string, not a RailOutcome
        result = self._make_result(return_value="not-a-rail-outcome")
        assert _extract_rail_outcome(result) is None

    def test_returns_rail_outcome_when_present(self):
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(return_value=outcome)
        found = _extract_rail_outcome(result)
        assert found is outcome

    def test_returns_first_rail_outcome_when_multiple_rails(self):
        from nemoguardrails.actions.rail_outcome import RailOutcome

        # First rail has a RailOutcome; second also does — should return the first.
        class FakeAction:
            def __init__(self, rv):
                self.return_value = rv

        class FakeRail:
            def __init__(self, rv):
                self.executed_actions = [FakeAction(rv)]

        class FakeLog:
            def __init__(self):
                o1 = RailOutcome.block(metadata={"policy_violations": ["Violence"]})
                o2 = RailOutcome.allow(metadata={"policy_violations": []})
                self.activated_rails = [FakeRail(o1), FakeRail(o2)]

        class Result:
            log = FakeLog()

        found = _extract_rail_outcome(Result())
        assert found is not None
        assert found.is_blocked is True


class TestParseVerdictProductionParser:
    """Additional parse_verdict tests that exercise production-parser-specific behaviour."""

    def test_duplicate_verdict_is_error(self):
        # Two "User Safety:" lines — the production parser raises ValueError.
        raw = "User Safety: safe\nUser Safety: unsafe\nSafety Categories: None"
        v = parse_verdict(raw)
        assert v.error is True

    def test_safe_no_categories_field(self):
        # Safe verdict with no "Safety Categories:" line is valid.
        raw = "User Safety: safe"
        v = parse_verdict(raw)
        assert v.is_safe is True
        assert v.error is False
        assert v.categories == []

    def test_unsafe_multiple_categories(self):
        raw = "User Safety: unsafe\nSafety Categories: Violence, Hate/Identity Hate"
        v = parse_verdict(raw)
        assert v.is_safe is False
        assert v.error is False
        assert "Violence" in v.categories
        assert "Hate/Identity Hate" in v.categories


class TestBuildContent:
    def test_text_only(self):
        result = _build_content("hello", None)
        assert result == "hello"

    def test_image_produces_list(self):
        data = b"\xff\xd8\xff"  # minimal JPEG header bytes
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            result = _build_content("describe this", path)
            assert isinstance(result, list)
            assert len(result) == 2
            img_block = result[0]
            assert img_block["type"] == "image_url"
            b64 = base64.b64encode(data).decode()
            assert img_block["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"
            text_block = result[1]
            assert text_block["type"] == "text"
            assert text_block["text"] == "describe this"
        finally:
            os.unlink(path)

    def test_png_mime_type(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG")
            path = f.name
        try:
            result = _build_content("x", path)
            assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(path)

    def test_unknown_extension_defaults_to_png_mime(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"BM")
            path = f.name
        try:
            result = _build_content("x", path)
            assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(path)
