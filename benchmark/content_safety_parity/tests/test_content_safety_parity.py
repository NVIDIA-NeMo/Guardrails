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

import asyncio
import base64
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmark.content_safety_parity.content_safety_parity import (
    Sample,
    Verdict,
    _build_content,
    _extract_rail_outcome,
    _stratified_sample,
    build_parser,
    compute_metrics,
    parse_verdict,
    print_comparison,
    print_group_breakdown,
    print_policy_comparison,
    print_results,
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


# ---------------------------------------------------------------------------
# Dataset loader tests (mock sys.modules for optional deps)
# ---------------------------------------------------------------------------


def _make_ds_rows(*rows):
    """Return an iterable that quacks like a HuggingFace dataset split."""
    return rows


class TestLoadAegis:
    def _fake_dataset(self, rows):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = mock_ds
        return datasets_mod

    def test_prompt_text_type(self):
        rows = [
            {
                "prompt": "hello",
                "response": "",
                "prompt_label": "safe",
                "response_label": "safe",
                "violated_categories": "",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            # reload uses the patched module within the with block
            samples = m.load_aegis(0, "prompt")
        assert len(samples) == 1
        assert samples[0].ground_truth == "safe"

    def test_response_text_type(self):
        rows = [
            {
                "prompt": "",
                "response": "resp",
                "prompt_label": "safe",
                "response_label": "unsafe",
                "violated_categories": "Violence",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_aegis(0, "response")
        assert len(samples) == 1
        assert samples[0].ground_truth == "unsafe"

    def test_both_text_type(self):
        rows = [
            {
                "prompt": "p",
                "response": "r",
                "prompt_label": "safe",
                "response_label": "unsafe",
                "violated_categories": "",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_aegis(0, "both")
        assert len(samples) == 2

    def test_empty_text_rows_skipped(self):
        rows = [
            {"prompt": "", "response": "", "prompt_label": "safe", "response_label": "safe", "violated_categories": ""},
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_aegis(0, "both")
        assert len(samples) == 0


class TestLoadNemotronSg:
    def _fake_dataset(self, rows):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = mock_ds
        return datasets_mod

    def test_prompt_basic(self):
        rows = [
            {
                "prompt": "hi",
                "response": "",
                "prompt_label": "safe",
                "response_label": None,
                "violated_categories": "",
                "language": "en",
                "tag": "t1",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_nemotron_sg(0, "prompt")
        assert len(samples) == 1
        assert samples[0].extra["language"] == "en"

    def test_response_valid_label(self):
        rows = [
            {
                "prompt": "",
                "response": "resp",
                "prompt_label": "safe",
                "response_label": "unsafe",
                "violated_categories": "",
                "language": "fr",
                "tag": "",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_nemotron_sg(0, "response")
        assert len(samples) == 1
        assert samples[0].ground_truth == "unsafe"

    def test_response_invalid_label_skipped(self):
        # response_label=None → empty string → not in {"safe","unsafe"} → skipped
        rows = [
            {
                "prompt": "",
                "response": "resp",
                "prompt_label": "safe",
                "response_label": None,
                "violated_categories": "",
                "language": "en",
                "tag": "",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_nemotron_sg(0, "response")
        assert len(samples) == 0

    def test_both_text_type(self):
        rows = [
            {
                "prompt": "p",
                "response": "r",
                "prompt_label": "unsafe",
                "response_label": "safe",
                "violated_categories": "",
                "language": "de",
                "tag": "",
            },
        ]
        ds_mod = self._fake_dataset(rows)
        with patch.dict(sys.modules, {"datasets": ds_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_nemotron_sg(0, "both")
        assert len(samples) == 2


class TestLoadCosapien:
    def _fake_dataset_for_split(self, rows):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(rows))
        return mock_ds

    def test_loads_all_splits(self):
        row = {"prompt": "do thing", "type": "safe", "scenario": "game_development policy text"}
        datasets_mod = MagicMock()
        # Must return a fresh iterable each call (each split call exhausts the iterator)
        datasets_mod.load_dataset.side_effect = lambda *a, **kw: self._fake_dataset_for_split([row])
        with patch.dict(sys.modules, {"datasets": datasets_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_cosapien(0)
        # 5 splits × 1 row = 5 samples
        assert len(samples) == 5
        assert datasets_mod.load_dataset.call_count == 5

    def test_allowed_maps_to_safe(self):
        row = {"prompt": "x", "type": "allowed", "scenario": "policy"}
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = self._fake_dataset_for_split([row])
        with patch.dict(sys.modules, {"datasets": datasets_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_cosapien(0)
        assert all(s.ground_truth == "safe" for s in samples)

    def test_disallowed_maps_to_unsafe(self):
        row = {"prompt": "x", "type": "disallowed", "scenario": "policy"}
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = self._fake_dataset_for_split([row])
        with patch.dict(sys.modules, {"datasets": datasets_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_cosapien(0)
        assert all(s.ground_truth == "unsafe" for s in samples)

    def test_partial_maps_to_unsafe(self):
        row = {"prompt": "x", "type": "partial", "scenario": "policy"}
        datasets_mod = MagicMock()
        datasets_mod.load_dataset.return_value = self._fake_dataset_for_split([row])
        with patch.dict(sys.modules, {"datasets": datasets_mod}):
            from benchmark.content_safety_parity import content_safety_parity as m

            samples = m.load_cosapien(0)
        assert all(s.ground_truth == "unsafe" for s in samples)


class TestLoadDynaguardrail:
    def _make_mocks(self, rows):
        """Return (hf_hub_mod, pandas_mod) mocks given row dicts."""
        hf_mod = MagicMock()
        hf_mod.hf_hub_download.return_value = "/fake/path.parquet"

        df = MagicMock()
        df.iterrows.return_value = iter(enumerate(rows))
        pandas_mod = MagicMock()
        pandas_mod.read_parquet.return_value = df
        return hf_mod, pandas_mod

    def test_requires_hf_token(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure HF_TOKEN is not set
            os.environ.pop("HF_TOKEN", None)
            from benchmark.content_safety_parity import content_safety_parity as m

            with pytest.raises(SystemExit):
                m.load_dynaguardrail(0)

    def test_loads_rows(self):
        rows = [
            {"prompt": "invest in this", "label": "unsafe", "dataset_type": "finance", "trickiness": "high"},
            {"prompt": "hello world", "label": "safe", "dataset_type": "safety", "trickiness": "low"},
        ]
        hf_mod, pd_mod = self._make_mocks(rows)
        with patch.dict(os.environ, {"HF_TOKEN": "fake-token"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod, "pandas": pd_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_dynaguardrail(0)
        assert len(samples) == 2
        assert samples[0].ground_truth == "unsafe"
        assert samples[0].extra["domain"] == "finance"

    def test_invalid_label_becomes_unsafe(self):
        rows = [
            {"prompt": "text", "label": "other", "dataset_type": "unknown", "trickiness": "unknown"},
        ]
        hf_mod, pd_mod = self._make_mocks(rows)
        with patch.dict(os.environ, {"HF_TOKEN": "fake-token"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod, "pandas": pd_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_dynaguardrail(0)
        assert samples[0].ground_truth == "unsafe"


class TestLoadVlguard:
    def _make_zip(self, tmpdir, image_name="test/img.jpg", image_data=b"\xff\xd8\xff"):
        """Create a minimal zip with one image file and return its path."""
        zip_path = tmpdir / "test.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr(image_name, image_data)
        return str(zip_path)

    def _make_json(self, tmpdir, rows):
        json_path = tmpdir / "test.json"
        json_path.write_text(json.dumps(rows))
        return str(json_path)

    def _make_hf_mod(self, json_path, zip_path):
        hf_mod = MagicMock()
        call_count = [0]

        def side_effect(repo, filename, **kwargs):
            call_count[0] += 1
            if filename == "test.json":
                return json_path
            return zip_path

        hf_mod.hf_hub_download.side_effect = side_effect
        return hf_mod

    def test_requires_hf_token(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("HF_TOKEN", None)
            from benchmark.content_safety_parity import content_safety_parity as m

            with pytest.raises(SystemExit):
                m.load_vlguard(0)

    def test_instruction_mode_safe_rows(self, tmp_path):
        rows = [
            {
                "safe": True,
                "image": "img.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [
                    {"safe_instruction": "describe safely", "unsafe_instruction": "do harm", "response": "ok"},
                ],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)
        hf_mod = self._make_hf_mod(json_path, zip_path)
        images_dir = str(tmp_path / "imgs")
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="instruction", images_dir=images_dir)
        # Only safe_instruction sampled in "instruction" mode
        assert len(samples) == 1
        assert samples[0].ground_truth == "safe"

    def test_all_mode_includes_unsafe_instruction(self, tmp_path):
        rows = [
            {
                "safe": True,
                "image": "img.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [
                    {"safe_instruction": "safe text", "unsafe_instruction": "unsafe text", "response": "ok"},
                ],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)
        hf_mod = self._make_hf_mod(json_path, zip_path)
        images_dir = str(tmp_path / "imgs2")
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="all", images_dir=images_dir)
        assert len(samples) == 2

    def test_unsafe_rows(self, tmp_path):
        rows = [
            {
                "safe": False,
                "image": "img.jpg",
                "harmful_category": "violence",
                "harmful_subcategory": "physical",
                "instr-resp": [
                    {"instruction": "how to harm", "response": "no"},
                ],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)
        hf_mod = self._make_hf_mod(json_path, zip_path)
        images_dir = str(tmp_path / "imgs3")
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="instruction", images_dir=images_dir)
        assert len(samples) == 1
        assert samples[0].ground_truth == "unsafe"

    def test_missing_image_skipped(self, tmp_path):
        rows = [
            {
                "safe": True,
                "image": "nonexistent.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [{"safe_instruction": "hi", "unsafe_instruction": "bad", "response": "ok"}],
            }
        ]
        # Zip doesn't have the image
        zip_path = self._make_zip(tmp_path, image_name="test/other.jpg")
        json_path = self._make_json(tmp_path, rows)
        hf_mod = self._make_hf_mod(json_path, zip_path)
        images_dir = str(tmp_path / "imgs4")
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="instruction", images_dir=images_dir)
        assert len(samples) == 0

    def test_local_files_only_fallback(self, tmp_path):
        """When local_files_only=True raises, the non-cached download is used as fallback."""
        rows = [
            {
                "safe": False,
                "image": "img.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [{"instruction": "bad", "response": "no"}],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)
        images_dir = str(tmp_path / "imgs_fallback")

        hf_mod = MagicMock()
        call_log = []

        def hf_side_effect(repo, filename, repo_type=None, token=None, local_files_only=False):
            call_log.append((filename, local_files_only))
            if local_files_only:
                raise FileNotFoundError("not cached")
            return json_path if filename == "test.json" else zip_path

        hf_mod.hf_hub_download.side_effect = hf_side_effect
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="instruction", images_dir=images_dir)
        # Fallback calls (local_files_only=False) should have been made
        fallback_calls = [c for c in call_log if not c[1]]
        assert len(fallback_calls) == 2

    def test_default_images_dir(self, tmp_path):
        """When images_dir is None, img_root defaults to zip sibling dir."""
        rows = [
            {
                "safe": False,
                "image": "img.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [{"instruction": "bad", "response": "no"}],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)

        hf_mod = MagicMock()
        hf_mod.hf_hub_download.side_effect = lambda repo, filename, **kw: (
            json_path if filename == "test.json" else str(zip_path)
        )
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                # images_dir=None → extracts to zip_path.parent / "vlguard_test_images"
                samples = m.load_vlguard(0, mode="instruction", images_dir=None)
        # Image is extracted from the zip, so 1 unsafe sample is found
        assert len(samples) == 1

    def test_sentinel_skips_extraction(self, tmp_path):
        """When sentinel file already exists, extraction is skipped."""
        rows = [
            {
                "safe": False,
                "image": "img.jpg",
                "harmful_category": "n/a",
                "harmful_subcategory": "n/a",
                "instr-resp": [{"instruction": "bad", "response": "no"}],
            }
        ]
        zip_path = self._make_zip(tmp_path)
        json_path = self._make_json(tmp_path, rows)
        hf_mod = self._make_hf_mod(json_path, zip_path)
        images_dir = str(tmp_path / "imgs5")
        # Pre-create sentinel and the image so load_vlguard doesn't extract
        img_root = Path(images_dir)
        (img_root / "test").mkdir(parents=True, exist_ok=True)
        (img_root / "test" / "img.jpg").write_bytes(b"\xff\xd8\xff")
        (img_root / "_extraction_complete").touch()
        with patch.dict(os.environ, {"HF_TOKEN": "tok"}):
            with patch.dict(sys.modules, {"huggingface_hub": hf_mod}):
                from benchmark.content_safety_parity import content_safety_parity as m

                samples = m.load_vlguard(0, mode="instruction", images_dir=images_dir)
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# _call_nim_once tests
# ---------------------------------------------------------------------------


class TestCallNimOnce:
    def _safe_response(self):
        resp = MagicMock()
        resp.choices[0].message.content = "User Safety: safe\nSafety Categories: None"
        return resp

    def _unsafe_response(self):
        resp = MagicMock()
        resp.choices[0].message.content = "User Safety: unsafe\nSafety Categories: Violence"
        return resp

    def test_happy_path_safe(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._safe_response())
        sem = asyncio.Semaphore(1)
        v = asyncio.run(m._call_nim_once(client, "text", sem))
        assert v.is_safe is True
        assert v.error is False

    def test_happy_path_unsafe(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._unsafe_response())
        sem = asyncio.Semaphore(1)
        v = asyncio.run(m._call_nim_once(client, "text", sem))
        assert v.is_safe is False

    def test_custom_policy_kwargs(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=self._safe_response())
        sem = asyncio.Semaphore(1)
        asyncio.run(m._call_nim_once(client, "text", sem, custom_policy="my policy"))
        call_kwargs = client.chat.completions.create.call_args[1]
        assert call_kwargs["extra_body"]["chat_template_kwargs"] == {"custom_policy": "my policy"}

    def test_permanent_error_breaks_immediately(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        exc = RuntimeError("bad request")
        exc.response = MagicMock()
        exc.response.status_code = 400
        exc.response.headers = {}
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=exc)
        sem = asyncio.Semaphore(1)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            v = asyncio.run(m._call_nim_once(client, "text", sem, _retries=3))
        # Should have only called create once (break on 400)
        assert client.chat.completions.create.call_count == 1
        assert v.error is True

    def test_transient_error_retries_and_exhausts(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        exc = RuntimeError("server error")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=exc)
        sem = asyncio.Semaphore(1)
        with patch("asyncio.sleep", new_callable=AsyncMock):
            v = asyncio.run(m._call_nim_once(client, "text", sem, _retries=2))
        assert client.chat.completions.create.call_count == 2
        assert v.error is True

    def test_invalid_retry_after_header_uses_backoff(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        exc = RuntimeError("rate limited")
        exc.response = MagicMock()
        exc.response.status_code = 429
        exc.response.headers = {"retry-after": "not-a-number"}
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[exc, self._safe_response()])
        sem = asyncio.Semaphore(1)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            v = asyncio.run(m._call_nim_once(client, "text", sem, _retries=3))
        assert v.error is False
        # sleep called with exponential backoff (not from header since header was invalid)
        mock_sleep.assert_called_once()

    def test_retry_after_header_honored(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        exc = RuntimeError("rate limited")
        exc.response = MagicMock()
        exc.response.status_code = 429
        exc.response.headers = {"retry-after": "0.01"}
        # Only fail once, then succeed
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=[exc, self._safe_response()])
        sem = asyncio.Semaphore(1)
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            v = asyncio.run(m._call_nim_once(client, "text", sem, _retries=3))
        assert v.error is False
        # sleep should have been called with at least 0.01 (from header)
        mock_sleep.assert_called_once()
        sleep_arg = mock_sleep.call_args[0][0]
        assert sleep_arg >= 0.01


# ---------------------------------------------------------------------------
# run_direct_eval tests
# ---------------------------------------------------------------------------


class TestRunDirectEval:
    def _make_openai_mod(self, verdict):
        openai_mod = MagicMock()
        mock_client = MagicMock()
        openai_mod.AsyncOpenAI.return_value = mock_client
        return openai_mod, mock_client

    def test_response_text_type_raises(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [Sample(text="x", ground_truth="safe", text_type="response")]
        openai_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": openai_mod}):
            with pytest.raises(ValueError, match="response"):
                asyncio.run(m.run_direct_eval(samples, "http://localhost", "key", 1))

    def test_happy_path(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        safe_v = Verdict(is_safe=True, raw="ok", latency_ms=10.0)
        samples = [Sample(text="hello", ground_truth="safe")]
        openai_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": openai_mod}):
            with patch(
                "benchmark.content_safety_parity.content_safety_parity._call_nim_once",
                new=AsyncMock(return_value=safe_v),
            ):
                verdicts = asyncio.run(m.run_direct_eval(samples, "http://localhost", "key", 1))
        assert len(verdicts) == 1
        assert verdicts[0].is_safe is True

    def test_no_categories_warning_printed(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m

        # Unsafe verdict with no categories → warning
        unsafe_v = Verdict(is_safe=False, raw="unsafe", latency_ms=10.0, categories=[])
        samples = [Sample(text="bad", ground_truth="unsafe")]
        openai_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": openai_mod}):
            with patch(
                "benchmark.content_safety_parity.content_safety_parity._call_nim_once",
                new=AsyncMock(return_value=unsafe_v),
            ):
                asyncio.run(m.run_direct_eval(samples, "http://localhost", "key", 1))
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_custom_policy_no_warning(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m

        unsafe_v = Verdict(is_safe=False, raw="unsafe", latency_ms=10.0, categories=[])
        samples = [Sample(text="bad", ground_truth="unsafe")]
        openai_mod = MagicMock()
        with patch.dict(sys.modules, {"openai": openai_mod}):
            with patch(
                "benchmark.content_safety_parity.content_safety_parity._call_nim_once",
                new=AsyncMock(return_value=unsafe_v),
            ):
                asyncio.run(m.run_direct_eval(samples, "http://localhost", "key", 1, use_custom_policy=True))
        out = capsys.readouterr().out
        # No category warning when use_custom_policy=True
        assert "Warning: no 'Safety Categories'" not in out


# ---------------------------------------------------------------------------
# run_guardrails_eval tests
# ---------------------------------------------------------------------------


class TestRunGuardrailsEval:
    def setup_method(self):
        self._env_patcher = patch.dict(os.environ, {})
        self._env_patcher.start()

    def teardown_method(self):
        self._env_patcher.stop()

    def _make_nemo_mocks(self, generate_result, generate_exc=None):
        """Return patched nemoguardrails module mocks."""

        nemo_mod = MagicMock()
        rails_config = MagicMock()
        rails_config.models = []
        nemo_mod.RailsConfig.from_path.return_value = rails_config

        rails_instance = MagicMock()
        if generate_exc:
            rails_instance.generate_async = AsyncMock(side_effect=generate_exc)
        else:
            rails_instance.generate_async = AsyncMock(return_value=generate_result)
        nemo_mod.LLMRails.return_value = rails_instance
        return nemo_mod

    def _make_result(self, outcome):
        """Build a fake generate_async result with the given RailOutcome."""

        class FakeAction:
            def __init__(self, rv):
                self.return_value = rv

        class FakeRail:
            def __init__(self, rv):
                self.executed_actions = [FakeAction(rv)]

        class FakeLog:
            def __init__(self, rv):
                self.activated_rails = [FakeRail(rv)]

        class FakeResult:
            def __init__(self, rv):
                self.log = FakeLog(rv)

        return FakeResult(outcome)

    def test_block_verdict(self):
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.block(metadata={"policy_violations": ["Violence"]})
        result = self._make_result(outcome)
        nemo_mod = self._make_nemo_mocks(result)
        samples = [Sample(text="bad text", ground_truth="unsafe")]
        with patch.dict(
            sys.modules,
            {
                "nemoguardrails": nemo_mod,
                "nemoguardrails.LLMRails": nemo_mod.LLMRails,
                "nemoguardrails.RailsConfig": nemo_mod.RailsConfig,
            },
        ):
            with patch("benchmark.content_safety_parity.content_safety_parity.run_guardrails_eval.__module__"):
                pass
        # Patch at the module-level import point
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                verdicts = asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        assert verdicts[0].is_safe is False
        assert "Violence" in verdicts[0].categories

    def test_allow_verdict(self):
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(outcome)
        nemo_mod = self._make_nemo_mocks(result)
        samples = [Sample(text="safe text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                verdicts = asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        assert verdicts[0].is_safe is True
        assert verdicts[0].error is False

    def test_none_outcome_is_error(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m

        # Return value is a string (not RailOutcome) → _extract_rail_outcome returns None
        result = MagicMock()
        result.log = MagicMock()
        result.log.activated_rails = []
        nemo_mod = self._make_nemo_mocks(result)
        samples = [Sample(text="text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                verdicts = asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        assert verdicts[0].error is True

    def test_exception_in_generate_async(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m

        nemo_mod = self._make_nemo_mocks(None, generate_exc=RuntimeError("oops"))
        samples = [Sample(text="text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                verdicts = asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        assert verdicts[0].error is True
        out = capsys.readouterr().out
        assert "Warning" in out

    def test_failed_outcome_is_error(self):
        # RailOutcome is a frozen dataclass; create one with failed=True directly
        import dataclasses

        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.block(metadata={"policy_violations": []})
        failed_outcome = dataclasses.replace(outcome, failed=True)
        result = self._make_result(failed_outcome)
        nemo_mod = self._make_nemo_mocks(result)
        samples = [Sample(text="text", ground_truth="unsafe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                verdicts = asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        assert verdicts[0].error is True

    def test_image_sample_warning(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(outcome)
        nemo_mod = self._make_nemo_mocks(result)
        samples = [Sample(text="text", ground_truth="safe", extra={"image_path": "/some/img.jpg"})]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", None))
        out = capsys.readouterr().out
        assert "image" in out.lower()

    def test_no_nim_base_url_sets_api_key_env_var(self):
        """When nim_base_url is None, content_safety and main models get api_key_env_var."""
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(outcome)

        content_model = MagicMock()
        content_model.type = "content_safety"
        content_model.parameters = {}
        main_model = MagicMock()
        main_model.type = "main"
        main_model.parameters = {}
        rails_config = MagicMock()
        rails_config.models = [content_model, main_model]

        nemo_mod = MagicMock()
        nemo_mod.RailsConfig.from_path.return_value = rails_config
        rails_instance = MagicMock()
        rails_instance.generate_async = AsyncMock(return_value=result)
        nemo_mod.LLMRails.return_value = rails_instance

        samples = [Sample(text="text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", nim_base_url=None))
        # With no nim_base_url, api_key_env_var should be set on both models
        assert content_model.api_key_env_var == "NVIDIA_API_KEY"
        assert main_model.api_key_env_var == "NVIDIA_API_KEY"

    def test_main_model_stub_wired(self):
        """The main model config entry gets stubbed when nim_base_url is provided."""
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(outcome)

        content_model = MagicMock()
        content_model.type = "content_safety"
        content_model.parameters = {}
        main_model = MagicMock()
        main_model.type = "main"
        main_model.parameters = {}
        rails_config = MagicMock()
        rails_config.models = [content_model, main_model]

        nemo_mod = MagicMock()
        nemo_mod.RailsConfig.from_path.return_value = rails_config
        rails_instance = MagicMock()
        rails_instance.generate_async = AsyncMock(return_value=result)
        nemo_mod.LLMRails.return_value = rails_instance

        samples = [Sample(text="text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", "http://nim:8000"))
        # main model should have been stubbed with the NIM model and base_url
        assert main_model.model == "nvidia/nemotron-3.5-content-safety"
        assert main_model.parameters["base_url"] == "http://nim:8000"

    def test_nim_base_url_propagates(self):
        from benchmark.content_safety_parity import content_safety_parity as m
        from nemoguardrails.actions.rail_outcome import RailOutcome

        outcome = RailOutcome.allow(metadata={"policy_violations": []})
        result = self._make_result(outcome)

        # Give config a model with type="content_safety"
        model_cfg = MagicMock()
        model_cfg.type = "content_safety"
        model_cfg.parameters = {}
        rails_config = MagicMock()
        rails_config.models = [model_cfg]

        nemo_mod = MagicMock()
        nemo_mod.RailsConfig.from_path.return_value = rails_config
        rails_instance = MagicMock()
        rails_instance.generate_async = AsyncMock(return_value=result)
        nemo_mod.LLMRails.return_value = rails_instance

        samples = [Sample(text="text", ground_truth="safe")]
        with patch("nemoguardrails.LLMRails", nemo_mod.LLMRails):
            with patch("nemoguardrails.RailsConfig", nemo_mod.RailsConfig):
                asyncio.run(m.run_guardrails_eval(samples, "/fake/config", "key", "http://nim:8000"))
        assert model_cfg.parameters["base_url"] == "http://nim:8000"


# ---------------------------------------------------------------------------
# print_* function tests
# ---------------------------------------------------------------------------


class TestPrintResults:
    def _metrics(self, **overrides):
        base = {
            "accuracy": 0.9,
            "f1": 0.88,
            "precision": 0.9,
            "recall": 0.86,
            "tp": 9,
            "fp": 1,
            "tn": 8,
            "fn": 2,
            "total": 20,
            "errors": 0,
        }
        base.update(overrides)
        return base

    def test_prints_without_latencies(self, capsys):
        verdicts = [Verdict(is_safe=True, raw="ok", latency_ms=0.0)]
        print_results("aegis", "prompt", self._metrics(), verdicts)
        out = capsys.readouterr().out
        assert "accuracy" in out.lower()

    def test_prints_with_latencies(self, capsys):
        verdicts = [Verdict(is_safe=True, raw="ok", latency_ms=100.0)]
        print_results("aegis", "prompt", self._metrics(), verdicts)
        out = capsys.readouterr().out
        assert "p50" in out

    def test_prints_error_count(self, capsys):
        verdicts = [Verdict(is_safe=True, raw="ERROR", latency_ms=0.0, error=True)]
        print_results("aegis", "prompt", self._metrics(errors=1), verdicts)
        out = capsys.readouterr().out
        assert "errored" in out


class TestPrintGroupBreakdown:
    def test_single_group_returns_early(self, capsys):
        samples = [Sample(text="x", ground_truth="safe", extra={"language": "en"})]
        verdicts = [Verdict(is_safe=True, raw="ok")]
        print_group_breakdown(samples, verdicts, extra_key="language")
        out = capsys.readouterr().out
        # With only 1 group, early return — no header printed
        assert "BREAKDOWN" not in out

    def test_multiple_groups(self, capsys):
        samples = [
            Sample(text="x", ground_truth="safe", extra={"language": "en"}),
            Sample(text="y", ground_truth="unsafe", extra={"language": "fr"}),
        ]
        verdicts = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=False, raw="bad")]
        print_group_breakdown(samples, verdicts, extra_key="language", group_label="Lang")
        out = capsys.readouterr().out
        assert "BREAKDOWN" in out
        assert "en" in out
        assert "fr" in out


class TestPrintPolicyComparison:
    def test_basic_output(self, capsys):
        samples = [
            Sample(text="x", ground_truth="safe", extra={"scenario": "game_development", "type": "safe"}),
            Sample(text="y", ground_truth="unsafe", extra={"scenario": "game_development", "type": "disallowed"}),
        ]
        default_v = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=False, raw="bad")]
        custom_v = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=False, raw="bad")]
        print_policy_comparison(samples, default_v, custom_v)
        out = capsys.readouterr().out
        assert "game_development" in out
        assert "OVERALL" in out

    def test_disagreement_breakdown_with_type(self, capsys):
        samples = [
            Sample(text="x", ground_truth="safe", extra={"scenario": "s1", "type": "allowed"}),
            Sample(text="y", ground_truth="unsafe", extra={"scenario": "s1", "type": "disallowed"}),
        ]
        default_v = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=False, raw="bad")]
        custom_v = [Verdict(is_safe=False, raw="bad"), Verdict(is_safe=True, raw="ok")]  # disagree on both
        print_policy_comparison(samples, default_v, custom_v)
        out = capsys.readouterr().out
        assert "Disagreements by original label type" in out

    def test_policy_errors_excluded(self, capsys):
        samples = [
            Sample(text="x", ground_truth="safe", extra={"scenario": "s1", "type": "safe"}),
        ]
        default_v = [Verdict(is_safe=True, raw="ERROR", error=True)]
        custom_v = [Verdict(is_safe=True, raw="ok")]
        print_policy_comparison(samples, default_v, custom_v)
        out = capsys.readouterr().out
        assert "errored pair" in out

    def test_no_type_key_skips_breakdown(self, capsys):
        samples = [
            Sample(text="x", ground_truth="safe", extra={"scenario": "s1"}),
        ]
        default_v = [Verdict(is_safe=True, raw="ok")]
        custom_v = [Verdict(is_safe=True, raw="ok")]
        print_policy_comparison(samples, default_v, custom_v)
        out = capsys.readouterr().out
        assert "Disagreements by original label type" not in out


class TestPrintComparison:
    def test_no_valid_pairs_inconclusive(self, capsys):
        samples = [Sample(text="x", ground_truth="safe")]
        direct_v = [Verdict(is_safe=True, raw="ERROR", error=True)]
        gr_v = [Verdict(is_safe=True, raw="ERROR", error=True)]
        print_comparison(samples, direct_v, gr_v)
        out = capsys.readouterr().out
        assert "Inconclusive" in out

    def test_full_agreement(self, capsys):
        samples = [Sample(text="x", ground_truth="safe")]
        direct_v = [Verdict(is_safe=True, raw="ok")]
        gr_v = [Verdict(is_safe=True, raw="ok")]
        print_comparison(samples, direct_v, gr_v)
        out = capsys.readouterr().out
        assert "No disagreements" in out

    def test_disagreements_shown(self, capsys):
        samples = [Sample(text="bad text", ground_truth="unsafe")]
        direct_v = [Verdict(is_safe=False, raw="unsafe")]
        gr_v = [Verdict(is_safe=True, raw="safe")]
        print_comparison(samples, direct_v, gr_v)
        out = capsys.readouterr().out
        assert "Disagreements" in out
        assert "bad text" in out

    def test_many_disagreements_truncated(self, capsys):
        samples = [Sample(text=f"text{i}", ground_truth="unsafe") for i in range(15)]
        direct_v = [Verdict(is_safe=False, raw="unsafe") for _ in range(15)]
        gr_v = [Verdict(is_safe=True, raw="safe") for _ in range(15)]
        print_comparison(samples, direct_v, gr_v)
        out = capsys.readouterr().out
        assert "... and 5 more" in out

    def test_image_samples_skipped(self, capsys):
        samples = [
            Sample(text="img_text", ground_truth="safe", extra={"image_path": "/img.jpg"}),
            Sample(text="plain_text", ground_truth="safe"),
        ]
        direct_v = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=True, raw="ok")]
        gr_v = [Verdict(is_safe=True, raw="ok"), Verdict(is_safe=True, raw="ok")]
        print_comparison(samples, direct_v, gr_v)
        out = capsys.readouterr().out
        assert "image samples excluded" in out


# ---------------------------------------------------------------------------
# build_parser tests
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_returns_argument_parser(self):
        import argparse

        p = build_parser()
        assert isinstance(p, argparse.ArgumentParser)

    def test_default_dataset(self):
        p = build_parser()
        args = p.parse_args([])
        assert args.dataset == "aegis"

    def test_all_flags_parseable(self):
        p = build_parser()
        args = p.parse_args(
            [
                "--dataset",
                "cosapien",
                "--text-type",
                "prompt",
                "--sample",
                "50",
                "--full",
                "--nim-base-url",
                "http://localhost:8000/v1",
                "--concurrency",
                "4",
                "--vlguard-mode",
                "all",
                "--compare-guardrails",
                "--guardrails-config",
                "/tmp/cfg",
                "--guardrails-concurrency",
                "3",
            ]
        )
        assert args.dataset == "cosapien"
        assert args.full is True
        assert args.concurrency == 4


# ---------------------------------------------------------------------------
# main() tests
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the async main() orchestration function."""

    def _safe_verdict(self):
        return Verdict(is_safe=True, raw="ok", latency_ms=50.0)

    def _patch_argv(self, args_list):
        return patch("sys.argv", ["prog"] + args_list)

    def test_no_api_key_no_base_url_exits(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        with self._patch_argv([]):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("NVIDIA_API_KEY", None)
                with pytest.raises(SystemExit):
                    asyncio.run(m.main())

    def test_response_text_type_exits(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        with self._patch_argv(["--text-type", "response", "--nim-base-url", "http://x"]):
            with pytest.raises(SystemExit):
                asyncio.run(m.main())

    def test_empty_samples_exits(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        with self._patch_argv(["--nim-base-url", "http://x"]):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"aegis": lambda a: []},
            ):
                with pytest.raises(SystemExit):
                    asyncio.run(m.main())

    def test_aegis_happy_path(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [Sample(text="hi", ground_truth="safe")]
        verdicts = [self._safe_verdict()]
        with self._patch_argv(["--nim-base-url", "http://x"]):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"aegis": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=verdicts),
                ):
                    asyncio.run(m.main())  # should not raise

    def test_cosapien_runs_custom_policy(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [
            Sample(
                text="hi",
                ground_truth="safe",
                extra={"scenario": "game_development", "type": "safe", "custom_policy": "policy"},
            )
        ]
        verdicts = [self._safe_verdict()]
        with self._patch_argv(["--nim-base-url", "http://x", "--dataset", "cosapien"]):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"cosapien": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=verdicts),
                ):
                    asyncio.run(m.main())

    def test_dynaguardrail_runs_custom_policy(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [
            Sample(
                text="invest",
                ground_truth="unsafe",
                extra={"domain": "finance", "trickiness": "high", "custom_policy": "policy"},
            )
        ]
        verdicts = [Verdict(is_safe=False, raw="unsafe", latency_ms=10.0)]
        with self._patch_argv(["--nim-base-url", "http://x", "--dataset", "dynaguardrail"]):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"dynaguardrail": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=verdicts),
                ):
                    asyncio.run(m.main())

    def test_vlguard_all_mode_breakdown(self):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [
            Sample(text="t", ground_truth="safe", extra={"category": "violence", "instr_type": "safe_instruction"}),
        ]
        verdicts = [self._safe_verdict()]
        with self._patch_argv(["--nim-base-url", "http://x", "--dataset", "vlguard", "--vlguard-mode", "all"]):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"vlguard": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=verdicts),
                ):
                    asyncio.run(m.main())

    def test_compare_guardrails_config_missing_warns(self, capsys):
        from benchmark.content_safety_parity import content_safety_parity as m

        samples = [Sample(text="hi", ground_truth="safe")]
        verdicts = [self._safe_verdict()]
        with self._patch_argv(
            ["--nim-base-url", "http://x", "--compare-guardrails", "--guardrails-config", "/no/such/path"]
        ):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"aegis": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=verdicts),
                ):
                    asyncio.run(m.main())
        out = capsys.readouterr().out
        assert "not found" in out

    def test_compare_guardrails_config_exists(self, tmp_path):
        from benchmark.content_safety_parity import content_safety_parity as m

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        samples = [Sample(text="hi", ground_truth="safe")]
        direct_verdicts = [self._safe_verdict()]
        gr_verdicts = [self._safe_verdict()]
        with self._patch_argv(
            ["--nim-base-url", "http://x", "--compare-guardrails", "--guardrails-config", str(cfg_dir)]
        ):
            with patch.dict(
                "benchmark.content_safety_parity.content_safety_parity.DATASET_LOADERS",
                {"aegis": lambda a: samples},
            ):
                with patch(
                    "benchmark.content_safety_parity.content_safety_parity.run_direct_eval",
                    new=AsyncMock(return_value=direct_verdicts),
                ):
                    with patch(
                        "benchmark.content_safety_parity.content_safety_parity.run_guardrails_eval",
                        new=AsyncMock(return_value=gr_verdicts),
                    ):
                        asyncio.run(m.main())
