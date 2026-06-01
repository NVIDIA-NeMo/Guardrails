# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for sensitive_data_detection that do not require the optional
Presidio / spaCy stack. The integration-style cases live in
``test_sensitive_data_detection.py`` and are skipped when those dependencies
aren't installed.
"""

import pytest

from nemoguardrails import RailsConfig


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mask_sensitive_data_honors_configured_score_threshold(monkeypatch):
    """Regression: ``mask_sensitive_data`` must honor ``options.score_threshold``.

    The masking path previously called ``_get_analyzer()`` with no arguments,
    so the analyzer was built (and ``lru_cache``-d) at the default 0.4
    regardless of what the user configured. ``_get_analyzer`` is the only
    place ``default_score_threshold`` is set on the underlying Presidio
    ``AnalyzerEngine``, so values between the configured threshold and 0.4
    were still masked even though ``detect_sensitive_data`` (which already
    passed the threshold through) reported them as non-sensitive.
    """
    from nemoguardrails.library.sensitive_data_detection import actions as sdd

    captured: dict = {}

    class _StubAnalyzer:
        def analyze(self, text, language, entities, ad_hoc_recognizers):
            return []

    def _fake_get_analyzer(score_threshold: float = 0.4):
        captured["score_threshold"] = score_threshold
        return _StubAnalyzer()

    class _StubAnonymizer:
        def anonymize(self, text, analyzer_results, operators):
            class _R:
                pass

            r = _R()
            r.text = text
            return r

    monkeypatch.setattr(sdd, "_get_analyzer", _fake_get_analyzer)
    # `AnonymizerEngine` / `OperatorConfig` come from `presidio_anonymizer`,
    # which is optional and may not be importable in this CI tier — the
    # module-level `try/except ImportError: pass` swallows their absence, so
    # we set them on the module with `raising=False` for the stub run.
    monkeypatch.setattr(sdd, "AnonymizerEngine", _StubAnonymizer, raising=False)
    monkeypatch.setattr(sdd, "OperatorConfig", lambda *a, **kw: object(), raising=False)

    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:
                  input:
                    score_threshold: 0.85
                    entities:
                      - PERSON
        """,
    )

    await sdd.mask_sensitive_data(source="input", text="My name is John", config=config)

    assert captured["score_threshold"] == 0.85
