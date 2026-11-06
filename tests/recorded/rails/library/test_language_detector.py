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

from __future__ import annotations

import pytest

from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.recorded.assertions import assert_rails_result
from tests.recorded.normalization import normalize_rails_result
from tests.recorded.rails.library.configs import LANGUAGE_DETECTOR_BLOCKLIST_CONFIG, LANGUAGE_DETECTOR_CONFIG
from tests.recorded.rails.library.helpers import check_rails
from tests.recorded.snapshots import snapshot

pytestmark = [pytest.mark.recorded, pytest.mark.asyncio]


def _stub_detect_language(monkeypatch: pytest.MonkeyPatch, lang: str) -> None:
    """Pin language detection to ``lang`` so tests are deterministic and network-free."""
    monkeypatch.setattr(
        "nemoguardrails.library.language_detector.actions.detect_language",
        lambda text, max_length: lang,
    )


async def test_language_detector_input_allows_english(monkeypatch):
    _stub_detect_language(monkeypatch, "en")

    result = await check_rails(
        LANGUAGE_DETECTOR_CONFIG,
        [{"role": "user", "content": "Can you explain your return policy?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "Can you explain your return policy?"}
    )


async def test_language_detector_input_blocks_spanish(monkeypatch):
    _stub_detect_language(monkeypatch, "es")

    result = await check_rails(
        LANGUAGE_DETECTOR_CONFIG,
        [{"role": "user", "content": "¿Puedes explicar tu política de devoluciones?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="language detector check input")
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "language detector check input",
            "content": "Lo siento, no puedo responder a eso.",
        }
    )


async def test_language_detector_blocklist_input_allows_english(monkeypatch):
    _stub_detect_language(monkeypatch, "en")

    result = await check_rails(
        LANGUAGE_DETECTOR_BLOCKLIST_CONFIG,
        [{"role": "user", "content": "Can you explain your return policy?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.PASSED)
    assert normalize_rails_result(result) == snapshot(
        {"status": "passed", "rail": None, "content": "Can you explain your return policy?"}
    )


async def test_language_detector_blocklist_input_blocks_spanish(monkeypatch):
    _stub_detect_language(monkeypatch, "es")

    result = await check_rails(
        LANGUAGE_DETECTOR_BLOCKLIST_CONFIG,
        [{"role": "user", "content": "¿Puedes explicar tu política de devoluciones?"}],
        rail_types=(RailType.INPUT,),
    )

    assert_rails_result(result, status=RailStatus.BLOCKED, rail="language detector check input")
    assert normalize_rails_result(result) == snapshot(
        {
            "status": "blocked",
            "rail": "language detector check input",
            "content": "Lo siento, no puedo responder a eso.",
        }
    )
