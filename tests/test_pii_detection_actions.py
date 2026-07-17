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

from typing import Any

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.library.gliner import actions as gliner_actions
from nemoguardrails.library.gliner.actions import gliner_detect_pii
from nemoguardrails.library.privateai import actions as privateai_actions
from nemoguardrails.library.privateai.actions import detect_pii
from nemoguardrails.library.sensitive_data_detection import actions as sensitive_data_actions
from nemoguardrails.library.sensitive_data_detection.actions import detect_sensitive_data


def _privateai_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: http://localhost:8080/process
                  input:
                    entities:
                      - EMAIL_ADDRESS
        """
    )


def _gliner_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                gliner:
                  server_endpoint: http://localhost:8080/gliner
                  input:
                    entities:
                      - email
        """
    )


def _sensitive_data_config() -> RailsConfig:
    return RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                sensitive_data_detection:
                  input:
                    entities:
                      - PERSON
        """
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ([{"entities_present": False}], RailOutcome.allow(metadata={"has_pii": False})),
        ([{"entities_present": True}], RailOutcome.block(metadata={"has_pii": True})),
    ],
)
async def test_privateai_detect_pii_returns_rail_outcome(monkeypatch, response, expected):
    async def fake_private_ai_request(text, enabled_entities, server_endpoint, api_key):
        return response

    monkeypatch.setattr(privateai_actions, "private_ai_request", fake_private_ai_request)

    outcome = await detect_pii(source="input", text="hello", config=_privateai_config())

    assert outcome == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("total_entities", "expected"),
    [
        (0, RailOutcome.allow(metadata={"has_pii": False})),
        (1, RailOutcome.block(metadata={"has_pii": True})),
    ],
)
async def test_gliner_detect_pii_returns_rail_outcome(monkeypatch, total_entities, expected):
    async def fake_gliner_request(**kwargs):
        return {"total_entities": total_entities}

    monkeypatch.setattr(gliner_actions, "gliner_request", fake_gliner_request)
    monkeypatch.setattr(gliner_actions, "_resolve_api_key", lambda gliner_config: None)

    outcome = await gliner_detect_pii(source="input", text="hello", config=_gliner_config())

    assert outcome == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("analyzer_results", "expected"),
    [
        ([], RailOutcome.allow(metadata={"has_sensitive_data": False})),
        (["PERSON"], RailOutcome.block(metadata={"has_sensitive_data": True})),
    ],
)
async def test_sensitive_data_detect_returns_rail_outcome(monkeypatch, analyzer_results, expected):
    class FakeAnalyzer:
        def analyze(self, **kwargs) -> list[Any]:
            return analyzer_results

    monkeypatch.setattr(sensitive_data_actions, "_get_analyzer", lambda score_threshold=0.4: FakeAnalyzer())
    monkeypatch.setattr(sensitive_data_actions, "_get_ad_hoc_recognizers", lambda sdd_config: [])

    outcome = await detect_sensitive_data(source="input", text="hello", config=_sensitive_data_config())

    assert outcome == expected


@pytest.mark.parametrize(
    "outcome_func",
    [
        privateai_actions._mask_pii_outcome,
        gliner_actions._mask_pii_outcome,
        sensitive_data_actions._mask_sensitive_data_outcome,
    ],
)
@pytest.mark.parametrize(
    ("masked_text", "expected"),
    [
        ("hello", RailOutcome.allow(metadata={"source": "input", "text": "hello", "masked_text": "hello"})),
        (
            "masked",
            RailOutcome.transform(
                [(TransformTarget.USER_MESSAGE, "masked")],
                metadata={"source": "input", "text": "hello", "masked_text": "masked"},
            ),
        ),
    ],
)
def test_mask_helpers_return_rail_outcome(outcome_func, masked_text, expected):
    assert outcome_func("input", "hello", masked_text) == expected
