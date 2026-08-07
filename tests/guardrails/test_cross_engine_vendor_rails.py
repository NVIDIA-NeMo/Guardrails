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

"""Cross-engine equivalence for the HTTP-vendor rails IORails newly runs.

One config, one canned vendor response, both engines, same user-visible answer. Widening the
enabled tier is one line per rail; these cases are what makes each of those lines defensible.

Both engines are handed the **same** ``RecordingHTTPClient``, so the vendor's reply is fixed
and the only variable left is the engine. LLMRails takes it through
``register_action_param``, which is how its Colang runtime injects action parameters by name;
IORails takes it through the client its ``EngineRegistry`` manages. Everything below that
seam — the real action body, its config parsing, its response parser — executes on both
sides, which is what distinguishes this from ``test_runtime_flow_gate_equivalence.py``,
where the action is stubbed and a rewritten rail would stay green.
"""

import copy
import json
import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.http import HTTPResponse
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.testing import RecordingHTTPClient
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

# Clavata's verdict is a nested job/result/report document, so its own test factory builds it
# rather than a hand-copied literal that would drift from the models it is parsed into.
from tests.test_clavata import create_clavata_response as clavata_response
from tests.utils import TestChat

USER_INPUT = "hello there"
MAIN_OUTPUT = "Hello! How can I help?"

PRIVATEAI_CONFIG = {
    "privateai": {
        "server_endpoint": "http://privateai.example/process",
        "input": {"entities": ["NAME"]},
        "output": {"entities": ["NAME"]},
    }
}
TREND_CONFIG = {
    "trend_micro": {
        "v1_url": "https://api.xdr.trendmicro.com/v3.0/aiSecurity/applyGuardrails",
        "api_key_env_var": "V1_API_KEY",
        "application_name": "test-app",
    }
}
AI_DEFENSE_CONFIG = {"ai_defense": {}}
AI_DEFENSE_ENV = {
    "AI_DEFENSE_API_ENDPOINT": "https://ai-defense.example/api/v1/inspect/chat",
    "AI_DEFENSE_API_KEY": "test-key",
}
F5_ENV = {"F5_GUARDRAILS_API_KEY": "test-key", "F5_GUARDRAILS_API_URL": "https://f5.example"}
FIDDLER_CONFIG = {"fiddler": {"fiddler_endpoint": "https://fiddler.example", "safety_threshold": 0.5}}
FIDDLER_ENV = {"FIDDLER_API_KEY": "test-key", "FIDDLER_ENVIRON": "test"}
GLINER_CONFIG = {"gliner": {"server_endpoint": "http://gliner.example/v1/extract"}}
POLYGRAF_CONFIG = {
    "polygraf": {
        "server_endpoint": "http://polygraf.example/v1/pii/text-detect",
        "input": {"entities": ["Email", "Person"]},
        "output": {"entities": ["Email", "Person"]},
    }
}
POLICYAI_ENV = {
    "POLICYAI_API_KEY": "test-key",
    "POLICYAI_BASE_URL": "https://policyai.example",
    "POLICYAI_TAG_NAME": "test-tag",
}
JAILBREAK_CONFIG = {"jailbreak_detection": {"server_endpoint": "http://jailbreak.example/heuristics"}}
CLAVATA_POLICY_ID = "00000000-0000-0000-0000-000000000000"
CLAVATA_CONFIG = {
    "clavata": {
        "policies": {"Violence": CLAVATA_POLICY_ID},
        "input": {"policy": "Violence"},
        "output": {"policy": "Violence"},
    }
}


@dataclass(frozen=True)
class VendorRail:
    """One HTTP-backed rail, with the vendor replies that drive it either way.

    *llmrails_block_text* records what LLMRails says when this rail blocks. It is per-rail
    because LLMRails renders a block through the rail's own Colang flow — ``bot refuse to
    respond`` for most, ``bot inform answer unknown`` for the PII family — whereas IORails
    emits one ``REFUSAL_MESSAGE`` for every rail. Recording it here makes that delta a
    reviewable table entry rather than something a reader discovers in production.

    *volatile_body_keys* names request-body fields that legitimately differ between two
    calls, such as a per-request nonce, so body comparison stays meaningful.
    """

    rail_id: str
    flow: str
    direction: str
    allow_payload: Any
    block_payload: Any
    rails_config: dict = None  # type: ignore[assignment]
    env: dict = None  # type: ignore[assignment]
    llmrails_block_text: str = REFUSAL_MESSAGE
    volatile_body_keys: frozenset = frozenset()


ANSWER_UNKNOWN = "I don't know the answer to that."


VENDOR_RAILS = [
    VendorRail(
        rail_id="activefence_input",
        flow="activefence moderation on input",
        direction="input",
        allow_payload={"violations": [], "errors": []},
        block_payload={
            "violations": [{"violation_type": "abusive_or_harmful.harassment_or_bullying", "risk_score": 0.95}],
            "errors": [],
        },
        env={"ACTIVEFENCE_API_KEY": "test-key"},
        volatile_body_keys=frozenset({"content_id"}),
    ),
    VendorRail(
        rail_id="activefence_output",
        flow="activefence moderation on output",
        direction="output",
        allow_payload={"violations": [], "errors": []},
        block_payload={
            "violations": [{"violation_type": "abusive_or_harmful.harassment_or_bullying", "risk_score": 0.95}],
            "errors": [],
        },
        env={"ACTIVEFENCE_API_KEY": "test-key"},
        volatile_body_keys=frozenset({"content_id"}),
    ),
    VendorRail(
        rail_id="privateai_detect_input",
        flow="detect pii on input",
        direction="input",
        allow_payload=[{"processed_text": "hello there", "entities_present": []}],
        block_payload=[{"processed_text": "hello [NAME_1]", "entities_present": ["NAME"]}],
        rails_config=PRIVATEAI_CONFIG,
        env={"PAI_API_KEY": "test-key"},
        llmrails_block_text=ANSWER_UNKNOWN,
    ),
    VendorRail(
        rail_id="privateai_detect_output",
        flow="detect pii on output",
        direction="output",
        allow_payload=[{"processed_text": "hello there", "entities_present": []}],
        block_payload=[{"processed_text": "hello [NAME_1]", "entities_present": ["NAME"]}],
        rails_config=PRIVATEAI_CONFIG,
        env={"PAI_API_KEY": "test-key"},
        llmrails_block_text=ANSWER_UNKNOWN,
    ),
    VendorRail(
        rail_id="trend_input",
        flow="trend ai guard input",
        direction="input",
        allow_payload={"action": "Allow", "reason": "no policy matched"},
        block_payload={"action": "Block", "reason": "Prompt Attack Detected"},
        rails_config=TREND_CONFIG,
        env={"V1_API_KEY": "test-key"},
    ),
    VendorRail(
        rail_id="trend_output",
        flow="trend ai guard output",
        direction="output",
        allow_payload={"action": "Allow", "reason": "no policy matched"},
        block_payload={"action": "Block", "reason": "Policy violation"},
        rails_config=TREND_CONFIG,
        env={"V1_API_KEY": "test-key"},
    ),
    VendorRail(
        rail_id="ai_defense_prompt",
        flow="ai defense inspect prompt",
        direction="input",
        allow_payload={"is_safe": True, "rules": []},
        block_payload={"is_safe": False, "rules": [{"rule_name": "Prompt Injection"}]},
        rails_config=AI_DEFENSE_CONFIG,
        env=AI_DEFENSE_ENV,
    ),
    VendorRail(
        rail_id="ai_defense_response",
        flow="ai defense inspect response",
        direction="output",
        allow_payload={"is_safe": True, "rules": []},
        block_payload={"is_safe": False, "rules": [{"rule_name": "Harmful Content"}]},
        rails_config=AI_DEFENSE_CONFIG,
        env=AI_DEFENSE_ENV,
    ),
    VendorRail(
        rail_id="f5_input",
        flow="f5 guardrails scan input",
        direction="input",
        allow_payload={"result": {"outcome": "cleared"}},
        block_payload={"result": {"outcome": "blocked"}},
        env=F5_ENV,
    ),
    VendorRail(
        rail_id="f5_output",
        flow="f5 guardrails scan output",
        direction="output",
        allow_payload={"result": {"outcome": "cleared"}},
        block_payload={"result": {"outcome": "blocked"}},
        env=F5_ENV,
    ),
    # Fiddler decides on ``score >= safety_threshold``, so the payloads straddle the
    # configured 0.5 rather than being flags.
    VendorRail(
        rail_id="fiddler_user_safety",
        flow="fiddler user safety",
        direction="input",
        allow_payload={"fdl_harmful": 0.0},
        block_payload={"fdl_harmful": 0.9},
        rails_config=FIDDLER_CONFIG,
        env=FIDDLER_ENV,
    ),
    VendorRail(
        rail_id="fiddler_bot_safety",
        flow="fiddler bot safety",
        direction="output",
        allow_payload={"fdl_harmful": 0.0},
        block_payload={"fdl_harmful": 0.9},
        rails_config=FIDDLER_CONFIG,
        env=FIDDLER_ENV,
    ),
    VendorRail(
        rail_id="gliner_detect_input",
        flow="gliner detect pii on input",
        direction="input",
        allow_payload={"total_entities": 0},
        block_payload={"total_entities": 2},
        rails_config=GLINER_CONFIG,
        env={},
    ),
    VendorRail(
        rail_id="gliner_detect_output",
        flow="gliner detect pii on output",
        direction="output",
        allow_payload={"total_entities": 0},
        block_payload={"total_entities": 2},
        rails_config=GLINER_CONFIG,
        env={},
    ),
    VendorRail(
        rail_id="polygraf_detect_input",
        flow="polygraf detect pii on input",
        direction="input",
        allow_payload={"entities": []},
        block_payload={"entities": [{"entity_type": "Email", "start": 0, "end": 5}]},
        rails_config=POLYGRAF_CONFIG,
        env={"POLYGRAF_API_KEY": "test-key"},
        llmrails_block_text=ANSWER_UNKNOWN,
    ),
    VendorRail(
        rail_id="polygraf_detect_output",
        flow="polygraf detect pii on output",
        direction="output",
        allow_payload={"entities": []},
        block_payload={"entities": [{"entity_type": "Email", "start": 0, "end": 5}]},
        rails_config=POLYGRAF_CONFIG,
        env={"POLYGRAF_API_KEY": "test-key"},
        llmrails_block_text=ANSWER_UNKNOWN,
    ),
    VendorRail(
        rail_id="policyai_input",
        flow="policyai moderation on input",
        direction="input",
        allow_payload={"data": [{"status": "ok", "assessment": "SAFE"}]},
        block_payload={
            "data": [{"status": "ok", "assessment": "UNSAFE", "category": "pii", "severity": 3, "reason": "blocked"}]
        },
        env=POLICYAI_ENV,
    ),
    VendorRail(
        rail_id="policyai_output",
        flow="policyai moderation on output",
        direction="output",
        allow_payload={"data": [{"status": "ok", "assessment": "SAFE"}]},
        block_payload={
            "data": [{"status": "ok", "assessment": "UNSAFE", "category": "pii", "severity": 3, "reason": "blocked"}]
        },
        env=POLICYAI_ENV,
    ),
    # The detailed variant scores each violation against its own threshold rather than one
    # shared cut-off, so the blocking score must clear the rule it names: harassment is 0.8.
    VendorRail(
        rail_id="activefence_input_detailed",
        flow="activefence moderation on input detailed",
        direction="input",
        allow_payload={"violations": [], "errors": []},
        block_payload={
            "violations": [{"violation_type": "abusive_or_harmful.harassment_or_bullying", "risk_score": 0.95}],
            "errors": [],
        },
        env={"ACTIVEFENCE_API_KEY": "test-key"},
        volatile_body_keys=frozenset({"content_id"}),
        # The detailed flow answers per violation category rather than with one refusal, so
        # this text follows from the harassment payload above. It is the widest engine gap in
        # the table: four possible messages on LLMRails against one on IORails.
        llmrails_block_text="I will not engage in any abusive or harmful behavior.",
    ),
    VendorRail(
        rail_id="jailbreak_heuristics",
        flow="jailbreak detection heuristics",
        direction="input",
        allow_payload={"jailbreak": False},
        block_payload={"jailbreak": True},
        rails_config=JAILBREAK_CONFIG,
        env={},
    ),
    VendorRail(
        rail_id="clavata_input",
        flow="clavata check input",
        direction="input",
        allow_payload=clavata_response(labels={}),
        block_payload=clavata_response(labels={"Violence": True}),
        rails_config=CLAVATA_CONFIG,
        env={"CLAVATA_API_KEY": "test-key"},
    ),
    VendorRail(
        rail_id="clavata_output",
        flow="clavata check output",
        direction="output",
        allow_payload=clavata_response(labels={}),
        block_payload=clavata_response(labels={"Violence": True}),
        rails_config=CLAVATA_CONFIG,
        env={"CLAVATA_API_KEY": "test-key"},
    ),
]


@dataclass(frozen=True)
class VendorCase:
    """One rail driven to one verdict."""

    case_id: str
    rail: VendorRail
    payload: Any
    expect_blocked: bool


VENDOR_CASES = [
    VendorCase(
        case_id=f"{rail.rail_id}_{suffix}",
        rail=rail,
        payload=payload,
        expect_blocked=blocked,
    )
    for rail in VENDOR_RAILS
    for suffix, payload, blocked in (
        ("allows", rail.allow_payload, False),
        ("blocks", rail.block_payload, True),
    )
]


def _http_response(payload: Any) -> HTTPResponse:
    """Wrap a vendor payload as the HTTP response its action will parse."""
    return HTTPResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode(),
    )


def _vendor_config(rail: VendorRail) -> dict:
    """Build the single-rail config both engines are given."""
    rails: dict = {rail.direction: {"flows": [rail.flow]}}
    if rail.rails_config:
        rails["config"] = copy.deepcopy(rail.rails_config)
    return {"models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])], "rails": rails}


def _stable_body(body: Any, rail: VendorRail) -> Any:
    """Drop the request-body fields that legitimately differ between two calls.

    ActiveFence stamps a fresh ``content_id`` per request, so comparing bodies whole would
    fail for a reason that says nothing about either engine. Dropping declared keys keeps the
    part that matters — above all the text being checked — under assertion.
    """
    if not rail.volatile_body_keys or not isinstance(body, dict):
        return body
    return {key: value for key, value in body.items() if key not in rail.volatile_body_keys}


def _assistant_content(response: object) -> str:
    """Return the assistant message content from a ``generate_async`` result."""
    assert isinstance(response, dict), f"expected a message dict, got {type(response).__name__}"
    return response["content"]


async def _llmrails_reply(config_dict: dict, client: RecordingHTTPClient) -> str:
    """Run one turn through LLMRails with *client* standing in for the vendor."""
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[MAIN_OUTPUT])
    chat.app.register_action_param("http_client", client)

    response = await chat.app.generate_async(messages=[{"role": "user", "content": USER_INPUT}])
    return _assistant_content(response)


async def _iorails_reply(config_dict: dict, client: RecordingHTTPClient, monkeypatch) -> str:
    """Run one turn through IORails with *client* standing in for the vendor.

    Patching the factory rather than assigning the attribute keeps the real injection path
    under test: ``start()`` installs the client, and a compiled rail reads it per request
    through ``RailDependencies.http_client``.
    """
    monkeypatch.setattr(
        "nemoguardrails.guardrails.engine_registry.create_http_client",
        lambda *args, **kwargs: client,
    )
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        main = iorails.engine_registry._engines["main"]
        assert isinstance(main, ModelEngine)
        main.chat_completion = AsyncMock(return_value=LLMResponse(content=MAIN_OUTPUT))

        response = await iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])
        return _assistant_content(response)


class TestVendorRailsAgreeAcrossEngines:
    """Each HTTP-vendor rail reaches the same verdict on both engines, for the same reply."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", VENDOR_CASES, ids=[case.case_id for case in VENDOR_CASES])
    async def test_engines_reach_the_same_decision(self, case: VendorCase, monkeypatch):
        """One canned vendor reply drives both engines to the same allow-or-block decision.

        The *decision* is what must agree, not the wording. LLMRails renders a block through
        the rail's own Colang flow, so its text varies by rail; IORails emits one refusal for
        every rail. Both texts are asserted below, from the table, so the difference is pinned
        rather than papered over — but neither engine is required to match the other's string.
        """
        for name, value in case.rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _vendor_config(case.rail)

        llmrails_content = await _llmrails_reply(config_dict, RecordingHTTPClient([_http_response(case.payload)]))
        iorails_content = await _iorails_reply(
            config_dict, RecordingHTTPClient([_http_response(case.payload)]), monkeypatch
        )

        if case.expect_blocked:
            assert llmrails_content == case.rail.llmrails_block_text
            assert iorails_content == REFUSAL_MESSAGE
        else:
            assert llmrails_content == MAIN_OUTPUT
            assert iorails_content == MAIN_OUTPUT

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", VENDOR_RAILS, ids=[rail.rail_id for rail in VENDOR_RAILS])
    async def test_both_engines_send_the_vendor_the_same_request(self, rail: VendorRail, monkeypatch):
        """Same turn in, same outbound call: method, URL and body, not merely the same verdict.

        Decision parity alone cannot catch a malformed request — a rail that sends the wrong
        text still returns a verdict, and a clean payload makes that verdict "allow" either
        way. Finding 13 is the precedent: two engines agreed on the decision while sending the
        classifier different conversations.
        """
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _vendor_config(rail)

        llmrails_client = RecordingHTTPClient([_http_response(rail.allow_payload)])
        await _llmrails_reply(config_dict, llmrails_client)
        iorails_client = RecordingHTTPClient([_http_response(rail.allow_payload)])
        await _iorails_reply(config_dict, iorails_client, monkeypatch)

        assert len(iorails_client.requests) == len(llmrails_client.requests) == 1
        llmrails_request, iorails_request = llmrails_client.requests[0], iorails_client.requests[0]
        assert (iorails_request.method, iorails_request.url) == (llmrails_request.method, llmrails_request.url)
        assert _stable_body(iorails_request.json, rail) == _stable_body(llmrails_request.json, rail)


class TestVendorRailsAreReachable:
    """A rail that works is still unreachable until the enabled tier admits it."""

    @pytest.mark.parametrize("rail", VENDOR_RAILS, ids=[rail.rail_id for rail in VENDOR_RAILS])
    def test_iorails_accepts_a_config_using_the_rail(self, rail: VendorRail, monkeypatch):
        """``can_handle`` admits the rail, so a real config routes here rather than to LLMRails.

        Separate from the equivalence cases above deliberately: those construct ``IORails``
        directly and would keep passing while every user's config silently fell back.
        """
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config = RailsConfig.from_content(config=_vendor_config(rail))

        assert IORails.unsupported_reason(config, llm=None) is None
