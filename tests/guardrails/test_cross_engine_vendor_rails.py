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
IORails takes it through the client ``RailsManager`` compiles into the rail. Everything below that
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

from nemoguardrails.guardrails.iorails import INTERNAL_ERROR_MESSAGE, REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.http import HTTPResponse
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType
from nemoguardrails.testing import RecordingHTTPClient
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import NEMOGUARDS_CONFIG

# Clavata's verdict is a nested job/result/report document, so its own test factory builds it
# rather than a hand-copied literal that would drift from the models it is parsed into.
from tests.test_clavata import create_clavata_response as clavata_response
from tests.utils import TestChat

USER_INPUT = "hello there"
MAIN_OUTPUT = "Hello! How can I help?"

# A rejected vendor credential: the one failure every one of these rails can produce, and the
# one an operator most needs to tell apart from a content decision.
VENDOR_ERROR_STATUS = 401
VENDOR_ERROR_BODY = {"message": "Invalid token"}

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
# A *remote* classifier, so the rail needs no ``transformers`` install: the backend is chosen
# by ``engine``, and only ``local`` reaches the in-process pipeline.
HF_CLASSIFIER_CONFIG = {
    "hf_classifier": {
        "toxicity": {
            "engine": "vllm",
            "model": "test/toxicity",
            "base_url": "http://hf.example",
            "threshold": 0.5,
            "blocked_labels": ["toxic"],
        }
    }
}
AUTOALIGN_CONFIG = {"autoalign": {"parameters": {"fact_check_endpoint": "http://autoalign.example/factcheck"}}}
HF_ALLOW = {"data": [{"label": "toxic", "probs": [0.1]}]}
HF_BLOCK = {"data": [{"label": "toxic", "probs": [0.9]}]}
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
    VendorRail(
        rail_id="hf_classifier_input",
        flow="hf classifier check input $classifier=toxicity",
        direction="input",
        allow_payload=HF_ALLOW,
        block_payload=HF_BLOCK,
        rails_config=HF_CLASSIFIER_CONFIG,
        env={},
    ),
    VendorRail(
        rail_id="hf_classifier_output",
        flow="hf classifier check output $classifier=toxicity",
        direction="output",
        allow_payload=HF_ALLOW,
        block_payload=HF_BLOCK,
        rails_config=HF_CLASSIFIER_CONFIG,
        env={},
    ),
    # The threshold is a manifest literal of 0.5, and a *low* score is the failure, so these
    # scores sit either side of it the opposite way round from the risk-scoring vendors.
    VendorRail(
        rail_id="autoalign_factcheck",
        flow="autoalign factcheck output",
        direction="output",
        allow_payload={"all_overall_fact_scores": [0.9]},
        block_payload={"all_overall_fact_scores": [0.1]},
        rails_config=AUTOALIGN_CONFIG,
        env={"AUTOALIGN_API_KEY": "test-key"},
        llmrails_block_text=ANSWER_UNKNOWN,
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


def _http_response(payload: Any, status_code: int = 200) -> HTTPResponse:
    """Wrap a vendor payload as the HTTP response its action will parse."""
    return HTTPResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=json.dumps(payload).encode(),
    )


def _rejecting_client() -> RecordingHTTPClient:
    """A vendor that rejects the call outright, which is what a bad API key looks like."""
    return RecordingHTTPClient([_http_response(VENDOR_ERROR_BODY, VENDOR_ERROR_STATUS)])


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
    under test: ``RailsManager`` calls it while compiling each API-backed rail, so *client*
    arrives through the same route a production client would. One instance is returned for
    every rail, which is what lets a single recorder observe whichever rail this case runs.
    """
    monkeypatch.setattr(
        "nemoguardrails.guardrails.rails_manager.create_http_client",
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


class TestVendorErrorsReadTheSameOnBothEngines:
    """A vendor that answers with an error must not read as a content decision on one engine only."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", VENDOR_RAILS, ids=[rail.rail_id for rail in VENDOR_RAILS])
    async def test_engines_agree_on_whether_a_vendor_error_broke_the_rail(self, rail: VendorRail, monkeypatch):
        """One rejected credential, and both engines agree on whether the rail broke or decided."""
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _vendor_config(rail)

        llmrails_content = await _llmrails_reply(config_dict, _rejecting_client())
        iorails_content = await _iorails_reply(config_dict, _rejecting_client(), monkeypatch)

        if llmrails_content == INTERNAL_ERROR_MESSAGE:
            assert iorails_content == INTERNAL_ERROR_MESSAGE, (
                f"{rail.rail_id} raised on the vendor error, so IORails must report an internal "
                f"error as LLMRails does, not {iorails_content!r}"
            )
        else:
            assert iorails_content != INTERNAL_ERROR_MESSAGE, (
                f"{rail.rail_id} handled the vendor error itself and returned a verdict, so "
                f"IORails must not report it as a rail failure"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rail",
        [rail for rail in VENDOR_RAILS if rail.rail_id.startswith("f5_")],
        ids=lambda rail: rail.rail_id,
    )
    async def test_a_rejected_f5_credential_is_not_the_refusal(self, rail: VendorRail, monkeypatch):
        """The reported case, pinned to a literal so equality above cannot agree on the wrong text.

        nvbug 6620440: an F5 401 reached the caller as the refusal a genuine F5 trip produces,
        so a provider outage was indistinguishable from a safety decision.
        """
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _vendor_config(rail)
        iorails_content = await _iorails_reply(config_dict, _rejecting_client(), monkeypatch)

        assert iorails_content == INTERNAL_ERROR_MESSAGE
        assert iorails_content != REFUSAL_MESSAGE


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


MASKED_INPUT = "hello [NAME_1]"
MASKED_OUTPUT = "Hello [NAME_1]! How can I help?"


@dataclass(frozen=True)
class RewritingVendorRail:
    """One HTTP-backed rail that rewrites, with the vendor reply that makes it do so."""

    rail_id: str
    flow: str
    direction: str
    payload: Any
    rewritten: str
    rails_config: dict
    env: dict


REWRITING_VENDOR_RAILS = [
    RewritingVendorRail(
        rail_id="privateai_mask_input",
        flow="mask pii on input",
        direction="input",
        payload=[{"processed_text": MASKED_INPUT, "entities_present": ["NAME"]}],
        rewritten=MASKED_INPUT,
        rails_config=PRIVATEAI_CONFIG,
        env={"PAI_API_KEY": "test-key"},
    ),
    RewritingVendorRail(
        rail_id="privateai_mask_output",
        flow="mask pii on output",
        direction="output",
        payload=[{"processed_text": MASKED_OUTPUT, "entities_present": ["NAME"]}],
        rewritten=MASKED_OUTPUT,
        rails_config=PRIVATEAI_CONFIG,
        env={"PAI_API_KEY": "test-key"},
    ),
]


def _rewriting_vendor_config(rail: RewritingVendorRail) -> dict:
    """Build the single-rail config both engines are given."""
    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])],
        "rails": {rail.direction: {"flows": [rail.flow]}, "config": copy.deepcopy(rail.rails_config)},
    }


def _vendor_checked_messages(rail: RewritingVendorRail) -> list[dict]:
    """The conversation each direction's rails are asked about."""
    if rail.direction == "input":
        return [{"role": "user", "content": USER_INPUT}]
    return [{"role": "user", "content": USER_INPUT}, {"role": "assistant", "content": MAIN_OUTPUT}]


async def _llmrails_check(config_dict: dict, rail: RewritingVendorRail, client: RecordingHTTPClient):
    """Run ``check_async`` through LLMRails with *client* standing in for the vendor."""
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[MAIN_OUTPUT])
    chat.app.register_action_param("http_client", client)

    return await chat.app.check_async(_vendor_checked_messages(rail), rail_types=[RailType(rail.direction)])


async def _iorails_check(config_dict: dict, rail: RewritingVendorRail, client: RecordingHTTPClient, monkeypatch):
    """Run ``check_async`` through IORails with *client* standing in for the vendor."""
    monkeypatch.setattr(
        "nemoguardrails.guardrails.rails_manager.create_http_client",
        lambda *args, **kwargs: client,
    )
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        return await iorails.check_async(_vendor_checked_messages(rail), rail_types=[RailType(rail.direction)])


class TestRewritingVendorRailsAgreeAcrossEngines:
    """A vendor rail that rewrites produces the same text on both engines, for the same reply."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", REWRITING_VENDOR_RAILS, ids=[rail.rail_id for rail in REWRITING_VENDOR_RAILS])
    async def test_both_engines_report_the_same_rewrite(self, rail: RewritingVendorRail, monkeypatch):
        """``check`` compares the two directly: one canned vendor reply, same status and text."""
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _rewriting_vendor_config(rail)

        llmrails_result = await _llmrails_check(config_dict, rail, RecordingHTTPClient([_http_response(rail.payload)]))
        iorails_result = await _iorails_check(
            config_dict, rail, RecordingHTTPClient([_http_response(rail.payload)]), monkeypatch
        )

        assert llmrails_result.status == RailStatus.MODIFIED
        assert iorails_result.status == RailStatus.MODIFIED
        assert llmrails_result.content == rail.rewritten
        assert iorails_result.content == rail.rewritten

    @pytest.mark.asyncio
    async def test_an_output_rewrite_reaches_the_caller_on_both_engines(self, monkeypatch):
        """Through ``generate``, the masked response is what the caller reads on either engine."""
        rail = next(candidate for candidate in REWRITING_VENDOR_RAILS if candidate.direction == "output")
        for name, value in rail.env.items():
            monkeypatch.setenv(name, value)
        config_dict = _rewriting_vendor_config(rail)

        llmrails_content = await _llmrails_reply(config_dict, RecordingHTTPClient([_http_response(rail.payload)]))
        iorails_content = await _iorails_reply(
            config_dict, RecordingHTTPClient([_http_response(rail.payload)]), monkeypatch
        )

        assert llmrails_content == rail.rewritten
        assert iorails_content == rail.rewritten

    @pytest.mark.parametrize("rail", REWRITING_VENDOR_RAILS, ids=[rail.rail_id for rail in REWRITING_VENDOR_RAILS])
    def test_iorails_accepts_a_config_using_the_rail(self, rail: RewritingVendorRail):
        """``can_handle`` admits the rail, so a real config routes here rather than to LLMRails."""
        config = RailsConfig.from_content(config=_rewriting_vendor_config(rail))

        assert IORails.unsupported_reason(config, llm=None) is None
