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

"""Cross-engine decision baseline for the four rails IORails ships today.

Every case runs one config and one model payload through **both** engines and asserts they
produce the same response, content for content. Same input, same verdict, two entirely
separate implementations.

These expectations are recorded against the engine being replaced, so any replacement has to
satisfy a table it did not author.

Complements ``tests/test_runtime_flow_gate_equivalence.py`` rather than duplicating it: that
suite stubs the action and pins how a *flow* reads a ``RailOutcome``, and would stay green
through a rewritten rail. These cases mock at the transport boundary, so the real action,
prompt rendering and parser all execute.
"""

import copy
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import (
    CONTENT_SAFETY_INPUT_PROMPT,
    CONTENT_SAFETY_OUTPUT_PROMPT,
    NEMOGUARDS_CONFIG,
    TOPIC_SAFETY_INPUT_PROMPT,
)
from tests.utils import FakeLLMModel, TestChat

USER_INPUT = "hello there"
MAIN_OUTPUT = "Hello! How can I help?"

SAFE_INPUT_JSON = '{"User Safety": "safe"}'
UNSAFE_INPUT_JSON = '{"User Safety": "unsafe", "Safety Categories": "S1: Violence"}'
SAFE_OUTPUT_JSON = '{"User Safety": "safe", "Response Safety": "safe"}'
UNSAFE_OUTPUT_JSON = '{"User Safety": "safe", "Response Safety": "unsafe", "Safety Categories": "S1: Violence"}'

ON_TOPIC = "on-topic"
OFF_TOPIC = "off-topic"

JAILBREAK_CONFIG = NEMOGUARDS_CONFIG["rails"]["config"]["jailbreak_detection"]
JAILBREAK_URL = JAILBREAK_CONFIG["nim_base_url"].rstrip("/") + JAILBREAK_CONFIG["nim_server_endpoint"]


@dataclass(frozen=True)
class ModelRailCase:
    """One model-backed rail driven to one verdict, expressed as the model's raw reply."""

    case_id: str
    flow: str
    direction: str
    model_type: str
    prompt_task: str
    prompt_content: str
    output_parser: Optional[str]
    rail_reply: str
    expect_blocked: bool


def _model_rail_cases() -> list[ModelRailCase]:
    """Build the safe and blocked case for each of the three model-backed rails."""
    specs = [
        (
            "content_safety_input",
            "content safety check input $model=content_safety",
            "input",
            "content_safety",
            "content_safety_check_input $model=content_safety",
            CONTENT_SAFETY_INPUT_PROMPT,
            "nemoguard_parse_prompt_safety",
            SAFE_INPUT_JSON,
            UNSAFE_INPUT_JSON,
        ),
        (
            "content_safety_output",
            "content safety check output $model=content_safety",
            "output",
            "content_safety",
            "content_safety_check_output $model=content_safety",
            CONTENT_SAFETY_OUTPUT_PROMPT,
            "nemoguard_parse_response_safety",
            SAFE_OUTPUT_JSON,
            UNSAFE_OUTPUT_JSON,
        ),
        (
            "topic_safety_input",
            "topic safety check input $model=topic_control",
            "input",
            "topic_control",
            "topic_safety_check_input $model=topic_control",
            TOPIC_SAFETY_INPUT_PROMPT,
            None,
            ON_TOPIC,
            OFF_TOPIC,
        ),
    ]

    cases = []
    for name, flow, direction, model_type, task, content, parser, allow_reply, block_reply in specs:
        for suffix, reply, blocked in (("allows", allow_reply, False), ("blocks", block_reply, True)):
            cases.append(
                ModelRailCase(
                    case_id=f"{name}_{suffix}",
                    flow=flow,
                    direction=direction,
                    model_type=model_type,
                    prompt_task=task,
                    prompt_content=content,
                    output_parser=parser,
                    rail_reply=reply,
                    expect_blocked=blocked,
                )
            )
    return cases


MODEL_RAIL_CASES = _model_rail_cases()


def _model_rail_config(case: ModelRailCase) -> dict:
    """Build the single-rail config both engines are given for *case*."""
    rail_model = next(model for model in NEMOGUARDS_CONFIG["models"] if model["type"] == case.model_type)
    prompt: dict = {"task": case.prompt_task, "content": case.prompt_content}
    if case.output_parser:
        prompt["output_parser"] = case.output_parser
        prompt["max_tokens"] = 50

    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0]), copy.deepcopy(rail_model)],
        "rails": {case.direction: {"flows": [case.flow]}},
        "prompts": [prompt],
    }


def _jailbreak_config() -> dict:
    """Build the jailbreak-only config both engines are given."""
    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])],
        "rails": {
            "input": {"flows": ["jailbreak detection model"]},
            "config": {"jailbreak_detection": copy.deepcopy(JAILBREAK_CONFIG)},
        },
    }


def _assistant_content(response: object) -> str:
    """Return the assistant message content from a ``generate_async`` result.

    Both engines return a plain dict without ``options``; the assertion pins that so an
    accidental ``GenerationResponse`` fails here rather than downstream.
    """
    assert isinstance(response, dict), f"expected a message dict, got {type(response).__name__}"
    return response["content"]


async def _llmrails_reply(
    config_dict: dict, rail_model: Optional[tuple[str, str]] = None, messages: Optional[list[dict]] = None
) -> str:
    """Run one turn through LLMRails and return the assistant content.

    *rail_model* is a ``(model_type, reply)`` pair, supplied through
    ``registered_action_params["llms"]`` as the Colang runtime does; ``None`` for model-free
    rails. The main model comes from ``TestChat``.
    """
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[MAIN_OUTPUT])
    if rail_model is not None:
        model_type, reply = rail_model
        chat.app.runtime.registered_action_params["llms"] = {model_type: FakeLLMModel(responses=[reply])}

    response = await chat.app.generate_async(messages=messages or [{"role": "user", "content": USER_INPUT}])
    return _assistant_content(response)


async def _iorails_reply(
    config_dict: dict, rail_reply: Optional[str] = None, messages: Optional[list[dict]] = None
) -> str:
    """Run one turn through IORails and return the assistant content."""
    # Mocks each engine's transport, so the whole RailsManager -> CompiledRail -> EngineRegistry
    # chain runs without a network call. Keyed by engine name, as EngineRegistry holds them.
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        for name, engine in iorails.engine_registry._engines.items():
            if not isinstance(engine, ModelEngine):
                continue
            if name == "main":
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=MAIN_OUTPUT))
            elif rail_reply is not None:
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=rail_reply))

        response = await iorails.generate_async(messages=messages or [{"role": "user", "content": USER_INPUT}])
        return _assistant_content(response)


class TestModelBackedRailsAgreeAcrossEngines:
    """The three model-backed rails reach the same decision on both engines."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", MODEL_RAIL_CASES, ids=[case.case_id for case in MODEL_RAIL_CASES])
    async def test_engines_reach_the_same_decision(self, case: ModelRailCase):
        """One model reply drives both engines to the same allow-or-block outcome."""
        config_dict = _model_rail_config(case)

        llmrails_content = await _llmrails_reply(config_dict, (case.model_type, case.rail_reply))
        iorails_content = await _iorails_reply(config_dict, case.rail_reply)

        expected_content = REFUSAL_MESSAGE if case.expect_blocked else MAIN_OUTPUT

        assert llmrails_content == expected_content
        assert iorails_content == expected_content

    @pytest.mark.asyncio
    async def test_refusal_text_matches_across_engines(self):
        """A blocked request renders the same user-visible refusal on both engines.

        The engines reach the string by unrelated routes — a constant in ``iorails.py``, a
        ``bot refuse to respond`` message per rail in ``flows.v1.co`` — and nothing binds the
        roughly ten hand-written copies together. Stated by name here, not just implied above.
        """
        case = next(c for c in MODEL_RAIL_CASES if c.case_id == "content_safety_input_blocks")
        config_dict = _model_rail_config(case)

        llmrails_content = await _llmrails_reply(config_dict, (case.model_type, case.rail_reply))
        iorails_content = await _iorails_reply(config_dict, case.rail_reply)

        assert iorails_content == REFUSAL_MESSAGE
        assert llmrails_content == REFUSAL_MESSAGE


CONVERSATION = [
    {"role": "user", "content": "do you ship to France?"},
    {"role": "assistant", "content": "Yes, we ship worldwide."},
    {"role": "user", "content": USER_INPUT},
]


async def _topic_safety_prompt(run, config_dict: dict, messages: list[dict]) -> list[dict]:
    """Capture the messages a run hands the topic-safety model, below the model call."""
    captured: list[list[dict]] = []

    async def spy(llm, sent, **kwargs):
        captured.append(sent)
        return LLMResponse(content=ON_TOPIC)

    with patch("nemoguardrails.library.topic_safety.actions.llm_call", new=spy):
        await run(config_dict, messages=messages)

    assert len(captured) == 1, f"expected one topic-safety call, got {len(captured)}"
    return captured[0]


class TestTopicSafetyPromptMatchesAcrossEngines:
    """Both engines hand the topic-safety model the same prompt, not merely the same verdict."""

    # Decision parity cannot catch a malformed prompt: a duplicated user turn left both engines
    # allowing, so the tests above stayed green while IORails sent the classifier a different
    # conversation. This compares what each engine actually sends.

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "messages",
        [[{"role": "user", "content": USER_INPUT}], CONVERSATION],
        ids=["single-turn", "multi-turn"],
    )
    async def test_both_engines_send_the_same_messages(self, messages):
        """Same conversation in, same message list to the topic-control model, turn for turn."""
        case = next(c for c in MODEL_RAIL_CASES if c.case_id == "topic_safety_input_allows")
        config_dict = _model_rail_config(case)

        llmrails_prompt = await _topic_safety_prompt(_llmrails_reply, config_dict, messages)
        iorails_prompt = await _topic_safety_prompt(_iorails_reply, config_dict, messages)

        assert iorails_prompt == llmrails_prompt

    @pytest.mark.asyncio
    async def test_the_turn_being_checked_is_sent_once(self):
        """The checked turn appears once: the action appends it, so history must withhold it."""
        case = next(c for c in MODEL_RAIL_CASES if c.case_id == "topic_safety_input_allows")

        prompt = await _topic_safety_prompt(_iorails_reply, _model_rail_config(case), CONVERSATION)

        assert [message["content"] for message in prompt].count(USER_INPUT) == 1


class TestJailbreakAgreesAcrossEngines:
    """Jailbreak detection reaches the same decision on both engines, now over one transport."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("verdict", "expect_blocked"),
        [
            ({"jailbreak": False, "score": 0.05}, False),
            ({"jailbreak": True, "score": 0.95}, True),
        ],
        ids=["jailbreak_model_allows", "jailbreak_model_blocks"],
    )
    async def test_engines_reach_the_same_decision(self, verdict, expect_blocked, httpx_mock):
        """One NIM verdict drives both engines to the same response, exactly."""
        config_dict = _jailbreak_config()

        httpx_mock.add_response(url=JAILBREAK_URL, json=verdict)
        llmrails_content = await _llmrails_reply(config_dict)

        httpx_mock.add_response(url=JAILBREAK_URL, json=verdict)
        iorails_content = await _iorails_reply(config_dict)

        expected_content = REFUSAL_MESSAGE if expect_blocked else MAIN_OUTPUT

        assert llmrails_content == expected_content
        assert iorails_content == expected_content

    @pytest.mark.asyncio
    async def test_both_engines_allow_when_the_endpoint_errors(self, httpx_mock):
        """An unreachable NIM lets the request through on both engines."""
        # Inverts the pre-migration expectation of IORails failing closed. The fail-open posture
        # is deliberate: a NIM outage now stops blocking jailbreaks on both engines.
        config_dict = _jailbreak_config()

        httpx_mock.add_exception(httpx.ConnectError("connection refused"), url=JAILBREAK_URL)
        llmrails_content = await _llmrails_reply(config_dict)

        httpx_mock.add_exception(httpx.ConnectError("connection refused"), url=JAILBREAK_URL, is_reusable=True)
        iorails_content = await _iorails_reply(config_dict)

        assert llmrails_content == MAIN_OUTPUT
        assert iorails_content == MAIN_OUTPUT
