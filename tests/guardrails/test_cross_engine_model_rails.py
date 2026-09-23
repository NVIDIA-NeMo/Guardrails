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

"""Cross-engine equivalence for the model-backed rails IORails newly runs.

The counterpart to ``test_cross_engine_vendor_rails.py``: same question, different seam. A
vendor rail is pinned by fixing its HTTP reply; these rails reach a *model*, so the driver
fixes the model's completion instead and both engines are given the same one.

Two model-resolution paths are covered, and they are not variations of one thing. Llama Guard
names its model in the manifest as a ``literal`` binding, so nothing in the configured flow
string mentions it -- and since compilation now rejects a rail naming an undeclared model,
the config must declare ``llama_guard`` or the rail will not compile at all. Self check names
no model anywhere and resolves one by task at request time, falling back through the default
task to ``main``.
"""

import copy
import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.utils import FakeLLMModel, TestChat

USER_INPUT = "hello there"
MAIN_OUTPUT = "Hello! How can I help?"

# Llama Guard answers with a verdict word, optionally followed by the policies it violated.
LLAMA_GUARD_SAFE = "safe"
LLAMA_GUARD_UNSAFE = "unsafe\nS1"

# Self check is prompted for a yes/no and read through the ``is_content_safe`` parser, where
# "yes" means the content should be blocked.
SELF_CHECK_ALLOW = "no"
SELF_CHECK_BLOCK = "yes"


@dataclass(frozen=True)
class ModelRail:
    """One model-backed rail and the completions that drive it either way."""

    rail_id: str
    flow: str
    direction: str
    model_type: str
    prompt_task: str
    allow_reply: str
    block_reply: str
    output_parser: Optional[str] = None


MODEL_RAILS = [
    ModelRail(
        rail_id="llama_guard_input",
        flow="llama guard check input",
        direction="input",
        model_type="llama_guard",
        prompt_task="llama_guard_check_input",
        allow_reply=LLAMA_GUARD_SAFE,
        block_reply=LLAMA_GUARD_UNSAFE,
    ),
    ModelRail(
        rail_id="llama_guard_output",
        flow="llama guard check output",
        direction="output",
        model_type="llama_guard",
        prompt_task="llama_guard_check_output",
        allow_reply=LLAMA_GUARD_SAFE,
        block_reply=LLAMA_GUARD_UNSAFE,
    ),
    ModelRail(
        rail_id="self_check_input",
        flow="self check input",
        direction="input",
        model_type="self_check_input",
        prompt_task="self_check_input",
        allow_reply=SELF_CHECK_ALLOW,
        block_reply=SELF_CHECK_BLOCK,
        output_parser="is_content_safe",
    ),
    ModelRail(
        rail_id="self_check_output",
        flow="self check output",
        direction="output",
        model_type="self_check_output",
        prompt_task="self_check_output",
        allow_reply=SELF_CHECK_ALLOW,
        block_reply=SELF_CHECK_BLOCK,
        output_parser="is_content_safe",
    ),
]


@dataclass(frozen=True)
class ModelCase:
    """One rail driven to one verdict."""

    case_id: str
    rail: ModelRail
    reply: str
    expect_blocked: bool


MODEL_CASES = [
    ModelCase(case_id=f"{rail.rail_id}_{suffix}", rail=rail, reply=reply, expect_blocked=blocked)
    for rail in MODEL_RAILS
    for suffix, reply, blocked in (("allows", rail.allow_reply, False), ("blocks", rail.block_reply, True))
]


def _model_config(rail: ModelRail) -> dict:
    """Build the single-rail config both engines are given.

    The rail's model is declared as its own type rather than sharing ``main``, so the rail's
    completion and the main generation cannot be served out of one queue in an order the test
    would depend on.
    """
    rail_model = {"type": rail.model_type, "engine": "nim", "model": "meta/llama-3.1-8b-instruct"}
    prompt: dict = {"task": rail.prompt_task, "content": "Check the input: {{ user_input }}\nAnswer [yes/no]:"}
    if rail.output_parser:
        prompt["output_parser"] = rail.output_parser

    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0]), rail_model],
        "rails": {rail.direction: {"flows": [rail.flow]}},
        "prompts": [prompt],
    }


def _assistant_content(response: object) -> str:
    """Return the assistant message content from a ``generate_async`` result."""
    assert isinstance(response, dict), f"expected a message dict, got {type(response).__name__}"
    return response["content"]


async def _llmrails_reply(config_dict: dict, rail: ModelRail, reply: str) -> str:
    """Run one turn through LLMRails with the rail's model answering *reply*.

    The rail model arrives through ``registered_action_params["llms"]``, which is how the
    Colang runtime supplies it; the main model comes from ``TestChat``.
    """
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[MAIN_OUTPUT])
    chat.app.runtime.registered_action_params["llms"] = {rail.model_type: FakeLLMModel(responses=[reply])}

    response = await chat.app.generate_async(messages=[{"role": "user", "content": USER_INPUT}])
    return _assistant_content(response)


async def _iorails_reply(config_dict: dict, rail: ModelRail, reply: str) -> str:
    """Run one turn through IORails with the rail's model answering *reply*."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        for name, engine in iorails.engine_registry._engines.items():
            if not isinstance(engine, ModelEngine):
                continue
            content = MAIN_OUTPUT if name == "main" else reply
            engine.chat_completion = AsyncMock(return_value=LLMResponse(content=content))

        response = await iorails.generate_async(messages=[{"role": "user", "content": USER_INPUT}])
        return _assistant_content(response)


class TestModelRailsAgreeAcrossEngines:
    """Each model-backed rail reaches the same verdict on both engines, for the same completion."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", MODEL_CASES, ids=[case.case_id for case in MODEL_CASES])
    async def test_engines_reach_the_same_decision(self, case: ModelCase):
        """One canned model completion drives both engines to the same allow-or-block decision."""
        config_dict = _model_config(case.rail)

        llmrails_content = await _llmrails_reply(config_dict, case.rail, case.reply)
        iorails_content = await _iorails_reply(config_dict, case.rail, case.reply)

        if case.expect_blocked:
            assert llmrails_content == REFUSAL_MESSAGE
            assert iorails_content == REFUSAL_MESSAGE
        else:
            assert llmrails_content == MAIN_OUTPUT
            assert iorails_content == MAIN_OUTPUT


class TestModelRailsAreReachable:
    """A model-backed rail is only reachable if the enabled tier admits it and its model exists."""

    @pytest.mark.parametrize("rail", MODEL_RAILS, ids=[rail.rail_id for rail in MODEL_RAILS])
    def test_iorails_accepts_a_config_using_the_rail(self, rail: ModelRail):
        """``can_handle`` admits the rail, so a real config routes here rather than to LLMRails."""
        config = RailsConfig.from_content(config=_model_config(rail))

        assert IORails.unsupported_reason(config, llm=None) is None

    def test_llama_guard_without_its_model_routes_to_llmrails(self):
        """Dropping the ``llama_guard`` model makes the rail unservable rather than failing per request.

        The model is named by a manifest literal, so nothing in the flow string reveals the
        dependency and ``RailsConfig``'s ``$model=`` validation cannot see it. Without the
        compile-time check this config would be accepted and then fail inside the action on
        every request, which the fail-closed envelope would report as a guardrail block.
        """
        config_dict = _model_config(MODEL_RAILS[0])
        config_dict["models"] = [model for model in config_dict["models"] if model["type"] != "llama_guard"]

        reason = IORails.unsupported_reason(RailsConfig.from_content(config=config_dict), llm=None)

        assert reason is not None
        assert "llama_guard" in reason
