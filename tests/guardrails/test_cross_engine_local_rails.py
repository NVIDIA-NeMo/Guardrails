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

"""Cross-engine equivalence for the in-process rails IORails newly runs.

The third seam, after the HTTP client and the model. These rails decide locally, so there is
nothing to mock: the verdict follows from the text itself, and the driver varies the message
rather than a canned reply.

Also the home for what happens when a rail's optional dependency is *absent*, which is
peculiar to this group and is characterized rather than asserted as desirable -- see
``TestARailWhoseDependencyIsMissing``.
"""

import copy
import importlib.util
import os
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest

from nemoguardrails.guardrails.compiled_rail import RailCompilationError, RailDependencies, compile_rail
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.manifests import RailDirection as SurfaceDirection
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType
from nemoguardrails.types import LLMResponse
from tests.guardrails.test_data import NEMOGUARDS_CONFIG
from tests.utils import TestChat

BENIGN_INPUT = "hello there"
BENIGN_OUTPUT = "Hello! How can I help?"
SSN_TEXT = "my ssn is 123-45-6789"
SSN_PATTERN = r"\d{3}-\d{2}-\d{4}"

REGEX_CONFIG = {
    "regex_detection": {
        "input": {"patterns": [SSN_PATTERN]},
        "output": {"patterns": [SSN_PATTERN]},
    }
}


@dataclass(frozen=True)
class LocalRail:
    """One in-process rail, with the text that trips it and the text that does not."""

    rail_id: str
    flow: str
    direction: str
    rails_config: dict


LOCAL_RAILS = [
    LocalRail(rail_id="regex_input", flow="regex check input", direction="input", rails_config=REGEX_CONFIG),
    LocalRail(rail_id="regex_output", flow="regex check output", direction="output", rails_config=REGEX_CONFIG),
]


def _local_config(rail: LocalRail) -> dict:
    """Build the single-rail config both engines are given."""
    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])],
        "rails": {rail.direction: {"flows": [rail.flow]}, "config": copy.deepcopy(rail.rails_config)},
    }


def _texts(rail: LocalRail, blocked: bool) -> tuple[str, str]:
    """Return the (user, bot) texts that drive *rail* to the wanted verdict.

    An input rail is tripped by what the user sends and an output rail by what the model
    answers, so the offending text moves between the two depending on direction.
    """
    if rail.direction == "input":
        return (SSN_TEXT if blocked else BENIGN_INPUT), BENIGN_OUTPUT
    return BENIGN_INPUT, (SSN_TEXT if blocked else BENIGN_OUTPUT)


def _assistant_content(response: object) -> str:
    """Return the assistant message content from a ``generate_async`` result."""
    assert isinstance(response, dict), f"expected a message dict, got {type(response).__name__}"
    return response["content"]


async def _llmrails_reply(config_dict: dict, user_text: str, bot_text: str) -> str:
    """Run one turn through LLMRails, with the main model answering *bot_text*."""
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[bot_text])

    response = await chat.app.generate_async(messages=[{"role": "user", "content": user_text}])
    return _assistant_content(response)


async def _iorails_reply(config_dict: dict, user_text: str, bot_text: str) -> str:
    """Run one turn through IORails, with the main model answering *bot_text*."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        main = iorails.engine_registry._engines["main"]
        assert isinstance(main, ModelEngine)
        main.chat_completion = AsyncMock(return_value=LLMResponse(content=bot_text))

        response = await iorails.generate_async(messages=[{"role": "user", "content": user_text}])
        return _assistant_content(response)


class TestLocalRailsAgreeAcrossEngines:
    """Each in-process rail reaches the same verdict on both engines, for the same text."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", LOCAL_RAILS, ids=[rail.rail_id for rail in LOCAL_RAILS])
    @pytest.mark.parametrize("blocked", [False, True], ids=["allows", "blocks"])
    async def test_engines_reach_the_same_decision(self, rail: LocalRail, blocked: bool):
        """The same message drives both engines to the same allow-or-block decision."""
        config_dict = _local_config(rail)
        user_text, bot_text = _texts(rail, blocked)

        llmrails_content = await _llmrails_reply(config_dict, user_text, bot_text)
        iorails_content = await _iorails_reply(config_dict, user_text, bot_text)

        expected = REFUSAL_MESSAGE if blocked else bot_text
        assert llmrails_content == expected
        assert iorails_content == expected

    @pytest.mark.parametrize("rail", LOCAL_RAILS, ids=[rail.rail_id for rail in LOCAL_RAILS])
    def test_iorails_accepts_a_config_using_the_rail(self, rail: LocalRail):
        """``can_handle`` admits the rail, so a real config routes here rather than to LLMRails."""
        config = RailsConfig.from_content(config=_local_config(rail))

        assert IORails.unsupported_reason(config, llm=None) is None


SENSITIVE_DATA_CONFIG = {"sensitive_data_detection": {"input": {"entities": ["PERSON"]}}}


@pytest.mark.skipif(
    importlib.util.find_spec("presidio_analyzer") is not None,
    reason="asserts the refusal seen when the optional extra is absent",
)
class TestARailWhoseDependencyIsMissing:
    """A rail whose optional extra is not installed is refused when it is compiled.

    Library actions import their optional dependency lazily, inside the function, as
    ``nemoguardrails/AGENTS.md`` requires. So nothing fails until a request arrives, and left
    alone the ImportError would reach the fail-closed envelope and become a *block* --
    indistinguishable, to the caller, from a rail that genuinely tripped, with the real cause
    reaching the log only. Compilation checks the declared distribution instead, so the
    configuration error is reported once, before any request.
    """

    def _config(self) -> RailsConfig:
        return RailsConfig.from_content(
            config={
                "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])],
                "rails": {
                    "input": {"flows": ["detect sensitive data on input"]},
                    "config": copy.deepcopy(SENSITIVE_DATA_CONFIG),
                },
            }
        )

    def test_the_config_is_refused_and_names_what_is_missing(self):
        """The reason names the packages and the extra, so it is actionable without a stack trace."""
        reason = IORails.unsupported_reason(self._config(), llm=None)

        assert reason is not None
        assert "presidio-analyzer" in reason
        assert "'sdd' extra" in reason

    def test_no_request_is_needed_to_discover_it(self):
        """The refusal comes from compilation, so it costs one engine construction, not one per turn."""
        with pytest.raises(RailCompilationError, match="presidio-analyzer"):
            compile_rail(
                "detect sensitive data on input",
                SurfaceDirection.INPUT,
                RailDependencies(llms={"main": None}, llm_task_manager=None, config=None),
            )


SQL_OUTPUT = "This is a SELECT * FROM users; -- malicious comment in text"
SANITIZED_OUTPUT = "This is a  * FROM usersmalicious comment in text"
INJECTION_OMIT_CONFIG = {"injection_detection": {"injections": ["sqli"], "action": "omit"}}

BLOAT_LIMIT = 40
BLOATED_INPUT = "tell me about guardrails and every option they support, at length, in detail"
TRUNCATED_INPUT = BLOATED_INPUT[:BLOAT_LIMIT]
CONTEXT_BLOAT_CONFIG = {"context_bloat_detection": {"max_chars": BLOAT_LIMIT, "action": "truncate"}}


@dataclass(frozen=True)
class RewritingRail:
    """One in-process rail that rewrites, with the text that trips it and what it makes of it."""

    rail_id: str
    flow: str
    direction: str
    rails_config: dict
    original: str
    rewritten: str


REWRITING_RAILS = [
    RewritingRail(
        rail_id="injection_detection_omit",
        flow="injection detection",
        direction="output",
        rails_config=INJECTION_OMIT_CONFIG,
        original=SQL_OUTPUT,
        rewritten=SANITIZED_OUTPUT,
    ),
    RewritingRail(
        rail_id="context_bloat_truncate",
        flow="context bloat detection on input",
        direction="input",
        rails_config=CONTEXT_BLOAT_CONFIG,
        original=BLOATED_INPUT,
        rewritten=TRUNCATED_INPUT,
    ),
]


def _rewriting_config(rail: RewritingRail) -> dict:
    """Build the single-rail config both engines are given."""
    return {
        "models": [copy.deepcopy(NEMOGUARDS_CONFIG["models"][0])],
        "rails": {rail.direction: {"flows": [rail.flow]}, "config": copy.deepcopy(rail.rails_config)},
    }


def _checked_messages(rail: RewritingRail) -> list[dict]:
    """The conversation each direction's rails are asked about, carrying the offending text."""
    if rail.direction == "input":
        return [{"role": "user", "content": rail.original}]
    return [{"role": "user", "content": BENIGN_INPUT}, {"role": "assistant", "content": rail.original}]


async def _llmrails_check(config_dict: dict, rail: RewritingRail):
    """Run ``check_async`` through LLMRails, which needs no main model for a rails-only check."""
    config = RailsConfig.from_content(config=config_dict)
    chat = TestChat(config, llm_completions=[BENIGN_OUTPUT])

    return await chat.app.check_async(_checked_messages(rail), rail_types=[RailType(rail.direction)])


async def _iorails_check(config_dict: dict, rail: RewritingRail):
    """Run ``check_async`` through IORails on the same config and conversation."""
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(RailsConfig.from_content(config=config_dict))

    async with iorails:
        return await iorails.check_async(_checked_messages(rail), rail_types=[RailType(rail.direction)])


class TestRewritingRailsAgreeAcrossEngines:
    """Each rewriting rail produces the same text on both engines, for the same message."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", REWRITING_RAILS, ids=[rail.rail_id for rail in REWRITING_RAILS])
    async def test_both_engines_report_the_same_rewrite(self, rail: RewritingRail):
        """``check`` is where the two engines can be compared directly: same status, same text."""
        config_dict = _rewriting_config(rail)

        llmrails_result = await _llmrails_check(config_dict, rail)
        iorails_result = await _iorails_check(config_dict, rail)

        assert llmrails_result.status == RailStatus.MODIFIED
        assert iorails_result.status == RailStatus.MODIFIED
        assert llmrails_result.content == rail.rewritten
        assert iorails_result.content == rail.rewritten

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rail", REWRITING_RAILS, ids=[rail.rail_id for rail in REWRITING_RAILS])
    async def test_a_rewrite_names_no_rail_on_either_engine(self, rail: RewritingRail):
        """Neither engine treats a rewrite as a rail triggering, so neither names one."""
        config_dict = _rewriting_config(rail)

        assert (await _llmrails_check(config_dict, rail)).rail is None
        assert (await _iorails_check(config_dict, rail)).rail is None

    @pytest.mark.asyncio
    async def test_an_output_rewrite_reaches_the_caller_on_both_engines(self):
        """Through ``generate``, the sanitized response is what the caller reads on either engine."""
        rail = next(candidate for candidate in REWRITING_RAILS if candidate.direction == "output")
        config_dict = _rewriting_config(rail)

        llmrails_content = await _llmrails_reply(config_dict, BENIGN_INPUT, rail.original)
        iorails_content = await _iorails_reply(config_dict, BENIGN_INPUT, rail.original)

        assert llmrails_content == rail.rewritten
        assert iorails_content == rail.rewritten

    @pytest.mark.parametrize("rail", REWRITING_RAILS, ids=[rail.rail_id for rail in REWRITING_RAILS])
    def test_iorails_accepts_a_config_using_the_rail(self, rail: RewritingRail):
        """``can_handle`` admits the rail, so a real config routes here rather than to LLMRails."""
        config = RailsConfig.from_content(config=_rewriting_config(rail))

        assert IORails.unsupported_reason(config, llm=None) is None
