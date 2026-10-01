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

"""Integration tests for content safety actions with output parsers.

These tests verify that the modified parser interface (list format instead of tuple format)
works correctly with the actual content safety actions and their iterable unpacking logic.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.guardrails.compiled_rail import RailDependencies, compile_rail
from nemoguardrails.guardrails.iorails import REFUSAL_MESSAGE, IORails
from nemoguardrails.guardrails.model_engine import ModelEngine
from nemoguardrails.library.content_safety.actions import (
    content_safety_check_input,
    content_safety_check_output,
)
from nemoguardrails.llm.output_parsers import (
    is_content_safe,
    nemoguard_parse_prompt_safety,
    nemoguard_parse_response_safety,
    nemotron_reasoning_parse_prompt_safety,
    nemotron_reasoning_parse_response_safety,
)
from nemoguardrails.manifests import RailDirection
from nemoguardrails.types import LLMResponse
from tests.utils import FakeLLMModel, TestChat


def _create_mock_setup(llm_responses, parsed_result):
    mock_llm = FakeLLMModel(responses=llm_responses)
    llms = {"test_model": mock_llm}

    mock_task_manager = MagicMock()

    mock_task_manager.render_task_prompt.return_value = "test prompt"
    mock_task_manager.get_stop_tokens.return_value = []
    mock_task_manager.get_max_tokens.return_value = 3
    mock_task_manager.parse_task_output.return_value = parsed_result

    return llms, mock_task_manager


def _create_input_context(user_message="Hello, how are you?"):
    return {"user_message": user_message}


def _create_output_context(user_message="Hi", bot_message="Hello! How can I help you?"):
    return {"user_message": user_message, "bot_message": bot_message}


class TestContentSafetyParserIntegration:
    """Integration tests using parsers with content safety actions."""

    @pytest.mark.parametrize(
        "response,expected_allowed,expected_violations",
        [
            ("safe", True, []),
            ("unsafe", False, []),
            ("unsafe S1 S8", False, ["S1", "S8"]),
            ("yes", False, []),
            ("no", True, []),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_input_with_is_content_safe_parser(
        self, response, expected_allowed, expected_violations
    ):
        parsed_result = is_content_safe(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_input_context()

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked == (not expected_allowed)
        assert result.metadata["policy_violations"] == expected_violations

    @pytest.mark.asyncio
    async def test_content_safety_input_with_is_content_safe_parser_safe_with_violations(
        self,
    ):
        """Test input action with is_content_safe parser; safe with violations."""
        parsed_result = is_content_safe("safe S1 S8")
        llms, mock_task_manager = _create_mock_setup(["safe S1 S8"], parsed_result)
        context = _create_input_context("Dubious violent content")

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked is False
        assert result.metadata["policy_violations"] == ["S1", "S8"]

    @pytest.mark.parametrize(
        "response,expected_allowed,expected_violations",
        [
            ("safe", True, []),
            ("unsafe violence hate", False, ["violence", "hate"]),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_output_with_is_content_safe_parser(
        self, response, expected_allowed, expected_violations
    ):
        parsed_result = is_content_safe(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_output_context()

        result = await content_safety_check_output(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked == (not expected_allowed)
        assert result.metadata["policy_violations"] == expected_violations

    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemoguard_parser_safe(self):
        """Test input action with real nemoguard_parse_prompt_safety parser - safe response."""
        json_response = '{"User Safety": "safe"}'
        parsed_result = nemoguard_parse_prompt_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_input_context()

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked is False
        assert result.metadata["policy_violations"] == []

    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemoguard_parser_unsafe_with_categories(
        self,
    ):
        """Test input action with real nemoguard_parse_prompt_safety parser - unsafe with categories."""
        json_response = '{"User Safety": "unsafe", "Safety Categories": "S1, S8, S10"}'
        parsed_result = nemoguard_parse_prompt_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_input_context("Potentially harmful content")

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked is True
        assert result.metadata["policy_violations"] == ["S1", "S8", "S10"]

    @pytest.mark.parametrize(
        "json_response,expected_allowed,expected_violations",
        [
            ('{"Response Safety": "safe"}', True, []),
            (
                '{"Response Safety": "unsafe", "Safety Categories": "Violence, Hate Speech"}',
                False,
                ["Violence", "Hate Speech"],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_output_with_nemoguard_response_parser(
        self, json_response, expected_allowed, expected_violations
    ):
        parsed_result = nemoguard_parse_response_safety(json_response)
        llms, mock_task_manager = _create_mock_setup([json_response], parsed_result)
        context = _create_output_context()

        result = await content_safety_check_output(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked == (not expected_allowed)
        assert result.metadata["policy_violations"] == expected_violations

    @pytest.mark.parametrize(
        ("action", "parser", "context"),
        [
            (content_safety_check_input, nemoguard_parse_prompt_safety, _create_input_context("Some content")),
            (content_safety_check_output, nemoguard_parse_response_safety, _create_output_context()),
        ],
        ids=["input", "output"],
    )
    @pytest.mark.asyncio
    async def test_content_safety_action_propagates_parser_error(self, action, parser, context):
        llms, mock_task_manager = _create_mock_setup([""], None)
        mock_task_manager.parse_task_output.side_effect = lambda task, output: parser(output)

        with pytest.raises(ValueError, match="Failed to parse content safety model response"):
            await action(
                llms=llms,
                llm_task_manager=mock_task_manager,
                model_name="test_model",
                context=context,
            )

    @pytest.mark.asyncio
    async def test_content_safety_parser_error_fails_closed_through_compiled_rail(self):
        llms, mock_task_manager = _create_mock_setup([""], None)
        mock_task_manager.parse_task_output.side_effect = lambda task, output: nemoguard_parse_prompt_safety(output)
        dependencies = RailDependencies(llms=llms, llm_task_manager=mock_task_manager, config=MagicMock())

        outcome = await compile_rail(
            "content safety check input $model=test_model", RailDirection.INPUT, dependencies
        ).run([{"role": "user", "content": "Some content"}])

        assert outcome == RailOutcome.failure(
            reason="content safety check input error: Failed to parse content safety model response"
        )


class TestIterableUnpackingIntegration:
    """Test that the iterable unpacking works correctly with various parser outputs."""

    @pytest.mark.parametrize(
        "response,expected_safe,expected_violations",
        [
            ("safe", True, []),
            ("unsafe", False, []),
            ("unsafe S1 S8", False, ["S1", "S8"]),
            ("yes", False, []),
            ("no", True, []),
        ],
    )
    def test_iterable_unpacking_with_is_content_safe_outputs(self, response, expected_safe, expected_violations):
        """Test iterable unpacking directly with is_content_safe parser outputs."""
        result = is_content_safe(response)
        is_safe, *violated_policies = result
        assert is_safe is expected_safe
        assert violated_policies == expected_violations

    @pytest.mark.parametrize(
        "json_response,expected_safe,expected_violations",
        [
            ('{"User Safety": "safe"}', True, []),
            (
                '{"User Safety": "unsafe", "Safety Categories": "S1, S8"}',
                False,
                ["S1", "S8"],
            ),
            ('{"Response Safety": "safe"}', True, []),
            (
                '{"Response Safety": "unsafe", "Safety Categories": "Violence, Hate"}',
                False,
                ["Violence", "Hate"],
            ),
        ],
    )
    def test_iterable_unpacking_with_nemoguard_outputs(self, json_response, expected_safe, expected_violations):
        """Test iterable unpacking directly with real NemoGuard parser outputs."""
        if "User Safety" in json_response:
            result = nemoguard_parse_prompt_safety(json_response)
        else:
            result = nemoguard_parse_response_safety(json_response)

        is_safe, *violated_policies = result
        assert is_safe is expected_safe
        assert violated_policies == expected_violations

    def test_backward_compatibility_check(self):
        """Verify that the new list format is NOT compatible with the old tuple unpacking."""
        # this test documents the breaking change i.e. old tuple unpacking should fail
        result = is_content_safe("unsafe S1 S8")  # returns [False, "S1", "S8"]

        # old tuple unpacking should fail with ValueError
        with pytest.raises(ValueError, match="too many values to unpack"):
            is_safe, violated_policies = result

        # new iterable unpacking should work
        is_safe, *violated_policies = result
        assert is_safe is False
        assert violated_policies == ["S1", "S8"]


class TestReasoningEnabledEndToEnd:
    """End-to-end tests using TestChat and rails.explain() to verify prompt rendering."""

    @pytest.mark.parametrize(
        "reasoning_enabled,expected_token,is_harmful,safety_response,expected_response",
        [
            (True, "/think", False, "Prompt harm: unharmful", "Hello! How can I help you?"),
            (False, "/no_think", False, "Prompt harm: unharmful", "Hello! How can I help you?"),
            (True, "/think", True, "Prompt harm: harmful", "I'm sorry, I can't respond to that."),
            (False, "/no_think", True, "Prompt harm: harmful", "I'm sorry, I can't respond to that."),
        ],
    )
    def test_content_safety_input_renders_reasoning_token(
        self, reasoning_enabled, expected_token, is_harmful, safety_response, expected_response
    ):
        """Test content_safety_check_input renders correct token and handles safe/harmful input."""
        content_safety_llm = FakeLLMModel(responses=[safety_response])

        config = RailsConfig.from_content(
            yaml_content=textwrap.dedent(
                f"""
                models:
                  - type: main
                    engine: openai
                    model: gpt-4o-mini
                  - type: content_safety_reasoning
                    engine: openai
                    model: placeholder

                rails:
                  input:
                    flows:
                      - content safety check input $model=content_safety_reasoning
                  config:
                    content_safety:
                      reasoning:
                        enabled: {str(reasoning_enabled)}

                prompts:
                  - task: content_safety_check_input $model=content_safety_reasoning
                    content: |
                      Check: {{{{ user_input }}}}. {{% if reasoning_enabled %}}/think{{% else %}}/no_think{{% endif %}}
                    output_parser: nemotron_reasoning_parse_prompt_safety
                """
            ),
            colang_content=textwrap.dedent(
                """
                define flow content safety check input $model
                  $result = execute content_safety_check_input(model_name=$model)
                """
            ),
        )

        config.models = [m for m in config.models if m.type == "main"]

        chat = TestChat(
            config,
            llm_completions=["Hello! How can I help you?"],
        )

        chat.app.runtime.registered_action_params["llms"] = {"content_safety_reasoning": content_safety_llm}

        user_message = "Hello"
        response = chat.app.generate(messages=[{"role": "user", "content": user_message}])
        info = chat.app.explain()

        assert len(info.llm_calls) >= 1
        content_safety_prompt = info.llm_calls[0].prompt
        assert expected_token in content_safety_prompt
        assert user_message in content_safety_prompt

        assert content_safety_llm.inference_count == 1

        assert response is not None
        assert response["content"] == expected_response

        if is_harmful:
            assert len(info.llm_calls) == 1
        else:
            assert len(info.llm_calls) >= 2


class TestNemotronReasoningParserIntegration:
    """Integration tests for Nemotron reasoning parsers with content safety actions."""

    @pytest.mark.parametrize(
        "response,expected_allowed",
        [
            ("Prompt harm: unharmful\nResponse Harm: unharmful", True),
            ("Prompt harm: harmful\nResponse Harm: unharmful", False),
            ("<think>reasoning</think>\nPrompt harm: unharmful", True),
            ("<think>reasoning</think>\nPrompt harm: harmful", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_input_with_nemotron_reasoning_parser(self, response, expected_allowed):
        """Test input action with nemotron_reasoning_parse_prompt_safety parser."""
        parsed_result = nemotron_reasoning_parse_prompt_safety(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_input_context()

        result = await content_safety_check_input(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked == (not expected_allowed)
        assert result.metadata["policy_violations"] == []

    @pytest.mark.parametrize(
        "response,expected_allowed",
        [
            ("Prompt harm: unharmful\nResponse Harm: unharmful", True),
            ("Prompt harm: unharmful\nResponse Harm: harmful", False),
            ("<think>reasoning</think>\nResponse Harm: unharmful", True),
            ("<think>reasoning</think>\nResponse Harm: harmful", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_content_safety_output_with_nemotron_reasoning_parser(self, response, expected_allowed):
        """Test output action with nemotron_reasoning_parse_response_safety parser."""
        parsed_result = nemotron_reasoning_parse_response_safety(response)
        llms, mock_task_manager = _create_mock_setup([response], parsed_result)
        context = _create_output_context()

        result = await content_safety_check_output(
            llms=llms,
            llm_task_manager=mock_task_manager,
            model_name="test_model",
            context=context,
        )

        assert result.is_blocked == (not expected_allowed)
        assert result.metadata["policy_violations"] == []


PROMPT_SAFE = "User Safety: safe"
PROMPT_UNSAFE = "User Safety: unsafe\nSafety Categories: Criminal Planning/Confessions"
PROMPT_SAFE_RESPONSE_SAFE = "User Safety: safe\nResponse Safety: safe"
PROMPT_SAFE_RESPONSE_UNSAFE = (
    "User Safety: safe\nResponse Safety: unsafe\nSafety Categories: Violence, Criminal Planning/Confessions"
)
PROMPT_UNSAFE_RESPONSE_SAFE = (
    "User Safety: unsafe\nResponse Safety: safe\nSafety Categories: Criminal Planning/Confessions, Violence"
)
CROSS_ENGINE_USER_INPUT = "hello there"
CROSS_ENGINE_MAIN_OUTPUT = "Hello! How can I help?"

EXAMPLE_CONFIG_PATH = Path(__file__).parent.parent / "examples" / "configs" / "nemotron-3.5-content-safety"

EXAMPLE_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "request_categories": "/categories"}


def _load_example_config() -> RailsConfig:
    """Load the shipped example config, so these tests break when it drifts."""
    return RailsConfig.from_path(str(EXAMPLE_CONFIG_PATH))


def _turns(messages: Sequence[Any]) -> Tuple[Tuple[str, str], ...]:
    """Normalise ChatMessage objects (LLMRails) or wire dicts (IORails) to (role, content) pairs."""
    normalised = []
    for message in messages:
        if isinstance(message, dict):
            normalised.append((message["role"], message["content"]))
        else:
            normalised.append((message.role.value, message.content))
    return tuple(normalised)


class _RecordingFakeLLMModel(FakeLLMModel):
    """FakeLLMModel that also records the messages of every call."""

    def __init__(self, responses: List[str]):
        super().__init__(responses=responses)
        self.recorded_messages: List[Any] = []

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        self.recorded_messages.append(prompt)
        return await super().generate_async(prompt, stop=stop, **kwargs)


@dataclass(frozen=True)
class EngineRun:
    """What one turn through an engine produced, and what the guard model was asked for."""

    content: str
    safety_calls: int
    safety_turns: Tuple[Tuple[Tuple[str, str], ...], ...]
    safety_params: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class CrossEngineCase:
    """One scripted content-safety verdict sequence and the outcome both engines must reach."""

    case_id: str
    safety_replies: Tuple[str, ...]
    expect_blocked: bool
    expected_safety_calls: int


CROSS_ENGINE_CASES = [
    CrossEngineCase(
        case_id="input_safe",
        safety_replies=(PROMPT_SAFE, PROMPT_SAFE_RESPONSE_SAFE),
        expect_blocked=False,
        expected_safety_calls=2,
    ),
    CrossEngineCase(
        case_id="input_unsafe",
        safety_replies=(PROMPT_UNSAFE,),
        expect_blocked=True,
        expected_safety_calls=1,
    ),
    CrossEngineCase(
        case_id="input_safe_output_unsafe",
        safety_replies=(PROMPT_SAFE, PROMPT_SAFE_RESPONSE_UNSAFE),
        expect_blocked=True,
        expected_safety_calls=2,
    ),
    CrossEngineCase(
        case_id="input_unsafe_output_safe",
        safety_replies=(PROMPT_UNSAFE, PROMPT_UNSAFE_RESPONSE_SAFE),
        expect_blocked=True,
        expected_safety_calls=1,
    ),
]

ALL_SAFE_REPLIES = (PROMPT_SAFE, PROMPT_SAFE_RESPONSE_SAFE)


def _assistant_content(response: object) -> str:
    """Return the assistant message content from a generate_async result."""
    assert isinstance(response, dict), f"expected a message dict, got {type(response).__name__}"
    return response["content"]


async def _llmrails_turn(safety_replies: Tuple[str, ...]) -> EngineRun:
    """Run one turn through LLMRails with a scripted content-safety model."""
    chat = TestChat(_load_example_config(), llm_completions=[CROSS_ENGINE_MAIN_OUTPUT])

    safety_llm = _RecordingFakeLLMModel(list(safety_replies))
    chat.app.runtime.registered_action_params["llms"]["content_safety"] = safety_llm

    response = await chat.app.generate_async(messages=[{"role": "user", "content": CROSS_ENGINE_USER_INPUT}])
    return EngineRun(
        content=_assistant_content(response),
        safety_calls=safety_llm.inference_count,
        safety_turns=tuple(_turns(messages) for messages in safety_llm.recorded_messages),
        # The fake replaces the configured model outright, so there is no request body to inspect.
        safety_params=(),
    )


async def _iorails_turn(safety_replies: Tuple[str, ...]) -> EngineRun:
    """Run one turn through IORails with a scripted content-safety model."""
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        iorails = IORails(_load_example_config())

    async with iorails:
        safety_mock = AsyncMock(side_effect=[LLMResponse(content=reply) for reply in safety_replies])
        for name, engine in iorails.engine_registry._engines.items():
            if not isinstance(engine, ModelEngine):
                continue
            if name == "main":
                engine.chat_completion = AsyncMock(return_value=LLMResponse(content=CROSS_ENGINE_MAIN_OUTPUT))
            else:
                engine.chat_completion = safety_mock

        response = await iorails.generate_async(messages=[{"role": "user", "content": CROSS_ENGINE_USER_INPUT}])
        return EngineRun(
            content=_assistant_content(response),
            safety_calls=safety_mock.await_count,
            safety_turns=tuple(_turns(call.args[0]) for call in safety_mock.await_args_list),
            # generate_from_messages merges the model's `parameters` into the per-call kwargs before
            # reaching chat_completion, so the mock observes the request body the engine assembled.
            safety_params=tuple(dict(call.kwargs) for call in safety_mock.await_args_list),
        )


ENGINE_RUNNERS = {"llmrails": _llmrails_turn, "iorails": _iorails_turn}


class TestNemotronContentSafetyAcrossEngines:
    """Both engines reach the same decision from one scripted Nemotron content-safety model."""

    @pytest.mark.parametrize("engine", sorted(ENGINE_RUNNERS), ids=sorted(ENGINE_RUNNERS))
    @pytest.mark.parametrize("case", CROSS_ENGINE_CASES, ids=[case.case_id for case in CROSS_ENGINE_CASES])
    @pytest.mark.asyncio
    async def test_engines_reach_the_same_decision(self, case: CrossEngineCase, engine: str):
        """Test each engine blocks or allows as the scripted verdicts dictate, calling the guard equally often."""
        run = await ENGINE_RUNNERS[engine](case.safety_replies)

        expected_content = REFUSAL_MESSAGE if case.expect_blocked else CROSS_ENGINE_MAIN_OUTPUT
        assert run.content == expected_content
        assert run.safety_calls == case.expected_safety_calls


class TestNemotronContentSafetyExampleConfigWiring:
    """The shipped example config renders the turns and request parameters the model requires."""

    @pytest.mark.parametrize("engine", sorted(ENGINE_RUNNERS), ids=sorted(ENGINE_RUNNERS))
    @pytest.mark.asyncio
    async def test_input_rail_sends_only_the_user_turn(self, engine: str):
        """Test the input rail sends a lone user turn, which is why the model omits Response Safety there."""
        run = await ENGINE_RUNNERS[engine](ALL_SAFE_REPLIES)
        assert run.safety_turns[0] == (("user", CROSS_ENGINE_USER_INPUT),)

    @pytest.mark.parametrize("engine", sorted(ENGINE_RUNNERS), ids=sorted(ENGINE_RUNNERS))
    @pytest.mark.asyncio
    async def test_output_rail_sends_the_user_turn_then_the_assistant_turn(self, engine: str):
        """Test the output rail sends both turns in the order the model's chat template requires."""
        run = await ENGINE_RUNNERS[engine](ALL_SAFE_REPLIES)
        assert run.safety_turns[1] == (
            ("user", CROSS_ENGINE_USER_INPUT),
            ("assistant", CROSS_ENGINE_MAIN_OUTPUT),
        )

    @pytest.mark.asyncio
    async def test_chat_template_kwargs_reach_every_guard_request(self):
        """Test the example config's chat_template_kwargs are merged into every content-safety request."""
        run = await _iorails_turn(ALL_SAFE_REPLIES)

        assert len(run.safety_params) == 2
        for params in run.safety_params:
            assert params["chat_template_kwargs"] == EXAMPLE_CHAT_TEMPLATE_KWARGS
