# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import itertools
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
    GenerationLogOptions,
    GenerationOptions,
    GenerationRailsOptions,
    GenerationResponse,
)
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing.tracer import Tracer, new_uuid
from tests.utils import TestChat


class TestTracer(unittest.TestCase):
    def test_new_uuid(self):
        uuid_str = new_uuid()
        self.assertIsInstance(uuid_str, str)
        self.assertEqual(len(uuid_str), 36)  # UUID length

    def test_tracer_initialization(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=GenerationLog())
        tracer = Tracer(input=input_data, response=response)
        self.assertEqual(tracer._interaction_output.input, "test input")
        self.assertEqual(tracer._interaction_output.output, "test response")
        self.assertEqual(tracer._generation_log, response.log)

    def test_tracer_initialization_missing_log(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=None)
        with self.assertRaises(RuntimeError):
            Tracer(input=input_data, response=response)

    def test_generate_interaction_log(self):
        input_data = [{"content": "test input"}]

        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=[],
                executed_actions=[],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer = Tracer(input=input_data, response=response)
        interaction_log = tracer.generate_interaction_log()
        self.assertIsNotNone(interaction_log)

    def test_add_adapter(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=GenerationLog())
        tracer = Tracer(input=input_data, response=response)
        adapter = MagicMock(spec=InteractionLogAdapter)
        tracer.add_adapter(adapter)
        self.assertIn(adapter, tracer.adapters)

    def test_export(self):
        input_data = [{"content": "test input"}]

        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=["dummy_decision"],
                executed_actions=[
                    ExecutedAction(
                        action_name="dummy_action",
                        action_params={},
                        return_value=None,
                        llm_calls=[
                            LLMCallInfo(
                                task="dummy_task",
                                duration=1.0,
                                total_tokens=10,
                                prompt_tokens=5,
                                completion_tokens=5,
                                started_at=0.0,
                                finished_at=1.0,
                                prompt="dummy_prompt",
                                completion="dummy_completion",
                                raw_response={
                                    "token_usage": {
                                        "total_tokens": 10,
                                        "completion_tokens": 5,
                                        "prompt_tokens": 5,
                                    },
                                    "model_name": "dummy_model",
                                },
                                llm_model_name="dummy_model",
                            )
                        ],
                        started_at=0.0,
                        finished_at=1.0,
                        duration=1.0,
                    )
                ],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response_non_empty = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer_non_empty = Tracer(input=input_data, response=response_non_empty)
        adapter_non_empty = MagicMock(spec=InteractionLogAdapter)
        tracer_non_empty.add_adapter(adapter_non_empty)
        tracer_non_empty.export()
        adapter_non_empty.transform.assert_called_once()

    def test_export_async(self):
        input_data = [{"content": "test input"}]
        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=["dummy_decision"],
                executed_actions=[
                    ExecutedAction(
                        action_name="dummy_action",
                        action_params={},
                        return_value=None,
                        llm_calls=[
                            LLMCallInfo(
                                task="dummy_task",
                                duration=1.0,
                                total_tokens=10,
                                prompt_tokens=5,
                                completion_tokens=5,
                                started_at=0.0,
                                finished_at=1.0,
                                prompt="dummy_prompt",
                                completion="dummy_completion",
                                raw_response={
                                    "token_usage": {
                                        "total_tokens": 10,
                                        "completion_tokens": 5,
                                        "prompt_tokens": 5,
                                    },
                                    "model_name": "dummy_model",
                                },
                                llm_model_name="dummy_model",
                            )
                        ],
                        started_at=0.0,
                        finished_at=1.0,
                        duration=1.0,
                    )
                ],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response_non_empty = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer_non_empty = Tracer(input=input_data, response=response_non_empty)
        adapter_non_empty = AsyncMock(spec=InteractionLogAdapter)
        adapter_non_empty.__aenter__ = AsyncMock(return_value=adapter_non_empty)
        adapter_non_empty.__aexit__ = AsyncMock(return_value=None)
        tracer_non_empty.add_adapter(adapter_non_empty)

        asyncio.run(tracer_non_empty.export_async())
        adapter_non_empty.transform_async.assert_called_once()


@patch.object(Tracer, "export_async", return_value="")
@pytest.mark.asyncio
async def test_tracing_enable_no_crash_issue_1093(mockTracer):
    config = RailsConfig.from_content(
        colang_content="""
    define user express greeting
        "hello"

    define flow
        user express greeting
        bot express greeting

    define bot express greeting
        "Hello World!\\n NewLine World!"
    """,
        config={
            "models": [],
            "rails": {"dialog": {"user_messages": {"embeddings_only": True}}},
        },
    )
    # Force Tracing to be enabled
    config.tracing.enabled = True
    rails = LLMRails(config)
    res = await rails.generate_async(
        messages=[
            {"role": "user", "content": "hi!"},
            {"role": "assistant", "content": "hi!"},
            {"role": "user", "content": "hi!"},
        ]
    )
    assert mockTracer.called == True
    assert res.response != None


@pytest.mark.asyncio
async def test_tracing_does_not_mutate_user_options():
    """Test that tracing doesn't modify the user's original GenerationOptions object.

    This test verifies the core fix: when tracing is enabled, the user's options
    should not be modified. Before the fix, this test would have failed
    because the original options object was being mutated.
    """

    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! How can I assist you today?"
        """,
        config={
            "models": [],
            "tracing": {"enabled": True, "adapters": [{"name": "FileSystem"}]},
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    user_options = GenerationOptions(
        log=GenerationLogOptions(
            activated_rails=False,
            llm_calls=False,
            internal_events=False,
            colang_history=False,
        )
    )

    original_activated_rails = user_options.log.activated_rails
    original_llm_calls = user_options.log.llm_calls
    original_internal_events = user_options.log.internal_events
    original_colang_history = user_options.log.colang_history

    # mock file operations to focus on the mutation issue
    with patch.object(Tracer, "export_async", return_value=None):
        response = await chat.app.generate_async(
            messages=[{"role": "user", "content": "hello"}], options=user_options
        )

        # main fix: no mutation
        assert (
            user_options.log.activated_rails == original_activated_rails
        ), "User's original options were modified! This causes instability."
        assert (
            user_options.log.llm_calls == original_llm_calls
        ), "User's original options were modified! This causes instability."
        assert (
            user_options.log.internal_events == original_internal_events
        ), "User's original options were modified! This causes instability."
        assert (
            user_options.log.colang_history == original_colang_history
        ), "User's original options were modified! This causes instability."

        # verify that tracing still works
        assert (
            response.log is None
        ), "Tracing should still work correctly, without affecting returned log"


@pytest.mark.asyncio
async def test_tracing_with_none_options():
    """Test that tracing works correctly when no options are provided.

    This verifies that the fix doesn't break the case where users don't
    provide any options at all.
    """

    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! How can I assist you today?"
        """,
        config={
            "models": [],
            "tracing": {"enabled": True, "adapters": [{"name": "FileSystem"}]},
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    with patch.object(Tracer, "export_async", return_value=None):
        response = await chat.app.generate_async(
            messages=[{"role": "user", "content": "hello"}], options=None
        )

        assert response.log is None


@pytest.mark.asyncio
async def test_tracing_aggressive_override_when_all_disabled():
    """Test that tracing aggressively enables all logging when user disables all options.

    When user disables all three tracing related options, tracing still enables
    ALL of them to ensure comprehensive logging data. However, this should not contaminate the
    returned response object
    """

    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! How can I assist you today?"
        """,
        config={
            "models": [],
            "tracing": {"enabled": True, "adapters": [{"name": "FileSystem"}]},
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    # user explicitly disables ALL tracing related options
    user_options = GenerationOptions(
        log=GenerationLogOptions(
            activated_rails=False,
            llm_calls=False,
            internal_events=False,
            colang_history=True,
        )
    )

    original_activated_rails = user_options.log.activated_rails
    original_llm_calls = user_options.log.llm_calls
    original_internal_events = user_options.log.internal_events
    original_colang_history = user_options.log.colang_history

    with patch.object(Tracer, "export_async", return_value=None):
        response = await chat.app.generate_async(
            messages=[{"role": "user", "content": "hello"}], options=user_options
        )

        assert user_options.log.activated_rails == original_activated_rails
        assert user_options.log.llm_calls == original_llm_calls
        assert user_options.log.internal_events == original_internal_events
        assert user_options.log.colang_history == original_colang_history

        assert response.log is not None
        assert response.log.activated_rails == []
        assert response.log.llm_calls == []
        assert response.log.internal_events == []

        assert user_options.log.activated_rails == original_activated_rails
        assert user_options.log.llm_calls == original_llm_calls
        assert user_options.log.internal_events == original_internal_events
        assert user_options.log.activated_rails == False
        assert user_options.log.llm_calls == False
        assert user_options.log.internal_events == False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "activated_rails,llm_calls,internal_events,colang_history",
    list(itertools.product([False, True], repeat=4)),
)
async def test_tracing_preserves_specific_log_fields(
    activated_rails, llm_calls, internal_events, colang_history
):
    """Test that adding tracing respects the original user logging options in the response object"""

    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! How can I assist you today?"
        """,
        config={
            "models": [],
            "tracing": {"enabled": True, "adapters": [{"name": "FileSystem"}]},
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    # user enables some subset of log options
    user_options = GenerationOptions(
        log=GenerationLogOptions(
            activated_rails=activated_rails,
            llm_calls=llm_calls,
            internal_events=internal_events,
            colang_history=colang_history,
        )
    )

    original_activated_rails = user_options.log.activated_rails
    original_llm_calls = user_options.log.llm_calls
    original_internal_events = user_options.log.internal_events
    original_colang_history = user_options.log.colang_history

    with patch.object(Tracer, "export_async", return_value=None):
        response = await chat.app.generate_async(
            messages=[{"role": "user", "content": "hello"}], options=user_options
        )

        assert user_options.log.activated_rails == original_activated_rails
        assert user_options.log.llm_calls == original_llm_calls
        assert user_options.log.internal_events == original_internal_events
        assert user_options.log.colang_history == original_colang_history

        # verify that only the requested log options are returned in the response
        if not any(
            (
                user_options.log.activated_rails,
                user_options.log.llm_calls,
                user_options.log.internal_events,
                user_options.log.colang_history,
            )
        ):
            assert response.log is None
        else:
            assert response.log is not None

            if user_options.log.activated_rails:
                assert len(response.log.activated_rails) > 0
            else:
                assert len(response.log.activated_rails) == 0

            if user_options.log.llm_calls:
                assert len(response.log.llm_calls) > 0
            else:
                assert len(response.log.llm_calls) == 0

            if user_options.log.internal_events:
                assert len(response.log.internal_events) > 0
            else:
                assert len(response.log.internal_events) == 0

        assert user_options.log.activated_rails == original_activated_rails
        assert user_options.log.llm_calls == original_llm_calls
        assert user_options.log.internal_events == original_internal_events
        assert user_options.log.activated_rails == activated_rails
        assert user_options.log.llm_calls == llm_calls
        assert user_options.log.internal_events == internal_events


@pytest.mark.asyncio
async def test_tracing_aggressive_override_with_dict_options():
    """Test that tracing works correctly when options are passed as a dict.

    This tests that the fix handles both GenerationOptions objects and dicts,
    since the method signature allows both types.
    """

    config = RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! How can I assist you today?"
        """,
        config={
            "models": [],
            "tracing": {"enabled": True, "adapters": [{"name": "FileSystem"}]},
        },
    )

    chat = TestChat(
        config,
        llm_completions=[
            "user express greeting",
            "bot express greeting",
            "Hello! How can I assist you today?",
        ],
    )

    # user passes options as a dict with all tracing options disabled
    user_options_dict = {
        "log": {
            "activated_rails": False,
            "llm_calls": False,
            "internal_events": False,
            "colang_history": True,
        }
    }

    original_dict = {
        "log": {
            "activated_rails": False,
            "llm_calls": False,
            "internal_events": False,
            "colang_history": True,
        }
    }

    with patch.object(Tracer, "export_async", return_value=None):
        response = await chat.app.generate_async(
            messages=[{"role": "user", "content": "hello"}], options=user_options_dict
        )

        assert user_options_dict == original_dict

        assert response.log is not None
        assert (
            response.log.activated_rails == []
            and len(response.log.activated_rails) == 0
        )
        assert response.log.llm_calls == []
        assert response.log.internal_events == []


if __name__ == "__main__":
    unittest.main()
