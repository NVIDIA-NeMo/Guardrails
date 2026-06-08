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

from types import SimpleNamespace
from typing import Any, cast

import pytest

from nemoguardrails.context import llm_response_metadata_var
from nemoguardrails.rails.llm.generation import generation_response
from nemoguardrails.rails.llm.generation.generation_response import (
    GenerationEventMetadata,
    generation_event_metadata,
    generation_response_from_colang_turn,
)
from nemoguardrails.rails.llm.options import GenerationOptions


def _processing_log():
    return [
        {"type": "event", "timestamp": 1.0, "data": {"type": "StartInputRails"}},
        {"type": "event", "timestamp": 2.0, "data": {"type": "InputRailsFinished"}},
    ]


def test_generation_event_metadata_extracts_and_clears_context_metadata():
    token = llm_response_metadata_var.set({"provider": "metadata"})
    try:
        metadata = generation_event_metadata(
            [
                {"type": "BotToolCalls", "tool_calls": [{"id": "call-1"}]},
                {"type": "BotThinking", "content": "reasoning"},
            ]
        )
    finally:
        llm_response_metadata_var.reset(token)

    assert metadata.tool_calls == [{"id": "call-1"}]
    assert metadata.llm_metadata == {"provider": "metadata"}
    assert metadata.reasoning_content == "reasoning"


def test_generation_response_from_colang_1_turn_populates_response_fields_and_log():
    new_events = [{"type": "StartUtteranceBotAction", "script": "Hello"}]

    response = generation_response_from_colang_turn(
        colang_version="1.0",
        prompt=None,
        new_message={"role": "assistant", "content": "Hello"},
        events=[
            {"type": "ContextUpdate", "data": {"answer": 42}},
            {"type": "UserMessage", "text": "Hi"},
            {"type": "StartUtteranceBotAction", "script": "Hello"},
        ],
        new_events=new_events,
        processing_log=_processing_log(),
        gen_options=GenerationOptions(
            output_vars=["answer"],
            log=cast(Any, {"internal_events": True, "colang_history": True}),
        ),
        state={"events": []},
        output_state={"events": ["serialized"]},
        event_metadata=GenerationEventMetadata(
            tool_calls=[{"id": "call-1"}],
            llm_metadata={"provider": "metadata"},
            reasoning_content="reasoning",
        ),
    )

    assert response.response == [{"role": "assistant", "content": "Hello"}]
    assert response.tool_calls == [{"id": "call-1"}]
    assert response.llm_metadata == {"provider": "metadata"}
    assert response.reasoning_content == "reasoning"
    assert response.output_data == {"answer": 42}
    assert response.state == {"events": ["serialized"]}
    assert response.log is not None
    assert response.log.internal_events == new_events
    assert response.log.colang_history is not None
    assert '"Hello"' in response.log.colang_history


def test_generation_response_from_colang_1_turn_returns_prompt_content():
    response = generation_response_from_colang_turn(
        colang_version="1.0",
        prompt="Hi",
        new_message={"role": "assistant", "content": "Hello"},
        events=[],
        new_events=[],
        processing_log=_processing_log(),
        gen_options=GenerationOptions(),
        state=None,
        output_state=None,
        event_metadata=GenerationEventMetadata(
            tool_calls=None,
            llm_metadata=None,
            reasoning_content=None,
        ),
    )

    assert response.response == "Hello"


def test_generation_response_from_colang_1_turn_includes_raw_llm_output(monkeypatch):
    raw_response = {"raw": "completion"}
    fake_generation_log = SimpleNamespace(
        activated_rails=[
            SimpleNamespace(
                type="generation",
                executed_actions=[
                    SimpleNamespace(llm_calls=[SimpleNamespace(raw_response=raw_response)]),
                ],
            )
        ]
    )
    monkeypatch.setattr(
        generation_response,
        "compute_generation_log",
        lambda processing_log: fake_generation_log,
    )

    response = generation_response_from_colang_turn(
        colang_version="1.0",
        prompt=None,
        new_message={"role": "assistant", "content": "Hello"},
        events=[],
        new_events=[],
        processing_log=[],
        gen_options=GenerationOptions(llm_output=True),
        state=None,
        output_state=None,
        event_metadata=GenerationEventMetadata(
            tool_calls=None,
            llm_metadata=None,
            reasoning_content=None,
        ),
    )

    assert response.llm_output == raw_response


@pytest.mark.parametrize(
    ("gen_options", "message"),
    [
        (
            GenerationOptions(output_vars=True),
            "The `output_vars` option is not supported for Colang 2.0 configurations.",
        ),
        (
            GenerationOptions(log=cast(Any, {"internal_events": True})),
            "The `log` option is not supported for Colang 2.0 configurations.",
        ),
        (
            GenerationOptions(llm_output=True),
            "The `llm_output` option is not supported for Colang 2.0 configurations.",
        ),
    ],
)
def test_generation_response_from_colang_2_turn_rejects_unsupported_options(gen_options, message):
    with pytest.raises(ValueError, match=message):
        generation_response_from_colang_turn(
            colang_version="2.x",
            prompt=None,
            new_message={"role": "assistant", "content": "Hello"},
            events=[],
            new_events=[],
            processing_log=[],
            gen_options=gen_options,
            state=None,
            output_state=None,
            event_metadata=GenerationEventMetadata(
                tool_calls=None,
                llm_metadata=None,
                reasoning_content=None,
            ),
        )
