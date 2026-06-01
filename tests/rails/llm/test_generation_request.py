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

import pytest

from nemoguardrails.rails.llm.generation import generation_request
from nemoguardrails.rails.llm.generation.generation_request import (
    normalize_generation_request,
    prepare_generation_request_for_runtime,
)
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationRailsOptions


def test_generation_request_exports():
    assert generation_request.__all__ == [
        "GenerationRequest",
        "PreparedGenerationRequest",
        "normalize_generation_request",
        "prepare_generation_request_for_runtime",
    ]


def test_normalize_generation_request_requires_prompt_or_messages():
    with pytest.raises(ValueError, match="Either prompt or messages must be provided."):
        normalize_generation_request(
            prompt=None,
            messages=None,
            options=None,
            state=None,
        )


def test_normalize_generation_request_rejects_prompt_and_messages():
    with pytest.raises(ValueError, match="Only one of prompt or messages can be provided."):
        normalize_generation_request(
            prompt="hi",
            messages=[{"role": "user", "content": "hi"}],
            options=None,
            state=None,
        )


def test_normalize_generation_request_converts_prompt_to_user_message():
    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options=None,
        state=None,
    )

    assert request.prompt == "hi"
    assert request.messages == [{"role": "user", "content": "hi"}]
    assert request.options is None
    assert request.state is None


def test_normalize_generation_request_preserves_messages_object():
    messages = [{"role": "user", "content": "hi"}]

    request = normalize_generation_request(
        prompt=None,
        messages=messages,
        options=None,
        state=None,
    )

    assert request.prompt is None
    assert request.messages is messages
    assert request.options is None


def test_normalize_generation_request_accepts_options_dict_when_not_empty():
    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options={"output_vars": ["answer"]},
        state=None,
    )

    assert isinstance(request.options, GenerationOptions)
    assert request.options.output_vars == ["answer"]


def test_normalize_generation_request_preserves_options_object():
    options = GenerationOptions(output_vars=["answer"])

    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options=options,
        state=None,
    )

    assert request.options is options


def test_normalize_generation_request_rejects_empty_options_dict_without_state():
    with pytest.raises(TypeError, match="options must be a dict or GenerationOptions"):
        normalize_generation_request(
            prompt="hi",
            messages=None,
            options={},
            state=None,
        )


def test_normalize_generation_request_state_forces_generation_options():
    state = {"events": []}

    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options=None,
        state=state,
    )

    assert isinstance(request.options, GenerationOptions)
    assert request.state is state


def test_normalize_generation_request_accepts_empty_options_dict_with_state():
    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options={},
        state={"events": []},
    )

    assert isinstance(request.options, GenerationOptions)


def test_normalize_generation_request_deserializes_colang_2_state(monkeypatch):
    deserialized = object()

    def fake_json_to_state(state):
        assert state == {"flow_states": []}
        return deserialized

    monkeypatch.setattr(
        "nemoguardrails.rails.llm.generation.generation_request.json_to_state",
        fake_json_to_state,
    )

    request = normalize_generation_request(
        prompt="hi",
        messages=None,
        options=None,
        state={"version": "2.x", "state": {"flow_states": []}},
    )

    assert request.state is deserialized
    assert isinstance(request.options, GenerationOptions)


def test_prepare_generation_request_for_runtime_preserves_raw_messages_for_context():
    messages = [{"role": "user", "content": "hi"}]

    request = prepare_generation_request_for_runtime(
        prompt=None,
        messages=messages,
        options={"output_vars": ["answer"]},
        state=None,
    )

    assert request.request_messages is messages
    assert request.runtime_messages == [
        {
            "role": "context",
            "content": {"generation_options": GenerationOptions(output_vars=["answer"]).model_dump()},
        },
        {"role": "user", "content": "hi"},
    ]
    assert request.runtime_messages is not messages
    assert request.needs_llm is True


def test_prepare_generation_request_for_runtime_rewrites_assistant_message_when_dialog_disabled():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "draft"},
    ]

    request = prepare_generation_request_for_runtime(
        prompt=None,
        messages=messages,
        options={"rails": {"dialog": False}},
        state=None,
    )

    assert request.request_messages is messages
    assert request.runtime_messages == [
        {
            "role": "context",
            "content": {
                "generation_options": GenerationOptions(rails=GenerationRailsOptions(dialog=False)).model_dump(),
                "bot_message": "draft",
            },
        },
        {"role": "user", "content": "hi"},
    ]
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "draft"},
    ]
    assert request.needs_llm is False
