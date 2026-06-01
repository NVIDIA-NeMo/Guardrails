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

import pytest

from nemoguardrails.rails.llm.checks.rails_check import check_messages
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    GenerationLog,
    GenerationResponse,
    RailStatus,
    RailType,
)


class FakeRails:
    def __init__(self, response):
        self.config = SimpleNamespace(colang_version="1.0")
        self.runtime = SimpleNamespace()
        self.response = response
        self.generate_calls = []

    async def generate_async(self, *, messages, options):
        self.generate_calls.append({"messages": messages, "options": options})
        return self.response


class MutatingRails(FakeRails):
    async def generate_async(self, *, messages, options):
        messages[0]["content"]["nested"] = "changed"
        return await super().generate_async(messages=messages, options=options)


@pytest.mark.asyncio
async def test_check_messages_detects_input_rails_and_marks_passed():
    rails = FakeRails(GenerationResponse(response=[{"role": "user", "content": "hello"}]))

    result = await check_messages(rails, [{"role": "user", "content": "hello"}])

    assert result.status == RailStatus.PASSED
    assert result.content == "hello"
    assert rails.generate_calls == [
        {
            "messages": [{"role": "user", "content": "hello"}],
            "options": {
                "rails": ["input"],
                "log": {"activated_rails": True},
            },
        }
    ]


@pytest.mark.asyncio
async def test_check_messages_normalizes_output_only_messages():
    rails = FakeRails(GenerationResponse(response=[{"role": "assistant", "content": "hello"}]))

    result = await check_messages(rails, [{"role": "assistant", "content": "hello"}])

    assert result.status == RailStatus.PASSED
    assert rails.generate_calls[0]["messages"] == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "hello"},
    ]
    assert rails.generate_calls[0]["options"]["rails"] == ["output"]


@pytest.mark.asyncio
async def test_check_messages_honors_explicit_rail_types_and_reports_modified_content():
    rails = FakeRails(GenerationResponse(response=[{"role": "assistant", "content": "updated"}]))
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "original"},
    ]

    result = await check_messages(rails, messages, rail_types=[RailType.OUTPUT])

    assert result.status == RailStatus.MODIFIED
    assert result.content == "updated"
    assert rails.generate_calls[0]["messages"] == messages
    assert rails.generate_calls[0]["options"]["rails"] == ["output"]


@pytest.mark.asyncio
async def test_check_messages_does_not_expose_caller_messages_to_generation_mutation():
    rails = MutatingRails(GenerationResponse(response=[{"role": "user", "content": "hello"}]))
    messages = [
        {"role": "context", "content": {"nested": "original"}},
        {"role": "user", "content": "hello"},
    ]

    result = await check_messages(rails, messages)

    assert result.status == RailStatus.PASSED
    assert messages == [
        {"role": "context", "content": {"nested": "original"}},
        {"role": "user", "content": "hello"},
    ]
    assert rails.generate_calls[0]["messages"][0]["content"] == {"nested": "changed"}


@pytest.mark.asyncio
async def test_check_messages_reports_first_blocking_rail():
    rails = FakeRails(
        GenerationResponse(
            response="blocked",
            log=GenerationLog(
                activated_rails=[
                    ActivatedRail(type="input", name="first rail", stop=False),
                    ActivatedRail(type="output", name="blocking rail", stop=True),
                ]
            ),
        )
    )

    result = await check_messages(rails, [{"role": "user", "content": "hello"}])

    assert result.status == RailStatus.BLOCKED
    assert result.content == "blocked"
    assert result.rail == "blocking rail"


@pytest.mark.asyncio
async def test_check_messages_rejects_unexpected_generation_response_type():
    rails = FakeRails("not a generation response")

    with pytest.raises(RuntimeError, match="Expected GenerationResponse, got str"):
        await check_messages(rails, [{"role": "user", "content": "hello"}])
