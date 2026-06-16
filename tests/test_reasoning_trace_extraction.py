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

from nemoguardrails.actions.llm.utils import _store_reasoning_traces
from nemoguardrails.context import reasoning_trace_var
from nemoguardrails.types import LLMResponse
from tests.utils import FakeLLMModel


class TestStoreReasoningTracesUnit:
    def test_store_reasoning_traces_with_valid_reasoning_content(self):
        test_reasoning = "Step 1: Analyze the question\nStep 2: Formulate response"

        response = LLMResponse(content="The answer is 42", reasoning=test_reasoning)

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_empty_reasoning_content(self):
        response = LLMResponse(content="Response", reasoning="")

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_none_reasoning_content(self):
        response = LLMResponse(content="Response", reasoning=None)

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_without_reasoning(self):
        response = LLMResponse(content="Response")

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_overwrites_previous_trace(self):
        initial_trace = "Initial reasoning"
        new_trace = "New reasoning"

        reasoning_trace_var.set(initial_trace)

        response = LLMResponse(content="Response", reasoning=new_trace)

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == new_trace
        assert stored_trace != initial_trace

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_clears_previous_when_no_new_reasoning(self):
        previous_trace = "Previous safety check reasoning that should be cleared"
        reasoning_trace_var.set(previous_trace)

        response = LLMResponse(content="Regular response without reasoning")

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None, "Previous reasoning trace should be cleared when new response has no reasoning"

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_multiline_content(self):
        multiline_reasoning = """Thought process:
1. First, understand the user's intent
2. Second, check available data
3. Third, formulate a response
4. Finally, validate the response"""

        response = LLMResponse(content="Response", reasoning=multiline_reasoning)

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == multiline_reasoning

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_special_characters(self):
        special_reasoning = "Thinking: Let's analyze this <step> with \"quotes\" and 'apostrophes' & symbols!"

        response = LLMResponse(content="Response", reasoning=special_reasoning)

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == special_reasoning

        reasoning_trace_var.set(None)


class TestReasoningTraceIntegration:
    @pytest.mark.asyncio
    async def test_llm_call_extracts_reasoning_from_additional_kwargs(self):
        test_reasoning = "Let me think about this carefully..."

        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="The answer is 42", reasoning=test_reasoning)])

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "What is the answer?")

        assert result.content == "The answer is 42"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_handles_missing_reasoning_content(self):
        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="Regular response")])

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "Hello")

        assert result.content == "Regular response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_with_message_list_extracts_reasoning(self):
        test_reasoning = "Analyzing the conversation context..."

        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="Here's my response", reasoning=test_reasoning)])

        from nemoguardrails.actions.llm.utils import llm_call

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, messages)

        assert result.content == "Here's my response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_multiple_llm_calls_preserve_separate_reasoning_traces(self):
        first_reasoning = "First analysis"
        second_reasoning = "Second analysis"

        fake_llm = FakeLLMModel(
            llm_responses=[
                LLMResponse(content="First response", reasoning=first_reasoning),
                LLMResponse(content="Second response", reasoning=second_reasoning),
            ]
        )

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result1 = await llm_call(fake_llm, "First query")
        trace1 = reasoning_trace_var.get()

        reasoning_trace_var.set(None)
        result2 = await llm_call(fake_llm, "Second query")
        trace2 = reasoning_trace_var.get()

        assert trace1 == first_reasoning
        assert trace2 == second_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_reasoning_content_with_other_additional_kwargs(self):
        test_reasoning = "Complex reasoning process"

        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="Response", reasoning=test_reasoning)])

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "Query")

        assert result.content == "Response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_extracts_reasoning_from_think_tags(self):
        test_reasoning = "Let me analyze this step by step"

        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content=f"<think>{test_reasoning}</think>The answer is 42")])

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "What is the answer?")

        assert result.content == "The answer is 42"
        assert "<think>" not in result.content
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_prefers_additional_kwargs_over_think_tags(self):
        reasoning_from_kwargs = "This should be used"
        reasoning_from_tags = "This should be ignored"

        fake_llm = FakeLLMModel(
            llm_responses=[
                LLMResponse(
                    content=f"<think>{reasoning_from_tags}</think>Response",
                    reasoning=reasoning_from_kwargs,
                )
            ]
        )

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "Query")

        assert result.content == f"<think>{reasoning_from_tags}</think>Response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == reasoning_from_kwargs

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_extracts_multiline_reasoning_from_think_tags(self):
        multiline_reasoning = """Step 1: Understand the question
Step 2: Break down the problem
Step 3: Formulate the answer"""

        fake_llm = FakeLLMModel(
            llm_responses=[LLMResponse(content=f"<think>{multiline_reasoning}</think>Final answer")]
        )

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "Question")

        assert result.content == "Final answer"
        assert "<think>" not in result.content
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == multiline_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_handles_incomplete_think_tags(self):
        fake_llm = FakeLLMModel(llm_responses=[LLMResponse(content="<think>This is incomplete")])

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(fake_llm, "Query")

        assert result.content == "<think>This is incomplete"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)
