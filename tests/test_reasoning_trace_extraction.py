# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from nemoguardrails.actions.llm.utils import _store_reasoning_traces
from nemoguardrails.context import reasoning_trace_var


class TestStoreReasoningTracesUnit:
    def test_store_reasoning_traces_with_valid_reasoning_content(self):
        test_reasoning = "Step 1: Analyze the question\nStep 2: Formulate response"

        response = AIMessage(
            content="The answer is 42",
            additional_kwargs={"reasoning_content": test_reasoning},
        )

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_empty_reasoning_content(self):
        response = AIMessage(
            content="Response", additional_kwargs={"reasoning_content": ""}
        )

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_none_reasoning_content(self):
        response = AIMessage(
            content="Response", additional_kwargs={"reasoning_content": None}
        )

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_without_reasoning_content_key(self):
        response = AIMessage(
            content="Response", additional_kwargs={"other_key": "other_value"}
        )

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_empty_additional_kwargs(self):
        response = AIMessage(content="Response", additional_kwargs={})

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_without_additional_kwargs_attribute(self):
        class SimpleResponse:
            def __init__(self, content):
                self.content = content

        response = SimpleResponse("Response")

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_non_dict_additional_kwargs(self):
        class ResponseWithInvalidKwargs:
            def __init__(self):
                self.content = "Response"
                self.additional_kwargs = "not_a_dict"

        response = ResponseWithInvalidKwargs()

        reasoning_trace_var.set(None)
        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_overwrites_previous_trace(self):
        initial_trace = "Initial reasoning"
        new_trace = "New reasoning"

        reasoning_trace_var.set(initial_trace)

        response = AIMessage(
            content="Response", additional_kwargs={"reasoning_content": new_trace}
        )

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == new_trace
        assert stored_trace != initial_trace

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_multiline_content(self):
        multiline_reasoning = """Thought process:
1. First, understand the user's intent
2. Second, check available data
3. Third, formulate a response
4. Finally, validate the response"""

        response = AIMessage(
            content="Response",
            additional_kwargs={"reasoning_content": multiline_reasoning},
        )

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == multiline_reasoning

        reasoning_trace_var.set(None)

    def test_store_reasoning_traces_with_special_characters(self):
        special_reasoning = "Thinking: Let's analyze this <step> with \"quotes\" and 'apostrophes' & symbols!"

        response = AIMessage(
            content="Response",
            additional_kwargs={"reasoning_content": special_reasoning},
        )

        _store_reasoning_traces(response)

        stored_trace = reasoning_trace_var.get()
        assert stored_trace == special_reasoning

        reasoning_trace_var.set(None)


class TestReasoningTraceIntegration:
    @pytest.mark.asyncio
    async def test_llm_call_extracts_reasoning_from_additional_kwargs(self):
        test_reasoning = "Let me think about this carefully..."

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content="The answer is 42",
            additional_kwargs={"reasoning_content": test_reasoning},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "What is the answer?")

        assert result == "The answer is 42"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_handles_missing_reasoning_content(self):
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="Regular response", additional_kwargs={})
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "Hello")

        assert result == "Regular response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_with_message_list_extracts_reasoning(self):
        test_reasoning = "Analyzing the conversation context..."

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content="Here's my response",
            additional_kwargs={"reasoning_content": test_reasoning},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, messages)

        assert result == "Here's my response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_multiple_llm_calls_preserve_separate_reasoning_traces(self):
        first_reasoning = "First analysis"
        second_reasoning = "Second analysis"

        mock_llm = AsyncMock()
        call_count = 0

        async def mock_ainvoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(
                    content="First response",
                    additional_kwargs={"reasoning_content": first_reasoning},
                )
            else:
                return AIMessage(
                    content="Second response",
                    additional_kwargs={"reasoning_content": second_reasoning},
                )

        mock_llm.ainvoke = mock_ainvoke

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result1 = await llm_call(mock_llm, "First query")
        trace1 = reasoning_trace_var.get()

        reasoning_trace_var.set(None)
        result2 = await llm_call(mock_llm, "Second query")
        trace2 = reasoning_trace_var.get()

        assert trace1 == first_reasoning
        assert trace2 == second_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_reasoning_content_with_other_additional_kwargs(self):
        test_reasoning = "Complex reasoning process"

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content="Response",
            additional_kwargs={
                "reasoning_content": test_reasoning,
                "model": "test-model",
                "finish_reason": "stop",
                "other_metadata": {"key": "value"},
            },
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "Query")

        assert result == "Response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_extracts_reasoning_from_think_tags(self):
        test_reasoning = "Let me analyze this step by step"

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content=f"<think>{test_reasoning}</think>The answer is 42",
            additional_kwargs={},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "What is the answer?")

        assert result == "The answer is 42"
        assert "<think>" not in result
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == test_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_prefers_additional_kwargs_over_think_tags(self):
        reasoning_from_kwargs = "This should be used"
        reasoning_from_tags = "This should be ignored"

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content=f"<think>{reasoning_from_tags}</think>Response",
            additional_kwargs={"reasoning_content": reasoning_from_kwargs},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "Query")

        assert result == f"<think>{reasoning_from_tags}</think>Response"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == reasoning_from_kwargs

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_extracts_multiline_reasoning_from_think_tags(self):
        multiline_reasoning = """Step 1: Understand the question
Step 2: Break down the problem
Step 3: Formulate the answer"""

        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content=f"<think>{multiline_reasoning}</think>Final answer",
            additional_kwargs={},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "Question")

        assert result == "Final answer"
        assert "<think>" not in result
        stored_trace = reasoning_trace_var.get()
        assert stored_trace == multiline_reasoning

        reasoning_trace_var.set(None)

    @pytest.mark.asyncio
    async def test_llm_call_handles_incomplete_think_tags(self):
        mock_llm = AsyncMock()
        mock_response = AIMessage(
            content="<think>This is incomplete",
            additional_kwargs={},
        )
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from nemoguardrails.actions.llm.utils import llm_call

        reasoning_trace_var.set(None)
        result = await llm_call(mock_llm, "Query")

        assert result == "<think>This is incomplete"
        stored_trace = reasoning_trace_var.get()
        assert stored_trace is None

        reasoning_trace_var.set(None)
