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

"""
Tests for the RunnableRails pipe operator and passthrough behavior.
These tests specifically address the issues reported with complex chains.
"""

import pytest
from langchain_core.runnables import RunnableLambda

from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from tests.utils import FakeLLM


def test_basic_piping_compatibility():
    """Test that RunnableRails can be used in a pipe chain."""
    llm = FakeLLM(responses=["Response from LLM"])

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(config, llm=llm, input_key="query", output_key="result")

    chain = {"query": lambda x: x} | guardrails

    result = chain.invoke("Hello")
    assert "Response from LLM" in str(result)


def test_custom_keys_with_pipe_syntax():
    """Test that custom input/output keys work with pipe syntax."""
    llm = FakeLLM(responses=["Response from LLM"])

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(
        config,
        llm=llm,
        input_key="custom_input",
        output_key="custom_output",
    )

    chain = {"custom_input": lambda x: x} | guardrails

    result = chain.invoke("Hello")

    assert isinstance(result, dict)
    assert "custom_output" in result
    assert "Response from LLM" in str(result["custom_output"])


def test_operator_associativity():
    """Test that the pipe operator works correctly in complex chains."""
    llm = FakeLLM(responses=["Response from LLM"])

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(
        config, llm=llm, input_key="custom_input", output_key="custom_output"
    )

    # test associativity: (A | B) | C should be equivalent to A | (B | C)
    chain1 = ({"custom_input": lambda x: x} | guardrails) | RunnableLambda(
        lambda x: f"Processed: {x}"
    )
    chain2 = {"custom_input": lambda x: x} | (
        guardrails | RunnableLambda(lambda x: f"Processed: {x}")
    )

    result1 = chain1.invoke("Hello")
    result2 = chain2.invoke("Hello")

    assert "Processed" in str(result1)
    assert "Processed" in str(result2)


def test_user_reported_chain_pattern():
    """Test the specific chain pattern reported by the user."""
    llm = FakeLLM(
        responses=[
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "Paris is the capital of France.",
            "Paris is the capital of France.",
        ]
    )

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(
        config, llm=llm, input_key="question", output_key="response"
    )

    chain = RunnableLambda(lambda x: {"question": x}) | guardrails

    result = chain.invoke("What is Paris?")

    # as we set output_key="response", the output should have this key
    assert isinstance(result, dict)
    assert "response" in result

    chain_with_parentheses = RunnableLambda(lambda x: {"question": x}) | (
        guardrails | llm
    )
    result2 = chain_with_parentheses.invoke("What is Paris?")

    assert result2 is not None
