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

import os
from typing import List, Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.utils import Input, Output

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from nemoguardrails.logging.verbose import set_verbose
from tests.utils import FakeLLM


def has_nvidia_ai_endpoints():
    """Check if NVIDIA AI Endpoints package is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_nvidia_ai_endpoints")


def has_openai():
    """Check if OpenAI package is installed."""
    from nemoguardrails.imports import check_optional_dependency

    return check_optional_dependency("langchain_openai")


def test_string_in_string_out():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    assert result == "Paris."


def test_string_in_string_out_with_verbose_flag():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm, verbose=True)
    assert model_with_rails.rails.verbose is True

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    assert result == "Paris."


def test_configurable_passed_to_invoke():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config, llm=llm)

    prompt = PromptTemplate.from_template("The capital of {param1} ")
    chain = prompt | (rails | llm)

    configurable = {"configurable": {"param1": "value1", "param2": "value2"}}
    result = chain.invoke({"param1": "France"}, config=configurable)

    assert result == "Paris."


def test_string_in_string_out_pipe_syntax():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    rails = RunnableRails(config)

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | (rails | llm)

    result = chain.invoke(input={})

    assert result == "Paris."


def test_chat_in_chat_out():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config) | llm

    prompt = ChatPromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    assert isinstance(result, AIMessage)
    assert result.content == "Paris."


def test_dict_string_in_dict_string_out():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    result = model_with_rails.invoke(input={"input": "The capital of France is "})

    assert isinstance(result, dict)
    assert result["output"] == "Paris."


def test_dict_messages_in_dict_messages_out():
    llm = FakeLLM(
        responses=[
            "Paris.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    result = model_with_rails.invoke(input={"input": [{"role": "user", "content": "The capital of France is "}]})

    assert isinstance(result, dict)
    assert result["output"] == {"role": "assistant", "content": "Paris."}


def test_dict_system_message_in_dict_messages_out():
    """Tests that SystemMessage is correctly handled."""
    llm = FakeLLM(
        responses=[
            "Okay.",
        ]
    )
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm)

    original_generate_async = model_with_rails.rails.generate_async
    messages_passed = None

    async def mock_generate_async(*args, **kwargs):
        nonlocal messages_passed
        messages_passed = kwargs.get("messages")
        return await original_generate_async(*args, **kwargs)

    model_with_rails.rails.generate_async = mock_generate_async

    result = model_with_rails.invoke(
        input={
            "input": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Got it?"},
            ]
        }
    )

    assert isinstance(result, dict)
    assert result["output"] == {"role": "assistant", "content": "Okay."}
    assert messages_passed == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Got it?"},
    ]


def test_list_system_message_in_list_messages_out():
    """Tests that SystemMessage is correctly handled when input is ChatPromptValue."""
    llm_response = "Intent: user asks question"
    llm = FakeLLM(responses=[llm_response])

    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config)

    chain = model_with_rails | llm

    input_messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Got it?"),
    ]
    result = chain.invoke(input=ChatPromptValue(messages=input_messages))

    assert isinstance(result, AIMessage)
    assert result.content == llm_response


def test_context_passing():
    llm = FakeLLM(
        responses=[
            "  express greeting",
        ]
    )

    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting

        define bot express greeting
          "Hi, $name!"
    """,
    )
    model_with_rails = RunnableRails(config, llm=llm)

    result = model_with_rails.invoke(
        input={
            "input": [{"role": "user", "content": "Hi"}],
            "context": {"name": "John"},
        }
    )

    assert isinstance(result, dict)
    assert result["output"] == {"role": "assistant", "content": "Hi, John!"}


def test_string_passthrough_mode_off():
    llm = FakeLLM(responses=["Paris."])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm, passthrough=False)

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    info = model_with_rails.rails.explain()
    assert len(info.llm_calls) == 1

    # We check that the prompt was altered
    assert "User:" in info.llm_calls[0].prompt
    assert "Assistant:" in info.llm_calls[0].prompt
    assert result == "Paris."


def test_string_passthrough_mode_on_without_dialog_rails():
    llm = FakeLLM(responses=["Paris."])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm, passthrough=True)

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    info = model_with_rails.rails.explain()
    assert len(info.llm_calls) == 1

    # We check that the prompt was NOT altered
    # TODO: Investigate further why the "Human:" prefix ends up here.
    assert info.llm_calls[0].prompt == "Human: The capital of France is "
    assert result == "Paris."


def test_string_passthrough_mode_on_with_dialog_rails():
    llm = FakeLLM(responses=["  express greeting", "Paris."])
    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting
        """,
    )
    model_with_rails = RunnableRails(config, llm=llm, passthrough=True)

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    info = model_with_rails.rails.explain()
    assert len(info.llm_calls) == 2

    # In passthrough mode with dialog rails, the second call should use the message format
    # since RunnableRails converts StringPromptValue to message list, which gets formatted as "Human: ..."
    assert info.llm_calls[1].prompt == "Human: The capital of France is "
    assert result == "Paris."


def test_string_passthrough_mode_on_with_fn_and_without_dialog_rails():
    llm = FakeLLM(responses=["Paris."])
    config = RailsConfig.from_content(config={"models": []})
    model_with_rails = RunnableRails(config, llm=llm, passthrough=True)

    async def passthrough_fn(context: dict, events: List[dict]):
        return "PARIS."

    model_with_rails.rails.llm_generation_actions.passthrough_fn = passthrough_fn

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    info = model_with_rails.rails.explain()

    assert len(info.llm_calls) == 0
    assert result == "PARIS."


def test_string_passthrough_mode_on_with_fn_and_with_dialog_rails():
    llm = FakeLLM(responses=["  express greeting", "Paris."])
    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
        define user express greeting
          "hi"

        define flow
          user express greeting
          bot express greeting
        """,
    )
    model_with_rails = RunnableRails(config, llm=llm, passthrough=True)

    async def passthrough_fn(context: dict, events: List[dict]):
        return "PARIS."

    model_with_rails.rails.llm_generation_actions.passthrough_fn = passthrough_fn

    prompt = PromptTemplate.from_template("The capital of France is ")
    chain = prompt | model_with_rails

    result = chain.invoke(input={})

    info = model_with_rails.rails.explain()

    assert len(info.llm_calls) == 1
    assert result == "PARIS."


class MockRunnable(Runnable):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return {"output": "PARIS!!"}


def test_string_passthrough_mode_with_chain():
    config = RailsConfig.from_content(config={"models": []})

    runnable_with_rails = RunnableRails(config, passthrough=True, runnable=MockRunnable())

    chain = {"input": RunnablePassthrough()} | runnable_with_rails
    result = chain.invoke("The capital of France is ")
    info = runnable_with_rails.rails.explain()

    assert len(info.llm_calls) == 0
    assert result == {"output": "PARIS!!"}


def test_string_passthrough_mode_with_chain_and_dialog_rails():
    llm = FakeLLM(responses=["  ask general question", "Paris."])
    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
            define user ask general question
              "What is this?"

            define flow
              user ask general question
              bot respond
            """,
    )
    runnable_with_rails = RunnableRails(config, llm=llm, passthrough=True, runnable=MockRunnable())

    chain = {"input": RunnablePassthrough()} | runnable_with_rails
    result = chain.invoke("The capital of France is ")
    info = runnable_with_rails.rails.explain()

    assert len(info.llm_calls) == 1
    assert result == {"output": "PARIS!!"}


def test_string_passthrough_mode_with_chain_and_dialog_rails_2():
    llm = FakeLLM(responses=["  ask off topic question"])
    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
            define user ask general question
              "What is this?"

            define flow
              user ask general question
              bot respond

            define user ask off topic question
              "Can you help me cook something?"

            define flow
              user ask off topic question
              bot refuse to respond

            define bot refuse to respond
              "I'm sorry, I can't help with that."

            """,
    )

    runnable_with_rails = RunnableRails(config, llm=llm, passthrough=True, runnable=MockRunnable())

    chain = {"input": RunnablePassthrough()} | runnable_with_rails

    result = chain.invoke("This is an off topic question")
    info = runnable_with_rails.rails.explain()

    assert len(info.llm_calls) == 1
    assert result == {"output": "I'm sorry, I can't help with that."}


def test_string_passthrough_mode_with_chain_and_dialog_rails_2_pipe_syntax():
    llm = FakeLLM(responses=["  ask off topic question"])
    config = RailsConfig.from_content(
        config={"models": []},
        colang_content="""
            define user ask general question
              "What is this?"

            define flow
              user ask general question
              bot respond

            define user ask off topic question
              "Can you help me cook something?"

            define flow
              user ask off topic question
              bot refuse to respond

            define bot refuse to respond
              "I'm sorry, I can't help with that."

            """,
    )

    rails = RunnableRails(config, llm=llm)
    some_other_chain = MockRunnable()

    chain = {"input": RunnablePassthrough()} | (rails | some_other_chain)

    result = chain.invoke("This is an off topic question")
    info = rails.rails.explain()

    assert len(info.llm_calls) == 1
    assert result == {"output": "I'm sorry, I can't help with that."}


class MockRunnable2(Runnable):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return "PARIS!!"


def test_string_passthrough_mode_with_chain_and_string_output():
    config = RailsConfig.from_content(config={"models": []})
    runnable_with_rails = RunnableRails(config, passthrough=True, runnable=MockRunnable2())

    chain = {"input": RunnablePassthrough()} | runnable_with_rails
    result = chain.invoke("The capital of France is ")
    info = runnable_with_rails.rails.explain()

    assert len(info.llm_calls) == 0
    assert result == "PARIS!!"


def test_string_passthrough_mode_with_chain_and_string_input_and_output():
    config = RailsConfig.from_content(config={"models": []})
    runnable_with_rails = RunnableRails(config, passthrough=True, runnable=MockRunnable2())

    chain = runnable_with_rails
    result = chain.invoke("The capital of France is ")
    info = runnable_with_rails.rails.explain()

    assert len(info.llm_calls) == 0
    assert result == "PARIS!!"


def test_mocked_rag_with_fact_checking():
    set_verbose(True)
    config = RailsConfig.from_content(
        yaml_content="""
        models: []
        rails:
            output:
                flows:
                    - self check facts
        prompts:
        - task: self_check_facts
          content: <<NOT IMPORTANT>>
    """,
        colang_content="""
        define user ask question
          "What is the size?"

        define flow
          user ask question
          $check_facts = True
          bot respond to question
    """,
    )

    class MockRAGChain(Runnable):
        def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
            return "The price is $45."

    def mock_retriever(user_input):
        return "The price is $50"

    llm = FakeLLM(responses=["  ask question"])
    guardrails = RunnableRails(config, llm=llm)

    @action()
    async def self_check_facts(context):
        evidence = context.get("relevant_chunks", [])
        response = context.get("bot_message")

        assert "The price is $50" in evidence
        assert "The price is $45" in response

        return 0.0

    guardrails.rails.register_action(self_check_facts)

    rag_chain = MockRAGChain()
    rag_with_guardrails = {
        "input": RunnablePassthrough(),
        "relevant_chunks": RunnableLambda(mock_retriever),
    } | (guardrails | rag_chain)

    result = rag_with_guardrails.invoke("What is the price?")
    info = guardrails.rails.explain()

    assert len(info.llm_calls) == 1
    assert result == "I'm sorry, I can't respond to that."


@pytest.mark.skipif(
    not has_nvidia_ai_endpoints(),
    reason="langchain-nvidia-ai-endpoints package not installed",
)
def test_runnable_binding_treated_as_llm():
    """Test that RunnableBinding with LLM tools is treated as an LLM, not passthrough_runnable."""
    from langchain_core.tools import tool
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    @tool
    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's sunny in {city}!"

    config = RailsConfig.from_content(config={"models": []})
    guardrails = RunnableRails(config=config, passthrough=True)

    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
    llm_with_tools = llm.bind_tools([get_weather])

    piped = guardrails | llm_with_tools

    assert piped.llm is llm_with_tools
    assert piped.passthrough_runnable is None


@pytest.mark.skipif(not has_openai(), reason="langchain-openai package not installed")
def test_chat_prompt_template_with_runnable_rails_fixed():
    """Test that the fix works correctly."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not set")

    llm = FakeLLM(
        responses=[
            "no",
            "express greeting",
            "no",
            "Welcome to our clinic! I'm so glad you're here.",
            "no",
        ]
    )

    config = RailsConfig.from_path("examples/bots/abc")
    guardrails = RunnableRails(config=config, passthrough=True)

    system_prompt = """
    You are a specialized assistant for handling patient intake.
    """

    patient_intake_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    runnable_without_tools = patient_intake_prompt | (guardrails | llm)
    result = runnable_without_tools.invoke({"messages": [("user", "Hi!")]})

    assert "Welcome" in str(result)


def test_metadata_preservation_integration():
    """Integration test to verify that metadata is preserved through RunnableRails."""
    # Use FakeLLM instead of Mock to avoid registration issues

    from langchain_community.llms.fake import FakeListLLM

    fake_llm = FakeListLLM(responses=["Test response"])

    config = RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models:
          - type: main
            engine: openai
            model: gpt-3.5-turbo
        """,
    )

    runnable_rails = RunnableRails(config, llm=fake_llm, passthrough=True)

    # Mock the rails generate method to return GenerationResponse with metadata
    from unittest.mock import Mock

    mock_generation_response = Mock()
    mock_generation_response.response = "Test response"
    mock_generation_response.output_data = {}
    mock_generation_response.tool_calls = None
    mock_generation_response.llm_metadata = {
        "additional_kwargs": {"test_key": "test_value"},
        "response_metadata": {"model_name": "test-model", "token_usage": {"total": 10}},
        "usage_metadata": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
        "id": "test-id",
    }

    runnable_rails.rails.generate = Mock(return_value=mock_generation_response)

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([("human", "Test")])
    result = runnable_rails.invoke(prompt.format_prompt())

    assert isinstance(result, AIMessage)
    assert result.additional_kwargs == {"test_key": "test_value"}
    assert result.response_metadata["model_name"] == "test-model"
    assert result.usage_metadata["total_tokens"] == 10
    assert result.id == "test-id"
