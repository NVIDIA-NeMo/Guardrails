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

import pytest
from aioresponses import aioresponses

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.library.patronusai.actions import (
    check_guardrail_pass,
    patronus_evaluate_request,
)
from tests.utils import TestChat

PATRONUS_EVALUATE_API_URL = "https://api.patronus.ai/v1/evaluate"
COLANG_CONFIG = """
define user express greeting
  "hi"
define bot refuse to respond
  "I'm sorry, I can't respond to that."
"""

YAML_PREFIX = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct
rails:
  output:
    flows:
      - patronus api check output
"""


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


@pytest.mark.asyncio
def test_patronus_evaluate_api_success_strategy_all_pass(monkeypatch):
    """
    Test that the "all_pass" success strategy passes when all evaluators pass
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "all_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_evaluate_api_success_strategy_all_pass_fails_when_one_failure(
    monkeypatch,
):
    """
    Test that the "all_pass" success strategy fails when only one evaluator fails
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "all_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


def test_patronus_evaluate_api_success_strategy_any_pass_passes_when_one_failure(
    monkeypatch,
):
    """
    Test that the "any_pass" success strategy passes when only one evaluator fails
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


def test_patronus_evaluate_api_success_strategy_any_pass_fails_when_all_fail(
    monkeypatch,
):
    """
    Test that the "any_pass" success strategy fails when all evaluators fail
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


def test_patronus_evaluate_api_internal_error_when_no_env_set():
    """
    Test that an internal error is returned when the PATRONUS_API_KEY variable is not set
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


def test_patronus_evaluate_api_internal_error_when_no_evaluators_provided():
    """
    Test that an internal error is returned when no 'evaluators' dict
    is passed in teh evaluate_config params.
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


def test_patronus_evaluate_api_internal_error_when_evaluator_dict_does_not_have_evaluator_key():
    """
    Test that an internal error is returned when the passed evaluator dict in the
    evaluator_config does not have the 'evaluator' key.
    """
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          success_strategy: "any_pass"
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_success_strategy_is_all_pass_happy_case(
    monkeypatch,
):
    """
    Test that when the success strategy is omitted, the default "all_pass" is chosen,
    and thus the request passes since all evaluators pass.
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "Hi there! How are you doing?"


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_success_strategy_all_pass_fails_when_one_failure(
    monkeypatch,
):
    """
    Test that when the success strategy is omitted, the default "all_pass" is chosen,
    and thus the request fails since one evaluator also fails.
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    },
                    {
                        "evaluator_id": "answer-relevance-large-2024-07-23",
                        "criteria": "patronus:answer-relevance",
                        "status": "success",
                        "evaluation_result": {
                            "pass": False,
                        },
                    },
                ]
            },
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


@pytest.mark.asyncio
def test_patronus_evaluate_api_internal_error_when_400_status_code(
    monkeypatch,
):
    """
    Test that when the API returns a 4XX status code,
    the bot returns an internal error response
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=400,
        )

        chat >> "Hi"
        chat << "I'm sorry, an internal error has occurred."


@pytest.mark.asyncio
def test_patronus_evaluate_api_default_response_when_500_status_code(
    monkeypatch,
):
    """
    Test that when the API returns a 5XX status code,
    the bot returns the default fail response
    """
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    yaml_evaluate_config = """
  config:
    patronus:
      output:
        evaluate_config:
          params:
              {
                evaluators:
                    [
                      { "evaluator": "lynx" },
                      {
                          "evaluator": "answer-relevance",
                          "explain_strategy": "on-fail",
                      },
                    ],
                tags: { "hello": "world" },
              }
    """
    config = RailsConfig.from_content(
        colang_content=COLANG_CONFIG, yaml_content=YAML_PREFIX + yaml_evaluate_config
    )
    chat = TestChat(
        config,
        llm_completions=[
            "Mock generated user intent",
            "Mock generated next step",
            "  Hi there! How are you doing?",
        ],
    )

    with aioresponses() as m:
        chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=500,
        )

        chat >> "Hi"
        chat << "I don't know the answer to that."


def test_check_guardrail_pass_empty_response():
    """Test that empty/None responses return False"""
    assert check_guardrail_pass(None, "all_pass") is False


def test_check_guardrail_pass_missing_results():
    """Test that response without results key returns False"""
    assert check_guardrail_pass({}, "all_pass") is False


def test_check_guardrail_pass_all_pass_strategy_success():
    """Test that all_pass strategy returns True when all evaluators pass"""
    response = {
        "results": [
            {"evaluation_result": {"pass": True}},
            {"evaluation_result": {"pass": True}},
        ]
    }
    assert check_guardrail_pass(response, "all_pass") is True


def test_check_guardrail_pass_all_pass_strategy_failure():
    """Test that all_pass strategy returns False when one evaluator fails"""
    response = {
        "results": [
            {"evaluation_result": {"pass": True}},
            {"evaluation_result": {"pass": False}},
        ]
    }
    assert check_guardrail_pass(response, "all_pass") is False


def test_check_guardrail_pass_any_pass_strategy_success():
    """Test that any_pass strategy returns True when at least one evaluator passes"""
    response = {
        "results": [
            {"evaluation_result": {"pass": False}},
            {"evaluation_result": {"pass": True}},
        ]
    }
    assert check_guardrail_pass(response, "any_pass") is True


def test_check_guardrail_pass_any_pass_strategy_failure():
    """Test that any_pass strategy returns False when all evaluators fail"""
    response = {
        "results": [
            {"evaluation_result": {"pass": False}},
            {"evaluation_result": {"pass": False}},
        ]
    }
    assert check_guardrail_pass(response, "any_pass") is False


def test_check_guardrail_pass_malformed_evaluation_results():
    """Test that malformed evaluation results return False"""
    response = {
        "results": [{"evaluation_result": "not_a_dict"}, {"no_evaluation_result": {}}]
    }
    assert check_guardrail_pass(response, "all_pass") is False


@pytest.mark.asyncio
async def test_patronus_evaluate_request_success(monkeypatch):
    """Test successful API request to Patronus Evaluate endpoint"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    with aioresponses() as m:
        m.post(
            PATRONUS_EVALUATE_API_URL,
            payload={
                "results": [
                    {
                        "evaluator_id": "lynx-large-2024-07-23",
                        "criteria": "patronus:hallucination",
                        "status": "success",
                        "evaluation_result": {
                            "pass": True,
                        },
                    }
                ]
            },
        )

        response = await patronus_evaluate_request(
            api_params={
                "evaluators": [{"evaluator": "lynx"}],
                "tags": {"test": "true"},
            },
            user_input="Does NeMo Guardrails integrate with the Patronus API?",
            bot_response="Yes, NeMo Guardrails integrates with the Patronus API.",
            provided_context="Yes, NeMo Guardrails integrates with the Patronus API.",
        )

        assert "results" in response
        assert len(response["results"]) == 1
        assert response["results"][0]["evaluation_result"]["pass"] is True


@pytest.mark.asyncio
async def test_patronus_evaluate_request_400_error(monkeypatch):
    """Test that ValueError is raised with correct message for 400 status code"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    with aioresponses() as m:
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=400,
        )

        with pytest.raises(ValueError) as exc_info:
            await patronus_evaluate_request(
                api_params={
                    "evaluators": [{"evaluator": "lynx"}],
                },
                user_input="test",
                bot_response="test",
                provided_context="test",
            )
        assert "The Patronus Evaluate API call failed with status code 400." in str(
            exc_info.value
        )


@pytest.mark.asyncio
async def test_patronus_evaluate_request_500_error(monkeypatch):
    """Test that None is returned for 500 status code and no ValueError is raised"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")
    with aioresponses() as m:
        m.post(
            PATRONUS_EVALUATE_API_URL,
            status=500,
        )

        response = await patronus_evaluate_request(
            api_params={
                "evaluators": [{"evaluator": "lynx"}],
            },
            user_input="test",
            bot_response="test",
            provided_context="test",
        )

        assert response is None


@pytest.mark.asyncio
async def test_patronus_evaluate_request_missing_api_key():
    """Test that ValueError is raised with correct message when API key is missing"""
    with pytest.raises(ValueError) as exc_info:
        await patronus_evaluate_request(
            api_params={},
            user_input="test",
            bot_response="test",
            provided_context="test",
        )
    assert "PATRONUS_API_KEY environment variable not set" in str(exc_info.value)


@pytest.mark.asyncio
async def test_patronus_evaluate_request_missing_evaluators(monkeypatch):
    """Test that ValueError is raised when evaluators field is missing"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")

    with pytest.raises(ValueError) as exc_info:
        await patronus_evaluate_request(
            api_params={"tags": {"test": "true"}},
            user_input="test",
            bot_response="test",
            provided_context="test",
        )
    assert (
        "The Patronus Evaluate API parameters must contain an 'evaluators' field"
        in str(exc_info.value)
    )


@pytest.mark.asyncio
async def test_patronus_evaluate_request_evaluators_not_list(monkeypatch):
    """Test that ValueError is raised when evaluators is not a list"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")

    with pytest.raises(ValueError) as exc_info:
        await patronus_evaluate_request(
            api_params={"evaluators": {"evaluator": "lynx"}},
            user_input="test",
            bot_response="test",
            provided_context="test",
        )
    assert "The Patronus Evaluate API parameter 'evaluators' must be a list" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_patronus_evaluate_request_evaluator_not_dict(monkeypatch):
    """Test that ValueError is raised when evaluator is not a dictionary"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")

    with pytest.raises(ValueError) as exc_info:
        await patronus_evaluate_request(
            api_params={"evaluators": ["lynx"]},
            user_input="test",
            bot_response="test",
            provided_context="test",
        )
    assert "Each object in the 'evaluators' list must be a dictionary" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_patronus_evaluate_request_evaluator_missing_field(monkeypatch):
    """Test that ValueError is raised when evaluator dict is missing evaluator field"""
    monkeypatch.setenv("PATRONUS_API_KEY", "xxx")

    with pytest.raises(ValueError) as exc_info:
        await patronus_evaluate_request(
            api_params={"evaluators": [{"explain_strategy": "on-fail"}]},
            user_input="test",
            bot_response="test",
            provided_context="test",
        )
    assert (
        "Each dictionary in the 'evaluators' list must contain the 'evaluator' field"
        in str(exc_info.value)
    )
