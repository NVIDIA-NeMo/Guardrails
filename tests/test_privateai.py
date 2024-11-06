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

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.actions.actions import ActionResult, action
from tests.utils import TestChat


@action()
def retrieve_relevant_chunks():
    context_updates = {"relevant_chunks": "Mock retrieved context."}

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


def mock_detect_pii(return_value=True):
    def mock_request(*args, **kwargs):
        return return_value

    return mock_request


@pytest.mark.unit
def test_privateai_pii_detection_no_active_pii_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(mock_detect_pii(True), "detect_pii")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "Hi! My name is John as well."


@pytest.mark.unit
def test_privateai_pii_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  input:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              input:
                flows:
                  - detect pii on input
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(mock_detect_pii(True), "detect_pii")
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."


@pytest.mark.unit
def test_privateai_pii_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  output:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              output:
                flows:
                  - detect pii on output
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(mock_detect_pii(True), "detect_pii")
    chat >> "Hi!"
    chat << "I can't answer that."


@pytest.mark.skip(reason="This test needs refinement.")
@pytest.mark.unit
def test_privateai_pii_detection_retrieval_with_pii():
    # TODO: @pouyanpi and @letmerecall: Find an alternative approach to test this functionality.
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  retrieval:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              retrieval:
                flows:
                  - detect pii on retrieval
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(mock_detect_pii(True), "detect_pii")

    # When the relevant_chunks has_pii, a bot intent will get invoked via (bot inform answer unknown), which in turn
    # will invoke retrieve_relevant_chunks action.
    # With a mocked retrieve_relevant_chunks always returning something & mocked detect_pii always returning True,
    # the process goes in an infinite loop and raises an Exception: Too many events.
    with pytest.raises(Exception, match="Too many events."):
        chat >> "Hi!"
        chat << "I can't answer that."


@pytest.mark.unit
def test_privateai_pii_detection_retrieval_with_no_pii():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
              config:
                privateai:
                  server_endpoint: https://api.private-ai.com/cloud/v3/process/text
                  retrieval:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              retrieval:
                flows:
                  - detect pii on retrieval
        """,
        colang_content="""
            define user express greeting
              "hi"

            define flow
              user express greeting
              bot express greeting

            define bot inform answer unknown
              "I can't answer that."
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! My name is John as well."',
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat.app.register_action(mock_detect_pii(False), "detect_pii")

    chat >> "Hi!"
    chat << "Hi! My name is John as well."
