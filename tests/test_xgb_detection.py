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
    context_updates = {"relevant_chunks": "Mock retrieve context"}
    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )


COLANG_CONTENT = """
    define user express greeting
      "hi"

    define flow
      user express greeting
      bot express greeting

    define bot inform answer unknown
      "I can't answer that."
"""


@pytest.mark.unit
def test_xgb_spam_detection_no_active_spam_detection():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
                config:
                    xgb:
                        input:
                            detectors:
                                - SPAM
            input:
                flows:
                    - xgb detect on input
        """,
        colang_content=COLANG_CONTENT,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! Nice to meet you."',
        ],
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi!"
    chat << "Hi! Nice to meet you."


@pytest.mark.unit
def test_xgb_spam_detection_input():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
                config:
                    xgb:
                        input:
                            detectors:
                                - SPAM
            input:
                flows:
                    - xgb detect on input
        """,
        colang_content=COLANG_CONTENT,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            '  "Hi! Nice to meet you"',
        ],
    )
    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")

    chat >> "Hi! GENT! We are trying to contact you. Last weekends draw shows that you won a £1000 prize GUARANTEED. Call 09064012160.Claim Code K52. Valid 12hrs only. 150ppm!"

    chat << "I can't answer that."


@pytest.mark.unit
def test_xgb_spam_detection_output():
    config = RailsConfig.from_content(
        yaml_content="""
            models: []
            rails:
                config:
                    xgb:
                        output:
                            detectors:
                                - SPAM
                output:
                    flows:
                       - xgb detect on output
        """,
        colang_content=COLANG_CONTENT,
    )

    chat = TestChat(
        config,
        llm_completions=[
            "  express greeting",
            """  'GENT! We are trying to contact you. Last weekends draw shows that you won a £1000 prize GUARANTEED.
            Call 09064012160.Claim Code K52. Valid 12hrs only. 150ppm!' """,
        ],
    )

    chat.app.register_action(retrieve_relevant_chunks, "retrieve_relevant_chunks")
    chat >> "Hi!"
    chat << "I can't answer that."
