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

import logging

from rich.logging import RichHandler

from nemoguardrails import RailsConfig
from tests.utils import TestChat

FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X,%f]",
    handlers=[RichHandler(markup=True)],
)


def test_add_new_flow_action():
    """Test of action that adds a new flow during runtime."""
    new_flow_content = """
    flow test message a
      start UtteranceBotAction(script="message a!")

    flow test message b
      start UtteranceBotAction(script="message b!")
    """

    config = RailsConfig.from_content(
        colang_content="""
        flow main
          match UtteranceUserAction().Finished(final_transcript="start")
          $flows = await AddFlowsAction(config=$new_flow_content)
          start test message a
          start test message b
          #$flow_name_a = flow($flows[0])
          #$flow_name_b = flow($flows[1])
          #start $flow_name_a
          #start $flow_name_b
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    # Hack to add new flow content to main flow since multiline strings are not supported yet
    chat.state.main_flow_state.context["new_flow_content"] = new_flow_content

    chat >> "start"
    chat << "message a!\nmessage b!"  # "message a!\nmessage b!\nmessage a!\nmessage b!"


def test_check_for_active_flow_finished_match_action():
    """Test of action that checks for active flow finished matches."""
    config = RailsConfig.from_content(
        colang_content="""
        flow a
          await b "test"

        flow b $param
          match UtteranceUserAction().Finished(final_transcript="b")

        flow main
          match UtteranceUserAction().Finished(final_transcript="start")
          start a
          $ready = await CheckForActiveEventMatchAction(event_name="FlowFinished", flow_id="b", param="test")
          await UtteranceBotAction(script="message {$ready}")
          $ready = await CheckForActiveEventMatchAction(event_name="FlowFinished", flow_id="b", param="test1")
          await UtteranceBotAction(script="message {$ready}")
          $ready = await CheckForActiveEventMatchAction(event_name="FlowFinished", flow_id="a")
          await UtteranceBotAction(script="message {$ready}")
        """,
        yaml_content="""
        colang_version: "2.x"
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    chat >> "start"
    chat << "message True\nmessage False\nmessage False"


if __name__ == "__main__":
    test_check_for_active_flow_finished_match_action()
