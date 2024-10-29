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
from tests.utils import TestChat


@pytest.mark.unit
def test_privateai_pii_detection_input_output(monkeypatch):
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
                  output:
                    entities:
                      - EMAIL_ADDRESS
                      - NAME
              input:
                flows:
                  - detect pii on input
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

    def mock_detect_pii(return_value=True):
        def mock_request(*args, **kwargs):
            return return_value

        return mock_request

    chat.app.register_action(mock_detect_pii(True), "detect_pii")

    # This will trigger the input rail
    chat >> "Hi! I am Mr. John! And my email is test@gmail.com"
    chat << "I can't answer that."

    # This will trigger only the output one
    chat >> "Hi!"
    chat << "I can't answer that."
