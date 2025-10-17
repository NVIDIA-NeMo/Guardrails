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
    level=logging.DEBUG,
    format=FORMAT,
    datefmt="[%X,%f]",
    handlers=[RichHandler(markup=True)],
)


def test_1():
    """Test use of expression as statements."""
    config = RailsConfig.from_content(
        colang_content="""
        flow main
          match UtteranceUserActionFinished()
          $v1 = ..."Generate the company' name\\" from users input"
          $v2 = ... 'Generate the company\\' name" from users input'
          $v3 = ...'''Generate the company' name" from users input'''
          $v4 = ...  '''Generate the company'
                    name" from users input'''

          await UtteranceBotAction(script="{$v1}{$v2}{$v3}{$v4}")
        """,
        yaml_content="""
        colang_version: "2.x"
        models:
        - type: main
          engine: openai
          model: gpt-3.5-turbo-instruct
        """,
    )

    chat = TestChat(
        config,
        llm_completions=["'1'", "'2'", "'3'", "'4'"],
    )

    chat >> "hi"
    chat << "1234"


if __name__ == "__main__":
    test_1()
