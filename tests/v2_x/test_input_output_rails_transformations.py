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

from nemoguardrails import RailsConfig
from tests.utils import TestChat

yaml_content = """
colang_version: "2.x"
models:
  - type: main
    engine: openai
    model: gpt-4-turbo
"""


def test_1():
    """Test input and output rails transformations."""

    colang_content = """
    import core
    import guardrails

    flow input rails $input_text
        global $user_message
        $user_message = "{$input_text}, Dick"

    flow output rails $output_text
        global $user_message
        global $bot_message
        $bot_message = "{$user_message}, and Harry"

    flow main
        global $last_bot_message
        await user said "Tom"
        bot say "{$last_bot_message}"
    """

    config = RailsConfig.from_content(colang_content, yaml_content)
    chat = TestChat(
        config,
        llm_completions=[],
    )
    chat >> "Tom"
    chat << "Tom, Dick, and Harry"


if __name__ == "__main__":
    test_1()
