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

from nemoguardrails import RailsConfig
from tests.utils import TestChat

colang_content = """
    import core
    import llm

    flow main
        activate generating user intent for unhandled user utterance
        activate continuation on undefined flow
        await user expressed greeting
        bot say "hi there"
    """

yaml_content = """
colang_version: "2.x"
models:
  - type: main
    engine: openai
    model: gpt-4-turbo

    """


def test_1():
    config = RailsConfig.from_content(colang_content, yaml_content)

    chat = TestChat(
        config,
        llm_completions=["user intent: user expressed greeting"],
    )

    chat >> "hi"
    chat << "hi there"


if __name__ == "__main__":
    test_1()
