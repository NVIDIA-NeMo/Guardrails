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


def test_1():
    config = RailsConfig.from_content(
        """
        define flow example input rail
          done
        """,
        """
        rails:
            input:
                flows:
                    - example input rail
        """,
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    messages = [{"role": "user", "content": "Hello! What can you do for me?"}]

    response = chat.app.generate(
        messages=messages,
        options={
            "rails": ["input"],
            "log": {
                "activated_rails": True,
            },
        },
    )

    # We should only get the input rail here.
    assert len(response.log.activated_rails) == 1
