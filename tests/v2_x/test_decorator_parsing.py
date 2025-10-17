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

CONFIGS_FOLDER = os.path.join(os.path.dirname(__file__), "../test_configs")


def test_1():
    config = RailsConfig.from_content(
        colang_content="""

        @loop("bla")
        flow some flow
          return

        @meta(llm_exclude=True)
        @meta(user_intent=True)
        @loop(id="bla")
        flow some other flow
          return
          user said "hi"

        flow main
          return

    """,
        config={"colang_version": "2.x"},
    )

    chat = TestChat(
        config,
        llm_completions=[],
    )

    flows = chat.app.runtime.flow_configs

    assert flows["some flow"].loop_id == "bla"
    assert flows["some other flow"].loop_id == "bla"
    assert flows["some other flow"].decorators["meta"]["llm_exclude"]
    assert flows["some other flow"].decorators["meta"]["user_intent"]
