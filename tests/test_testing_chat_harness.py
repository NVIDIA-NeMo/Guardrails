# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_testchat_constructor_does_not_replace_active_explain_context():
    config = RailsConfig.from_content(
        config={
            "models": [],
            "instructions": [
                {
                    "type": "general",
                    "content": "This is a conversation between a user and a bot.",
                }
            ],
        }
    )

    chat1 = TestChat(config, llm_completions=["  Hello one"])
    chat1_info = chat1.app.explain()
    _chat2 = TestChat(config, llm_completions=["  Hello two"])

    chat1 >> "hello"
    chat1 << "Hello one"

    assert chat1.app.explain() is chat1_info
    assert len(chat1_info.llm_calls) == 1
