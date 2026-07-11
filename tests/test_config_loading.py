# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nemoguardrails.rails.llm.config import Instruction, _load_path


def test_default_instructions():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hello"
        """
    )

    assert config.instructions == [
        Instruction(
            type="general",
            content="Below is a conversation between a helpful AI assistant and a user. "
            "The bot is designed to generate human-like text based on the input that it receives. "
            "The bot is talkative and provides lots of specific details. "
            "If the bot does not know the answer to a question, it truthfully says it does not know.",
        )
    ]


def test_instructions_override():
    config = RailsConfig.from_content(
        """
        define user express greeting
          "hello"
        """,
        """
        instructions:
        - type: "general"
          content: |
            Below is a conversation between a helpful AI assistant and a user.
        """,
    )

    assert config.instructions == [
        Instruction(
            type="general",
            content="Below is a conversation between a helpful AI assistant and a user.\n",
        )
    ]


def test_load_path_kb_prefixed_names_are_not_dropped(tmp_path):
    # A file or folder whose name merely starts with "kb" (but is not inside the
    # `kb/` knowledge-base folder) must be loaded normally, not silently dropped
    # as an empty knowledge-base doc.
    (tmp_path / "kbsettings.yml").write_text('sample_conversation: "hello world"\n')
    (tmp_path / "kbase").mkdir()
    (tmp_path / "kbase" / "greeting.co").write_text("define flow greeting\n  bot express greeting\n")
    # A genuine kb/ document must still be collected as a doc (control).
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "doc.md").write_text("# real kb doc\n")

    raw_config, colang_files = _load_path(str(tmp_path))

    # kb-prefixed config file is parsed, not dropped.
    assert raw_config.get("sample_conversation") == "hello world"
    # kb-prefixed subfolder's colang file is registered.
    assert any(name == "greeting.co" for name, _ in colang_files)
    # a genuine kb/ document is still collected as a doc (control).
    assert {"format": "md", "content": "# real kb doc\n"} in raw_config.get("docs", [])
