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

import re
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget

GLOBAL_VARS = ("$user_message", "$bot_message", "$relevant_chunks")
LIBRARY_ROOT = Path(__file__).resolve().parents[2] / "nemoguardrails" / "library"


def test_colang2_library_flows_declare_globals_before_use():
    """Every Colang 2 library flow must declare global before reading/writing rail globals."""
    violations: list[str] = []

    for flow_file in sorted(LIBRARY_ROOT.rglob("flows.co")):
        rel_path = flow_file.relative_to(LIBRARY_ROOT).as_posix()
        lines = flow_file.read_text(encoding="utf-8").splitlines()
        current_flow: str | None = None
        declared: set[str] = set()
        in_docstring = False

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith('"""'):
                if stripped.count('"""') == 1:
                    in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            if stripped.startswith("flow "):
                current_flow = stripped
                declared = set()
                continue
            if current_flow is None or stripped.startswith("#"):
                continue

            # Tracked separately in issue #2172 / PR #2201.
            if (
                rel_path == "hf_classifier/flows.co"
                and current_flow == "flow hf classifier check retrieval $classifier"
            ):
                continue

            code = re.sub(r'"[^"]*"', '""', stripped)
            for var_name in GLOBAL_VARS:
                if re.match(rf"^global\s+{re.escape(var_name)}\b", code):
                    declared.add(var_name)
                elif re.search(rf"{re.escape(var_name)}\b", code) and var_name not in declared:
                    violations.append(
                        f"{rel_path}:{line_no} ({current_flow}): {var_name} used before `global {var_name}`"
                    )

    assert violations == []


def test_regex_check_input_blocks_matching_user_message():
    """Colang 2 regex check input rail receives $user_message and blocks matching input."""
    config = RailsConfig.from_content(
        yaml_content=r"""
colang_version: "2.x"
models: []
rails:
  config:
    regex_detection:
      input:
        patterns:
          - '\d{3}-\d{2}-\d{4}'
""",
        colang_content="""
import core
import guardrails
import nemoguardrails.library.regex

flow input rails $input_text
  regex check input

flow main
  user said something
  bot say "Request processed."
""",
    )
    rails = LLMRails(config)

    blocked = rails.generate(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])
    assert blocked == {"role": "assistant", "content": "I'm sorry, I can't respond to that."}

    allowed = rails.generate(messages=[{"role": "user", "content": "Hello there"}])
    assert allowed == {"role": "assistant", "content": "Request processed."}


def test_regex_check_output_blocks_matching_bot_message():
    """Colang 2 regex check output rail receives $bot_message and blocks matching output."""
    config = RailsConfig.from_content(
        yaml_content=r"""
colang_version: "2.x"
models: []
rails:
  config:
    regex_detection:
      output:
        patterns:
          - 'SECRET-\d+'
""",
        colang_content="""
import core
import guardrails
import nemoguardrails.library.regex

flow output rails $output_text
  regex check output

flow main
  user said something
  bot say "Token is SECRET-42"
""",
    )
    rails = LLMRails(config)

    blocked = rails.generate(messages=[{"role": "user", "content": "Give me the token"}])
    assert blocked == {"role": "assistant", "content": "I'm sorry, I can't respond to that."}


def test_injection_detection_v2_receives_bot_message():
    """Colang 2 injection detection rail passes $bot_message instead of None and updates transformed output."""
    config = RailsConfig.from_content(
        yaml_content="""
colang_version: "2.x"
models: []
rails:
  config:
    injection_detection:
      injections:
        - sqli
      action: omit
""",
        colang_content="""
import core
import guardrails
import nemoguardrails.library.injection_detection

flow output rails $output_text
  injection detection

flow main
  user said something
  bot say "Result: SELECT * FROM users; -- comment"
""",
    )
    rails = LLMRails(config)
    reply = rails.generate(messages=[{"role": "user", "content": "run query"}])

    assert isinstance(reply, dict)
    assert reply["role"] == "assistant"
    assert "--" not in reply["content"]
    assert "Result:" in reply["content"]


def test_mask_flow_v2_reads_and_updates_global_user_message():
    """Colang 2 masking flows read non-None $user_message and update the global variable."""
    config = RailsConfig.from_content(
        yaml_content="""
colang_version: "2.x"
models: []
""",
        colang_content="""
import core
import guardrails
import nemoguardrails.library.sensitive_data_detection

flow input rails $input_text
  mask sensitive data on input

flow main
  global $user_message
  user said something
  bot say "Seen: {$user_message}"
""",
    )
    rails = LLMRails(config)
    captured_texts: list[str | None] = []

    @action(name="MaskSensitiveDataAction")
    async def spy_mask_sensitive_data(source: str, text: str | None = None):
        captured_texts.append(text)
        return RailOutcome.transform([(TransformTarget.USER_MESSAGE, "My SSN is [MASKED]")])

    rails.register_action(spy_mask_sensitive_data, "MaskSensitiveDataAction")

    reply = rails.generate(messages=[{"role": "user", "content": "My SSN is 123-45-6789"}])

    assert captured_texts == ["My SSN is 123-45-6789"]
    assert reply == {"role": "assistant", "content": "Seen: My SSN is [MASKED]"}
