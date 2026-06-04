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

"""Regression tests for Colang 2.x LLM continuation dynamic flow parsing.

Documents https://github.com/NVIDIA-NeMo/Guardrails/issues/TBD — multiline or
truncated LLM ``bot action`` text is injected into Colang source and fails
``parse_colang_file``, yielding empty bot responses.

No live LLM is required.
"""

from __future__ import annotations

import pytest

from nemoguardrails.actions.llm.utils import (
    escape_flow_name,
    get_first_bot_action,
    get_first_bot_intent,
    remove_action_intent_identifiers,
)
from nemoguardrails.colang import parse_colang_file


def _build_generate_flow_continuation_body(
    bot_intent: str,
    bot_action: str,
    *,
    flow_uuid: str = "test0001",
) -> str:
    """Mirror ``GenerateFlowContinuationAction`` body assembly in generation.py."""
    escaped_intent = escape_flow_name(bot_intent.strip(" "))
    flow_name = f"_dynamic_{flow_uuid} {escaped_intent}"
    return f'@meta(bot_intent="{escaped_intent}")\n' + f"flow {flow_name}\n" + f"  {bot_action}"


def _parse_generated_flow_body(body: str) -> None:
    parse_colang_file(filename="", content=body, version="2.x", include_source_mapping=False)


# --- LLM output fixtures (no model call) ---

SIMPLE_LLM_CONTINUATION = """\
bot intent: bot greet user
bot action: bot say "hi"
"""

MULTILINE_WITH_BACKTICKS_LLM_CONTINUATION = """\
bot intent: bot explain recursive directory removal in Linux
bot action: bot say "To recursively remove a directory and all its contents in Linux, you can use
the `rm` command with the `-r` (recursive) and `-f` (force) options. Here's how you
would do it:"
"""


def test_simple_greeting_flow_body_parses():
    """Control: single-line ``bot say`` in a generated flow body is valid Colang."""
    lines = SIMPLE_LLM_CONTINUATION.splitlines()
    bot_intent = get_first_bot_intent(lines)
    assert bot_intent is not None
    bot_action = get_first_bot_action(lines)
    assert bot_action == 'bot say "hi"'

    body = _build_generate_flow_continuation_body(bot_intent, bot_action)
    _parse_generated_flow_body(body)


def test_get_first_bot_action_truncates_multiline_llm_output():
    """``get_first_bot_action`` stops at the second line, leaving an unclosed string."""
    lines = MULTILINE_WITH_BACKTICKS_LLM_CONTINUATION.splitlines()
    bot_action = get_first_bot_action(lines)

    assert bot_action is not None
    assert "\n" not in bot_action
    assert bot_action == (
        'bot say "To recursively remove a directory and all its contents in Linux, you can use'
    )
    assert not bot_action.endswith('would do it:"')


def test_multiline_bot_say_flow_body_fails_colang_parse():
    """Multi-line ``bot say`` inside generated flow source fails the Colang parser."""
    lines = MULTILINE_WITH_BACKTICKS_LLM_CONTINUATION.splitlines()
    bot_intent = get_first_bot_intent(lines)
    assert bot_intent is not None
    bot_action = get_first_bot_action(lines)
    assert bot_action is not None

    body = _build_generate_flow_continuation_body(bot_intent, bot_action)

    with pytest.raises(Exception) as exc_info:
        _parse_generated_flow_body(body)

    assert 'No terminal matches \'"\'"' in str(exc_info.value) or "Unexpected" in type(exc_info.value).__name__


def test_multiline_full_bot_action_also_fails_colang_parse():
    """Even with a complete multi-line utterance, inline ``bot say \"...\"`` is invalid Colang."""
    bot_intent = "bot explain recursive directory removal in Linux"
    bot_action = (
        'bot say "To recursively remove a directory and all its contents in Linux, you can use\n'
        'the `rm` command with the `-r` (recursive) and `-f` (force) options."'
    )
    body = _build_generate_flow_continuation_body(bot_intent, bot_action)

    with pytest.raises(Exception):
        _parse_generated_flow_body(body)


def test_add_flows_fallback_flow_name_extracted_from_meta_line_is_invalid():
    """Mirror broken fallback in ``RuntimeV2_x._add_flows_action`` (line 86)."""
    body = _build_generate_flow_continuation_body(
        "bot explain recursive directory removal in Linux",
        'bot say "truncated',
    )
    flow_name = body.split("\n")[0].split(" ", maxsplit=1)[1]

    assert flow_name.endswith('")')
    assert "explain recursive directory removal" in flow_name

    fixed_body = f"flow {flow_name}\n" + f'  bot say "Internal error on flow `{flow_name}`."'
    with pytest.raises(Exception):
        _parse_generated_flow_body(fixed_body)


def test_code_block_response_parsing_fails_via_truncation():
    """Backticks in a multi-line answer trigger failure via truncation + parse, not backticks alone."""
    line0 = remove_action_intent_identifiers(["bot intent: bot show rm example"])[0].strip()
    assert line0.startswith("bot ")

    single_line_with_backticks = 'bot say "Use `rm -rf` carefully."'
    body = _build_generate_flow_continuation_body("bot show rm example", single_line_with_backticks)
    _parse_generated_flow_body(body)

    truncated = get_first_bot_action(MULTILINE_WITH_BACKTICKS_LLM_CONTINUATION.splitlines())
    body_bad = _build_generate_flow_continuation_body("bot explain rm", truncated)
    with pytest.raises(Exception):
        _parse_generated_flow_body(body_bad)
