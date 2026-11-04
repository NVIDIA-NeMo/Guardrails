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

from unittest.mock import Mock, patch

import pytest
import yara

from nemoguardrails import Guardrails, RailsConfig
from nemoguardrails.library.injection_detection.actions import YARA_DIR, _omit_injection, injection_detection
from nemoguardrails.rails.llm.options import RailStatus, RailType
from tests.utils import TestChat

NESTED_SCRIPT = "<scr<script>x</script>ipt>alert(1)</scr<script>x</script>ipt>"
REFUSAL = "I'm sorry, the desired output triggered rule(s) designed to mitigate exploitation of markdown_xss."


def injection_config(injections, action="omit", yara_rules=None, colang_content=None, **kwargs):
    return RailsConfig.from_content(
        config={
            "models": [],
            "rails": {
                "config": {
                    "injection_detection": {
                        "injections": injections,
                        "action": action,
                        "yara_rules": yara_rules or {},
                    }
                },
                **({"output": {"flows": ["injection detection"]}} if kwargs.get("colang_version") != "2.x" else {}),
            },
            **kwargs,
        },
        colang_content=colang_content,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["reject", "omit"])
@pytest.mark.parametrize(
    ("module", "text", "rule"),
    [
        ("code", "Almost done. import os", "import_shells"),
        ("code", "requests first. import requests", "import_networking"),
        ("code", "os from os import system", "import_shells"),
        ("code", "requests from requests import get", "import_networking"),
        ("code", "import harmless; from os import system", "import_shells"),
        ("code", "os __import__('os').system('id')", "import_shells"),
        ("template", "}} {{ 7*7 }}", "jinja_injection"),
        ("template", "%} {% if true %}yes{% endif %}", "jinja_injection"),
        ("xss", 'javascript <a href="javascript:alert(1)">click</a>', "markdown_xss"),
    ],
)
async def test_prefix_does_not_suppress_detection(module, text, rule, action):
    outcome = await injection_detection(text, injection_config([module], action))

    assert outcome.is_blocked or outcome.is_transform
    assert outcome.metadata["is_injection"] is True
    assert rule in outcome.metadata["detections"]


@pytest.mark.asyncio
async def test_omit_rejects_newly_assembled_script():
    outcome = await injection_detection(NESTED_SCRIPT, injection_config(["xss"]))

    assert outcome.is_blocked
    assert not outcome.transforms
    assert outcome.metadata["is_injection"] is True
    assert outcome.metadata["detections"] == ["markdown_xss"]


@pytest.mark.asyncio
async def test_omit_rescans_all_configured_rules():
    rules = {
        "xss": (YARA_DIR / "xss.yara").read_text(),
        "assembled": 'rule assembled { strings: $s = "javascript" condition: $s }',
    }
    outcome = await injection_detection("java<script>x</script>script", injection_config(list(rules), yara_rules=rules))

    assert outcome.is_blocked
    assert set(outcome.metadata["detections"]) == {"markdown_xss", "assembled"}


def test_omit_removes_overlapping_byte_ranges_with_unicode():
    rules = yara.compile(source='rule overlap { strings: $a = "abc" $b = "bcd" condition: any of them }')
    text = "éabcd🙂abcdé"

    assert _omit_injection(text, rules.match(data=text)) == (True, "é🙂é")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "text"),
    [
        ("rule no_strings { condition: true }", "some text"),
        ("rule partial_utf8 { strings: $s = { A9 } condition: $s }", "é"),
        ("rule empty_match { strings: $s = /a*/ condition: $s }", "bbb"),
    ],
)
async def test_omit_unremovable_matches_fail_closed(source, text):
    outcome = await injection_detection(text, injection_config(["custom"], yara_rules={"custom": source}))

    assert outcome.is_blocked
    assert outcome.metadata["is_injection"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [AttributeError("private text"), ValueError("private text")])
async def test_omit_removal_error_fails_closed_without_logging_text(failure, caplog):
    with patch("nemoguardrails.library.injection_detection.actions._omit_injection", side_effect=failure):
        outcome = await injection_detection("<script>x</script>", injection_config(["xss"]))

    assert outcome.is_blocked
    assert "private text" not in caplog.text


@pytest.mark.asyncio
async def test_omit_rescan_error_fails_closed_without_logging_text(caplog):
    text = "<script>x</script>"
    rules = Mock()
    rules.match.side_effect = [
        yara.compile(filepath=str(YARA_DIR / "xss.yara")).match(data=text),
        yara.Error("private text"),
    ]
    with patch("nemoguardrails.library.injection_detection.actions._load_rules", return_value=rules):
        outcome = await injection_detection(text, injection_config(["xss"]))

    assert outcome.is_blocked
    assert "private text" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("exceptions", [False, True])
async def test_colang_1_omit_fallback_blocks_output(exceptions):
    config = injection_config(["xss"], enable_rails_exceptions=exceptions)
    chat = TestChat(config, llm_completions=[NESTED_SCRIPT])

    result = await chat.app.generate_async(messages=[{"role": "user", "content": "hello"}])

    if exceptions:
        assert result["role"] == "exception"
        assert result["content"]["type"] == "InjectionDetectionRailException"
    else:
        assert result["content"] == REFUSAL


@pytest.mark.parametrize(
    ("text", "exceptions", "expected"),
    [
        (NESTED_SCRIPT, False, REFUSAL),
        ("Hello <script>x</script> world", False, "Hello  world"),
        (NESTED_SCRIPT, True, {"type": "InjectionDetectionRailException"}),
    ],
)
def test_colang_2_omit_blocks_or_transforms_output(text, exceptions, expected):
    config = injection_config(
        ["xss"],
        colang_version="2.x",
        enable_rails_exceptions=exceptions,
        colang_content=f'''
import core
import llm
import guardrails
import nemoguardrails.library.injection_detection

flow output rails $output_text
  injection detection

flow main
  activate llm continuation
  user said something
  bot say "{text}"
''',
    )
    chat = TestChat(config)

    chat >> "hello"
    chat << expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "status", "content"),
    [
        (NESTED_SCRIPT, RailStatus.BLOCKED, "I'm sorry, I can't respond to that."),
        ("Hello <script>x</script> world", RailStatus.MODIFIED, "Hello  world"),
    ],
)
async def test_iorails_omit_blocks_or_transforms_output(text, status, content):
    async with Guardrails(injection_config(["xss"]), require_iorails=True) as rails:
        result = await rails.check_async(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": text}],
            rail_types=[RailType.OUTPUT],
        )

    assert result.status is status
    assert result.content == content
