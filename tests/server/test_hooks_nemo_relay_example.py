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

"""The policies of the Deep Agents example in examples/scripts/nemo_relay, through the hooks server's ASGI app."""

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from nemoguardrails.imports import check_optional_dependency
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.server.hooks.app import create_hooks_app

pytestmark = pytest.mark.skipif(not check_optional_dependency("cel"), reason="requires the 'cel' extra (Python 3.11+)")

CONFIGS = Path(__file__).parents[2] / "examples" / "scripts" / "nemo_relay" / "configs"
CEL_DENY = "cel check tool call"


@pytest.fixture(scope="module")
def client():
    """A hooks server for the example's model-free policy, `configs/policy`."""
    with TestClient(create_hooks_app(RailsConfig.from_path(str(CONFIGS / "policy")))) as client:
        yield client


@pytest.mark.parametrize(
    "tool_name, tool_input, reason",
    [
        ("read_log", {"service": "checkout", "lines": 200}, None),
        ("read_log", {"service": "checkout", "lines": 500}, None),
        ("read_log", {"service": "checkout", "lines": 501}, CEL_DENY),
        ("read_log", {"service": "checkout", "lines": 0}, CEL_DENY),
        ("read_log", {"service": "checkout", "lines": "200"}, CEL_DENY),
        ("read_log", {"service": "checkout"}, CEL_DENY),
        (
            "read_log",
            {"service": "checkout", "lines": 5, "path": "/etc/passwd"},
            "tool call 'read_log' passes 1 argument its allowed_tools entry doesn't list",
        ),
        ("http_request", {"method": "GET", "url": "https://status.internal.example/payments"}, None),
        ("http_request", {"method": "get", "url": "https://internal.example/"}, None),
        ("http_request", {"method": "POST", "url": "https://status.internal.example/payments"}, CEL_DENY),
        ("http_request", {"method": "GET", "url": "https://diag.example-attacker.net/?k=1"}, CEL_DENY),
        ("http_request", {"method": "GET", "url": "https://internal.example.attacker.net/"}, CEL_DENY),
        ("http_request", {"method": "GET", "url": "https://internal.example@attacker.net/"}, CEL_DENY),
        ("http_request", {"method": "GET", "url": "http://status.internal.example/"}, CEL_DENY),
        ("http_request", {"url": "https://status.internal.example/"}, CEL_DENY),
        ("http_request", {"method": 1, "url": "https://status.internal.example/"}, CEL_DENY),
        ("http_request", {"method": "GET"}, CEL_DENY),
        ("search_runbooks", {"query": "checkout"}, None),
        ("task", {"description": "x", "subagent_type": "general-purpose"}, None),
        ("execute", {"command": "ls"}, "tool call 'execute' is not an allowed tool"),
        ("delete", {"file_path": "/app/.env"}, "tool call 'delete' is not an allowed tool"),
    ],
    ids=[
        "read-within-limit",
        "read-at-limit",
        "read-over-limit",
        "read-zero-lines",
        "read-lines-not-an-int",
        "read-no-lines",
        "read-unlisted-argument",
        "get-internal-host",
        "get-lower-case-method",
        "post-internal-host",
        "get-other-host",
        "get-host-with-internal-prefix",
        "get-internal-userinfo",
        "get-plain-http",
        "request-no-method",
        "request-method-not-a-string",
        "request-no-url",
        "search-runbooks",
        "general-purpose-subagent",
        "shell",
        "delete",
    ],
)
def test_policy(client, tool_name, tool_input, reason):
    """The policy allows the example's safe calls and denies the rest, including calls that leave an argument out."""
    response = client.post(
        "/v1/hooks/nemo-relay",
        json={
            "hook_event_name": "NemoRelayToolConditionalExecutionGuardrail",
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
    )

    expected = {"decision": "allow"} if reason is None else {"decision": "deny", "reason": reason}
    assert (response.status_code, response.json()) == (200, expected)


def test_judge_config_loads(monkeypatch):
    """The judge config starts a hooks server, without calling its model."""
    monkeypatch.setenv("NVIDIA_API_KEY", "not-used")

    create_hooks_app(RailsConfig.from_path(str(CONFIGS / "policy_judge")))


def test_judge_config_has_the_policy_configs_checks():
    """The judge config repeats the policy config's tool policy and CEL expressions, and only adds the judge."""
    policy, judge = (yaml.safe_load((CONFIGS / name / "config.yml").read_text()) for name in ("policy", "policy_judge"))

    assert judge["hooks_server"] == policy["hooks_server"]
    assert judge["rails"]["config"]["cel"] == policy["rails"]["config"]["cel"]
    policy_rails, judge_rails = (config["rails"]["tool_call"]["per_tool"] for config in (policy, judge))
    assert judge_rails["read_log"] == policy_rails["read_log"]
    assert judge_rails["http_request"][:-1] == policy_rails["http_request"]
    assert judge_rails["http_request"][-1].startswith("tool safety check tool call $model=judge")
