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

"""Check that the buffered contract stays derived from the runtime endpoint."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental.provider.projection_policy import CONTRACT_VERSION, EXTENSION
from nemoguardrails.server.experimental.providers.openai.chat_completions.contract import export_contract
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT

ROOT = Path(__file__).parents[3]
CONTRACTS = ROOT / "nemoguardrails/server/experimental/contracts"
EXPORTED = CONTRACTS / "openai/_generated/chat-completions.buffered.guard.yaml"
MODULE = "nemoguardrails.server.experimental.providers.openai.chat_completions.contract"


def test_buffered_export_matches_checked_in_artifact_and_format():
    contract = export_contract()
    assert contract == yaml.safe_load(EXPORTED.read_text(encoding="utf-8"))
    schema = json.loads((CONTRACTS / "guard-contract.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(contract)
    assert contract["version"] == CONTRACT_VERSION == "1.0.0-alpha.1"
    assert "stream" not in contract
    assert "stream_hooks" not in contract["integration"]["endpoint"]


def test_export_uses_the_runtime_endpoint_metadata():
    exported = export_contract()
    endpoint = CHAT_COMPLETIONS_ENDPOINT
    assert exported["integration"]["endpoint"] == {
        "route_path": endpoint.route_path,
        "operation_label": endpoint.operation,
        "unsupported_request_code": endpoint.unsupported_request_code,
        "unsupported_response_code": endpoint.unsupported_response_code,
    }
    runtime_contract = endpoint.guarded_request_model.projection_contract
    assert runtime_contract is not None
    assert exported["profile"] == runtime_contract.profile.value
    assert exported["request"][EXTENSION]["model"] == endpoint.guarded_request_model.__name__
    assert exported["response"][EXTENSION]["model"] == endpoint.guarded_response_model.__name__
    assert (
        exported["request"][EXTENSION]["stream_selector_field"] == endpoint.guarded_request_model.stream_selector_field
    )


def test_export_never_reads_the_earlier_authored_contract(monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Export must derive from Python, not read YAML")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert export_contract()["operationId"] == "createChatCompletion"


def test_export_is_fresh_and_deterministic():
    first = export_contract()
    first["request"]["properties"].clear()
    second = export_contract()
    assert "messages" in second["request"]["properties"]
    assert second == export_contract()


@pytest.mark.parametrize("annotations", [None, [], [{"provider": "opaque"}]])
def test_export_preserves_nullable_annotations_and_replacement_policy(annotations):
    response = {"choices": [{"message": {"role": "assistant", "content": "answer", "annotations": annotations}}]}
    Draft202012Validator(export_contract()["response"]).validate(response)
    projection = CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(response)
    target = projection.locate_guarded_message(response)
    assert target.allows_replacement is (not bool(annotations))


def test_cli_check_and_regenerate(tmp_path):
    checked = subprocess.run([sys.executable, "-m", MODULE, "--check", str(EXPORTED)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    output = tmp_path / "exported.guard.yaml"
    generated = subprocess.run([sys.executable, "-m", MODULE, "--output", str(output)], capture_output=True, text=True)
    assert generated.returncode == 0, generated.stderr
    assert output.read_bytes() == EXPORTED.read_bytes()
    output.write_text("changed\n", encoding="utf-8")
    mismatch = subprocess.run([sys.executable, "-m", MODULE, "--check", str(output)], capture_output=True, text=True)
    assert mismatch.returncode == 1
    assert "Export differs" in mismatch.stderr
