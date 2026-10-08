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

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import CONTRACT_VERSION, EXTENSION
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT

ROOT = Path(__file__).parents[3]
CONTRACTS = ROOT / "nemoguardrails/server/experimental/contracts"
EXPORTED = CONTRACTS / "openai/_generated/chat-completions.buffered.guard.yaml"
MODULE = "nemoguardrails.server.experimental.provider.contract_export"
ENDPOINT = "nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint:CHAT_COMPLETIONS_ENDPOINT"
CLI = [sys.executable, "-m", MODULE, ENDPOINT, "--operation-id", "createChatCompletion", "--name", "chat_completions"]


def test_buffered_export_matches_checked_in_artifact_and_format():
    contract = export_guard_contract(
        CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions"
    )
    assert contract == yaml.safe_load(EXPORTED.read_text(encoding="utf-8"))
    schema = json.loads((CONTRACTS / "guard-contract.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(contract)
    assert contract["version"] == CONTRACT_VERSION == "1.0.0-alpha.1"
    assert "stream" not in contract
    assert "stream_hooks" not in contract["integration"]["endpoint"]


def test_export_uses_the_runtime_endpoint_metadata():
    exported = export_guard_contract(
        CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions"
    )
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


def test_export_does_not_read_files(monkeypatch):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Export must derive from Python, not read YAML")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert (
        export_guard_contract(CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions")[
            "operationId"
        ]
        == "createChatCompletion"
    )


def test_export_is_fresh_and_deterministic():
    first = export_guard_contract(
        CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions"
    )
    first["request"]["properties"].clear()
    second = export_guard_contract(
        CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions"
    )
    assert "messages" in second["request"]["properties"]
    assert second == export_guard_contract(
        CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions"
    )


@pytest.mark.parametrize("annotations", [None, [], [{"provider": "opaque"}]])
def test_export_preserves_nullable_annotations_and_replacement_policy(annotations):
    response = {"choices": [{"message": {"role": "assistant", "content": "answer", "annotations": annotations}}]}
    Draft202012Validator(
        export_guard_contract(CHAT_COMPLETIONS_ENDPOINT, operation_id="createChatCompletion", name="chat_completions")[
            "response"
        ]
    ).validate(response)
    projection = CHAT_COMPLETIONS_ENDPOINT.guarded_response_model.validate_payload(response)
    target = projection.locate_guarded_message(response)
    assert target.allows_replacement is (not bool(annotations))


def test_cli_check_and_regenerate(tmp_path):
    checked = subprocess.run([*CLI, "--check", str(EXPORTED)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    output = tmp_path / "exported.guard.yaml"
    generated = subprocess.run([*CLI, "--output", str(output)], capture_output=True, text=True)
    assert generated.returncode == 0, generated.stderr
    assert output.read_bytes() == EXPORTED.read_bytes()
    output.write_text("changed\n", encoding="utf-8")
    mismatch = subprocess.run([*CLI, "--check", str(output)], capture_output=True, text=True)
    assert mismatch.returncode == 1
    assert "Export differs" in mismatch.stderr
    assert output.read_text(encoding="utf-8") == "changed\n"


def test_cli_stdout_matches_checked_in_artifact():
    """The shared CLI preserves the deterministic default stdout mode."""
    completed = subprocess.run(CLI, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == EXPORTED.read_text(encoding="utf-8")


def test_cli_missing_check_does_not_create_an_artifact(tmp_path):
    """Check mode fails without writing a missing destination."""
    missing = tmp_path / "missing" / "contract.yaml"
    completed = subprocess.run([*CLI, "--check", str(missing)], capture_output=True, text=True)
    assert completed.returncode == 1
    assert "Export differs" in completed.stderr
    assert not missing.parent.exists()


@pytest.mark.parametrize("arguments", [["--operation-id", ""], ["--name", "Bad.Name"]])
def test_cli_invalid_identity_does_not_overwrite_an_artifact(tmp_path, arguments):
    """Bad metadata fails before an existing destination can be overwritten."""
    destination = tmp_path / "contract.yaml"
    destination.write_text("keep\n", encoding="utf-8")
    completed = subprocess.run([*CLI, *arguments, "--output", str(destination)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "error:" in completed.stderr
    assert destination.read_text(encoding="utf-8") == "keep\n"


def test_cli_rejects_wrong_endpoint_without_creating_an_artifact(tmp_path):
    """A module attribute must be an endpoint instance, not arbitrary Python data."""
    destination = tmp_path / "contract.yaml"
    completed = subprocess.run(
        [sys.executable, "-m", MODULE, "pathlib:Path", "--operation-id", "example", "--output", str(destination)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "not a GuardedJsonEndpoint" in completed.stderr
    assert not destination.exists()
