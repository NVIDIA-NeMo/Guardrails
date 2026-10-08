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

"""Exercise provider-neutral buffered export and its trusted local CLI."""

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import pytest
from jsonschema import Draft202012Validator

from nemoguardrails.server.experimental.provider.contract_export import _load_endpoint, export_guard_contract
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.payload import GuardedPayloadModel, GuardedRequestModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    PolicyModel,
    guarded,
    payload_contract,
    text_location,
)

MODULE = "nemoguardrails.server.experimental.provider.contract_export"
ROOT = Path(__file__).parents[3]


@pytest.fixture
def example_endpoint() -> GuardedJsonEndpoint:
    """Bind a non-Chat shape with a fixed buffered response mode."""

    class Request(PolicyModel, GuardedRequestModel):
        prompt: Annotated[str, guarded("user", replaceable=True, min_length=1)]

        @property
        def streams_response(self) -> bool:
            """This example has no request flag and always returns buffered JSON."""
            return False

    class Response(PolicyModel, GuardedPayloadModel):
        answer: Annotated[str, guarded("assistant", replaceable=False, min_length=2)]

    Request.projection_contract = payload_contract(Request, projection_id="example.request", direction="request")
    Request.guarded_text_location = text_location(Request)
    Response.projection_contract = payload_contract(Response, projection_id="example.response", direction="response")
    Response.guarded_text_location = text_location(Response)
    return GuardedJsonEndpoint(
        route_path="/v2/text",
        operation_name="example.text",
        operation="Example text",
        unsupported_request_code="unsupported_example_request",
        unsupported_response_code="unsupported_example_response",
        guarded_request_model=Request,
        guarded_response_model=Response,
    )


def test_export_supports_another_endpoint_without_a_stream_selector(example_endpoint):
    """The exporter derives both payloads and labels from the supplied endpoint."""
    contract = export_guard_contract(example_endpoint, operation_id="createExampleText")
    schema = json.loads(
        (ROOT / "nemoguardrails/server/experimental/contracts/guard-contract.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(contract)

    assert contract["operationId"] == "createExampleText"
    assert "name" not in contract["integration"]
    assert contract["integration"]["endpoint"] == {
        "route_path": "/v2/text",
        "operation_label": "Example text",
        "unsupported_request_code": "unsupported_example_request",
        "unsupported_response_code": "unsupported_example_response",
    }
    assert contract["request"]["title"] == "example.request"
    assert set(contract["request"]["properties"]) == {"prompt"}
    assert contract["response"]["title"] == "example.response"
    assert set(contract["response"]["properties"]) == {"answer"}
    assert "stream_selector_field" not in contract["request"][EXTENSION]
    assert "stream" not in contract


def test_export_follows_changed_endpoint_metadata(example_endpoint):
    """An endpoint variant changes the export without a provider-specific wrapper."""
    endpoint = replace(
        example_endpoint,
        route_path="/v3/text",
        operation_paths=(),
        operation="Different text operation",
        unsupported_response_code="different_response_error",
    )
    contract = export_guard_contract(endpoint, operation_id="otherOperation", name="other_text")
    assert contract["integration"]["name"] == "other_text"
    assert contract["integration"]["endpoint"]["route_path"] == "/v3/text"
    assert contract["integration"]["endpoint"]["operation_label"] == "Different text operation"
    assert contract["integration"]["endpoint"]["unsupported_response_code"] == "different_response_error"


@pytest.mark.parametrize("direction", ["request", "response"])
def test_export_requires_policy_annotated_models(example_endpoint, direction):
    """Valid handwritten runtime bindings alone do not promise exportable policy."""

    class PlainRequest(GuardedRequestModel):
        prompt: str

        @property
        def streams_response(self) -> bool:
            """Keep the example's fixed buffered response mode."""
            return False

    class PlainResponse(GuardedPayloadModel):
        answer: str

    if direction == "request":
        PlainRequest.projection_contract = example_endpoint.guarded_request_model.projection_contract
        PlainRequest.guarded_text_location = example_endpoint.guarded_request_model.guarded_text_location
        endpoint = replace(example_endpoint, guarded_request_model=PlainRequest)
    else:
        PlainResponse.projection_contract = example_endpoint.guarded_response_model.projection_contract
        PlainResponse.guarded_text_location = example_endpoint.guarded_response_model.guarded_text_location
        endpoint = replace(example_endpoint, guarded_response_model=PlainResponse)

    with pytest.raises(TypeError, match="policy-annotated"):
        export_guard_contract(endpoint, operation_id="example")


@pytest.mark.parametrize(
    ("operation_id", "name", "message"),
    [("", None, "must not be blank"), ("  ", None, "must not be blank"), ("example", "Bad.Name", "snake case")],
)
def test_export_rejects_invalid_document_identity(example_endpoint, operation_id, name, message):
    """Reject malformed operation metadata independently of serialization."""
    with pytest.raises(ValueError, match=message):
        export_guard_contract(example_endpoint, operation_id=operation_id, name=name)


@pytest.mark.parametrize("reference", ["missing_separator", ":value", "pathlib:Path()", "pathlib:Path:extra"])
def test_endpoint_reference_rejects_expressions_before_import(reference, monkeypatch):
    """Only a module and one attribute are allowed; references are never evaluated."""

    def unexpected_import(*args):
        raise AssertionError("Malformed references must fail before imports")

    monkeypatch.setattr(
        "nemoguardrails.server.experimental.provider.contract_export.importlib.import_module", unexpected_import
    )
    with pytest.raises(argparse.ArgumentTypeError, match="module:attribute"):
        _load_endpoint(reference)


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("no_such_guard_contract_provider:endpoint", "Cannot load endpoint"),
        ("pathlib:NO_SUCH_ENDPOINT", "Cannot load endpoint"),
        ("pathlib:Path", "not a GuardedJsonEndpoint"),
    ],
)
def test_endpoint_reference_reports_missing_or_wrong_objects(reference, message):
    """The CLI requires an existing endpoint instance, not a class or factory."""
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        _load_endpoint(reference)


def test_export_module_import_does_not_load_provider_integrations():
    """Importing shared export machinery does not discover or import providers."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import " + MODULE + "; "
            "assert not any(name.startswith('nemoguardrails.server.experimental.providers.') for name in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
