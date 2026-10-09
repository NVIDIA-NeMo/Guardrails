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
from nemoguardrails.server.experimental.provider.payload import (
    GuardedPayloadModel,
    GuardedRequestModel,
    GuardedTextLocation,
)
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    PolicyModel,
    constrained,
    guarded,
    opaque,
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
        provider_operation_id="createExampleText",
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
    contract = export_guard_contract(example_endpoint)
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


@pytest.mark.parametrize("direction", ["request", "response"])
@pytest.mark.parametrize("mutation", ["replacement", "coverage", "custom_extraction"])
def test_export_rejects_binding_policy_drift(example_endpoint, direction, mutation, monkeypatch):
    """Export checks the actual binding, not only whether every field is listed."""
    model = getattr(example_endpoint, f"guarded_{direction}_model")
    if mutation == "replacement":
        location = model.guarded_text_location
        monkeypatch.setattr(
            model, "guarded_text_location", replace(location, allows_replacement=not location.allows_replacement)
        )
    elif mutation == "coverage":
        contract = model.projection_contract
        coverage = replace(contract.root, guarded_fields=frozenset(), constrained_fields=contract.root.guarded_fields)
        monkeypatch.setattr(model, "projection_contract", replace(contract, root=coverage))
    else:

        def custom_extraction(self, payload):
            return self.guarded_text_location.locate(payload)

        monkeypatch.setattr(model, "locate_guarded_message", custom_extraction)
    with pytest.raises(ValueError, match="differs from field policy|Custom text extraction"):
        export_guard_contract(example_endpoint)


def test_export_rejects_a_binding_that_inspects_an_opaque_field(example_endpoint):
    """A valid runtime locator cannot silently contradict the annotated subject."""

    class Request(PolicyModel, GuardedRequestModel):
        prompt: Annotated[str, guarded("user")]
        audit: Annotated[str, opaque()]

        @property
        def streams_response(self) -> bool:
            return False

    Request.projection_contract = payload_contract(Request, projection_id="example.request", direction="request")
    Request.guarded_text_location = GuardedTextLocation("user", (), "audit", allows_replacement=False)
    endpoint = replace(example_endpoint, guarded_request_model=Request)
    payload = {"prompt": "user content", "audit": "opaque label"}
    assert Request.validate_payload(payload).locate_guarded_message(payload).message.content == "opaque label"
    with pytest.raises(ValueError, match="text binding differs from field policy"):
        export_guard_contract(endpoint)


@pytest.mark.parametrize("selector", ["missing", "prompt"])
def test_export_rejects_non_boolean_stream_selectors(example_endpoint, selector, monkeypatch):
    """A declared stream selector must name a boolean field, not arbitrary metadata."""
    monkeypatch.setattr(example_endpoint.guarded_request_model, "stream_selector_field", selector)
    with pytest.raises(ValueError, match="boolean field"):
        export_guard_contract(example_endpoint)


def test_export_rejects_custom_selection_with_a_declared_stream_selector(example_endpoint):
    """An exported boolean selector must describe the runtime selection rule."""

    class Request(PolicyModel, GuardedRequestModel):
        prompt: Annotated[str, guarded("user")]
        stream: Annotated[bool, constrained()] = False

        @property
        def streams_response(self) -> bool:
            return not self.stream

    Request.projection_contract = payload_contract(Request, projection_id="example.request", direction="request")
    Request.guarded_text_location = text_location(Request)
    Request.stream_selector_field = "stream"
    with pytest.raises(ValueError, match="Custom response-mode selection"):
        export_guard_contract(replace(example_endpoint, guarded_request_model=Request))


def test_export_follows_changed_endpoint_metadata(example_endpoint):
    """An endpoint variant changes the export without a provider-specific wrapper."""
    endpoint = replace(
        example_endpoint,
        provider_operation_id="otherOperation",
        contract_name="other_text",
        route_path="/v3/text",
        operation_paths=(),
        operation="Different text operation",
        unsupported_response_code="different_response_error",
    )
    contract = export_guard_contract(endpoint)
    assert contract["operationId"] == "otherOperation"
    assert contract["integration"]["name"] == "other_text"
    assert contract["integration"]["endpoint"]["route_path"] == "/v3/text"
    assert contract["integration"]["endpoint"]["operation_label"] == "Different text operation"
    assert contract["integration"]["endpoint"]["unsupported_response_code"] == "different_response_error"


def test_http_schema_does_not_require_exportable_unions(example_endpoint):
    """Policy-aware HTTP schemas preserve unions outside the contract format."""
    from nemoguardrails.server.experimental._guarded_proxy import create_buffered_guarded_http_operation
    from nemoguardrails.server.experimental.provider.projection_policy import constrained
    from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING

    class UnionRequest(PolicyModel, GuardedRequestModel):
        prompt: Annotated[str, guarded("user")]
        option: Annotated[str | int, constrained()]

        @property
        def streams_response(self) -> bool:
            return False

    UnionRequest.projection_contract = payload_contract(
        UnionRequest, projection_id="example.request", direction="request"
    )
    UnionRequest.guarded_text_location = text_location(UnionRequest)
    endpoint = replace(example_endpoint, guarded_request_model=UnionRequest)

    with pytest.raises(ValueError, match="disjoint nullable"):
        export_guard_contract(endpoint)

    operation = create_buffered_guarded_http_operation(endpoint, OPENAI_ERROR_MAPPING)
    schema = operation.openapi_extra["requestBody"]["content"]["application/json"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["option"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    for option in ("value", 1):
        payload = {"prompt": "q", "option": option}
        Draft202012Validator(schema).validate(payload)
        UnionRequest.validate_payload(payload)


def test_http_schema_supports_models_without_policy_annotations(example_endpoint):
    """Handwritten bindings need not support contract export to construct a route."""
    from nemoguardrails.server.experimental._guarded_proxy import create_buffered_guarded_http_operation
    from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING

    class PlainRequest(GuardedRequestModel):
        prompt: str

        @property
        def streams_response(self) -> bool:
            return False

    PlainRequest.projection_contract = example_endpoint.guarded_request_model.projection_contract
    PlainRequest.guarded_text_location = example_endpoint.guarded_request_model.guarded_text_location
    endpoint = replace(example_endpoint, guarded_request_model=PlainRequest)

    operation = create_buffered_guarded_http_operation(endpoint, OPENAI_ERROR_MAPPING)

    assert operation.openapi_extra["requestBody"]["content"]["application/json"]["schema"] == (
        PlainRequest.model_json_schema()
    )


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
        export_guard_contract(endpoint)


@pytest.mark.parametrize(
    ("operation_id", "name", "message"),
    [("", None, "must not be blank"), ("  ", None, "must not be blank"), ("example", "Bad.Name", "snake case")],
)
def test_endpoint_rejects_invalid_document_identity(example_endpoint, operation_id, name, message):
    """Reject malformed operation metadata independently of serialization."""
    with pytest.raises(ValueError, match=message):
        replace(example_endpoint, provider_operation_id=operation_id, contract_name=name)


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


def test_export_imports_endpoint_without_http_execution_dependencies():
    source = """
import importlib.abc
import sys

class BlockHttpExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastapi" or fullname.startswith("fastapi.") or fullname == "starlette.routing" or fullname == "nemoguardrails.server.experimental._http_kernel":
            raise AssertionError(f"Export must not import {fullname}")

sys.meta_path.insert(0, BlockHttpExecution())
from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
assert export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)["operationId"] == "createChatCompletion"
"""
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
