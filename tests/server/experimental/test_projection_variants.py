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

"""Keep heterogeneous text selection, runtime bindings, and export aligned."""

from dataclasses import replace
from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import StrictBool, ValidationError

from nemoguardrails.server.experimental.provider.payload import (
    GuardedArraySelection,
    GuardedPayloadModel,
    GuardedRequestModel,
)
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    ArrayVariant,
    ObjectPolicy,
    PolicyModel,
    constrained,
    export_payload_schema,
    guarded,
    payload_contract,
    text_location,
)
from nemoguardrails.server.experimental.provider.stream_classifier import StreamArraySelection
from nemoguardrails.server.experimental.provider.stream_policy import stream_text_path


class MetadataItem(PolicyModel):
    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        variant=ArrayVariant(match=(("type", "metadata"),), cardinality="any"), opaque=("id",)
    )
    type: Annotated[Literal["metadata"], constrained()]


class MessageItem(PolicyModel):
    policy: ClassVar[ObjectPolicy] = ObjectPolicy(
        variant=ArrayVariant(match=(("type", "message"),), cardinality="exactly_one")
    )
    type: Annotated[Literal["message"], constrained()]
    text: Annotated[str, guarded("assistant", replaceable=False, min_length=1)]


class Response(PolicyModel, GuardedPayloadModel):
    output: Annotated[list[MetadataItem | MessageItem], guarded()]


Response.guarded_text_location = text_location(Response)
Response.projection_contract = payload_contract(Response, projection_id="example.response", direction="response")


@pytest.mark.parametrize(
    "items",
    [
        [{"type": "message", "text": "answer"}],
        [{"type": "metadata", "id": "a"}, {"type": "message", "text": "answer"}],
        [{"type": "message", "text": "answer"}, {"type": "metadata", "id": "b"}],
    ],
)
def test_buffered_variant_binding_selects_original_text(items):
    document = {"output": items}
    projection = Response.validate_payload(document)
    target = projection.locate_guarded_message(document)
    assert target.message.content == "answer"
    assert target.allows_replacement is False
    assert Response.guarded_text_location.object_path == (
        "output",
        GuardedArraySelection((("type", "message"),), "exactly_one"),
    )


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"type": "metadata"}],
        [{"type": "message", "text": "a"}, {"type": "message", "text": "b"}],
        [{"type": "unknown", "text": "hidden"}],
    ],
)
def test_buffered_variant_binding_rejects_missing_ambiguous_or_unreviewed_targets(items):
    with pytest.raises(ValidationError):
        Response.validate_payload({"output": items})


def test_variant_export_preserves_disjoint_branches_and_cardinality():
    exported = export_payload_schema(Response, projection_id="example.response")
    variants = exported["properties"]["output"]["items"]["oneOf"]
    assert [entry[EXTENSION]["variant"] for entry in variants] == [
        {"match": {"type": "metadata"}, "cardinality": "any"},
        {"match": {"type": "message"}, "cardinality": "exactly_one"},
    ]
    assert variants[1]["properties"]["text"][EXTENSION]["subject"]["role"] == "assistant"


def test_stream_variant_path_allows_initially_missing_message_but_not_two():
    class SnapshotMessage(MessageItem):
        policy: ClassVar[ObjectPolicy] = replace(
            MessageItem.policy, variant=ArrayVariant((("type", "message"),), "at_most_one")
        )

    class Snapshot(PolicyModel):
        output: Annotated[list[MetadataItem | SnapshotMessage], guarded()]

    selector = StreamArraySelection((("type", "message"),), "at_most_one")
    assert stream_text_path(Snapshot) == ("output", selector, "text")
    assert selector.select([]) is None
    with pytest.raises(ValueError):
        selector.select([SnapshotMessage(type="message", text="a"), SnapshotMessage(type="message", text="b")])
    with pytest.raises(ValueError, match="exactly_one"):
        text_location(Snapshot)


@pytest.mark.parametrize("match", [(), (("type", "a"), ("type", "b")), (("type", None),)])
def test_variant_match_requires_unique_named_scalar_discriminators(match):
    with pytest.raises(ValueError):
        ArrayVariant(match, "exactly_one")


@pytest.mark.parametrize("annotation,default", [(str, ...), (Literal["other"], ...), (Literal["message"], "message")])
def test_variant_discriminators_must_be_required_matching_literals(annotation, default):
    with pytest.raises(ValueError, match="required.*Literal"):

        class Invalid(PolicyModel):
            policy: ClassVar[ObjectPolicy] = ObjectPolicy(variant=ArrayVariant((("type", "message"),), "exactly_one"))
            type: Annotated[annotation, constrained()] = default


def test_export_still_rejects_overlapping_object_unions():
    class Overlap(PolicyModel):
        type: Annotated[Literal["message"], constrained()]
        text: Annotated[str, guarded("assistant")]

    class Invalid(PolicyModel):
        output: Annotated[list[MessageItem | Overlap], guarded()]

    with pytest.raises(ValueError, match="disjoint"):
        export_payload_schema(Invalid, projection_id="invalid")
    with pytest.raises(ValueError, match="variant"):
        text_location(Invalid)


def test_complete_endpoint_export_checks_variant_bindings():
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator
    from pydantic import StrictBool

    from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
    from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
    from nemoguardrails.server.experimental.provider.payload import GuardedRequestModel
    from nemoguardrails.server.experimental.provider.stream import (
        StreamBinding,
        StreamCapabilityProfile,
        StreamEventRole,
        StreamProjectionContract,
    )
    from nemoguardrails.server.experimental.provider.stream_classifier import (
        StreamClassifierDefinition,
        StreamEventRule,
    )
    from nemoguardrails.server.experimental.provider.stream_policy import (
        build_policy_stream_classifier,
        stream_shape_coverage,
    )

    class Request(PolicyModel, GuardedRequestModel):
        input: Annotated[str, guarded("user")]
        stream: Annotated[StrictBool, constrained()] = False
        stream_selector_field = "stream"

    Request.guarded_text_location = text_location(Request)
    Request.projection_contract = payload_contract(Request, projection_id="example.request", direction="request")

    class Snapshot(PolicyModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="ExampleSnapshot")
        output: Annotated[list[MetadataItem | MessageItem], guarded()]

    rules = (
        StreamEventRule(
            shape="snapshot",
            model=Snapshot,
            role=StreamEventRole.TEXT_SNAPSHOT,
            required_fields=frozenset({"output"}),
            text_path=stream_text_path(Snapshot),
        ),
    )
    classifier = build_policy_stream_classifier(
        StreamClassifierDefinition(
            subject="example.stream",
            contract=StreamProjectionContract(
                "example.stream", StreamCapabilityProfile.SINGLE_TEXT_DELTA_V1, stream_shape_coverage(rules)
            ),
            rules=rules,
        )
    )

    def hooks():
        raise AssertionError("Export must not construct runtime hooks")

    endpoint = GuardedJsonEndpoint(
        provider_operation_id="createExample",
        route_path="/example",
        operation_name="example.create",
        operation="Example",
        unsupported_request_code="unsupported_request",
        unsupported_response_code="unsupported_response",
        guarded_request_model=Request,
        guarded_response_model=Response,
        stream=StreamBinding(classifier, hooks),
    )
    exported = export_guard_contract(endpoint)
    schema_path = Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts/guard-contract.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(exported)
    assert (
        exported["stream"]["oneOf"][0]["properties"]["output"]["items"]["oneOf"][1][EXTENSION]["variant"]["cardinality"]
        == "exactly_one"
    )
    wrong_rule = replace(rules[0], text_path=("output", 0, "text"))
    wrong_classifier = replace(classifier, rules=(wrong_rule,))
    with pytest.raises(ValueError, match="text binding differs"):
        export_guard_contract(replace(endpoint, stream=StreamBinding(wrong_classifier, hooks)))


def test_unselected_variants_cannot_hide_text():
    class Hidden(MetadataItem):
        text: Annotated[str, guarded("assistant")]

    class Invalid(PolicyModel):
        output: Annotated[list[Hidden | MessageItem], guarded()]

    for derive in (text_location, stream_text_path):
        with pytest.raises(ValueError, match="hide guarded"):
            derive(Invalid)


@pytest.fixture(params=[(Literal[1], 1), (Literal[True], True), (Literal[False], False)])
def scalar_variant_request(request):
    annotation, expected = request.param

    class ScalarMessage(PolicyModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(
            variant=ArrayVariant((("type", "message"), ("kind", expected), ("revision", 2)), "exactly_one")
        )
        type: Annotated[Literal["message"], constrained()]
        kind: Annotated[annotation, constrained()]
        revision: Annotated[Literal[2], constrained()]
        text: Annotated[str, guarded("user", replaceable=False, min_length=1)]

    class Request(PolicyModel, GuardedRequestModel):
        output: Annotated[list[MetadataItem | ScalarMessage], guarded()]
        stream: Annotated[StrictBool, constrained()] = False
        stream_selector_field = "stream"

    Request.guarded_text_location = text_location(Request)
    Request.projection_contract = payload_contract(Request, projection_id="scalar.request", direction="request")
    return Request, expected


def test_scalar_variant_validation_preserves_original_target(scalar_variant_request):
    model, expected = scalar_variant_request
    item = {"type": "message", "kind": expected, "revision": 2, "text": "answer"}
    document = {"output": [{"type": "metadata"}, item]}
    projection = model.validate_payload(document)
    target = projection.locate_guarded_message(document)
    assert target.message.content == "answer"
    item["text"] = "updated"
    assert target.message.content == "updated"
    assert type(item["kind"]) is type(expected)
    assert model.model_validate(projection) is projection


def test_scalar_variant_rejects_coercion_before_extraction(scalar_variant_request):
    model, expected = scalar_variant_request
    for value in (True, False, 0, 1, 0.0, 1.0, "1", "true", None):
        if type(value) is type(expected) and value == expected:
            continue
        document = {"output": [{"type": "message", "kind": value, "revision": 2, "text": "answer"}]}
        with pytest.raises(ValidationError):
            model.validate_payload(document)


@pytest.mark.parametrize("revision", [2.0, "2", True, None])
def test_variant_checks_each_discriminator(scalar_variant_request, revision):
    model, expected = scalar_variant_request
    with pytest.raises(ValidationError):
        model.validate_payload(
            {"output": [{"type": "message", "kind": expected, "revision": revision, "text": "answer"}]}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, 1.0])
async def test_coerced_variant_request_returns_unsupported_shape(scalar_variant_request, value):
    import httpx
    from fastapi import FastAPI

    from nemoguardrails.server.experimental._content_checker import ContentInspectionPolicy
    from nemoguardrails.server.experimental._guarded_proxy import create_guarded_http_operation
    from nemoguardrails.server.experimental._http_kernel import create_http_proxy_router
    from nemoguardrails.server.experimental.providers.openai.errors import OPENAI_ERROR_MAPPING
    from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT
    from tests.server.experimental.checkers import StaticChecker

    model, expected = scalar_variant_request
    if type(value) is type(expected) and value == expected:
        value = 1
    endpoint = replace(RESPONSES_ENDPOINT, guarded_request_model=model)
    inspector = StaticChecker(ContentInspectionPolicy(True, True))

    async def dispatch(request):
        raise AssertionError("Invalid discriminator must not reach the provider")

    app = FastAPI()
    app.include_router(
        create_http_proxy_router(
            operations=(
                create_guarded_http_operation(
                    endpoint,
                    OPENAI_ERROR_MAPPING,
                    max_stream_event_bytes=1024,
                    max_pending_stream_bytes=1024,
                ),
            ),
            checker=inspector,
            dispatch=dispatch,
            render_outcome=OPENAI_ERROR_MAPPING.renderer,
        )
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy.test") as client:
        response = await client.post(
            "/v1/responses",
            json={"output": [{"type": "message", "kind": value, "revision": 2, "text": "answer"}]},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_responses_shape"
    assert inspector.calls == []
