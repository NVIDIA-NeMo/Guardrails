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

from typing import Literal

import pytest
from pydantic import BaseModel, RootModel, ValidationError

from nemoguardrails.server.experimental.provider.payload import (
    GuardedArraySelection,
    GuardedMessageTarget,
    GuardedRequestModel,
    GuardedTextAlternatives,
    GuardedTextLocation,
    PayloadCapabilityProfile,
    PayloadProjectionContract,
    ProjectionFieldCoverage,
    ProjectionModelContract,
    guarded_schema_error,
    validate_payload_projection_contract,
)


class TextContent(RootModel[str]):
    """Represent one scalar text encoding used by the test projection."""

    pass


class TextBlock(BaseModel):
    """Represent one discriminated text block used by the test projection."""

    type: Literal["text"]
    text: str


class BlockContent(RootModel[list[TextBlock]]):
    """Represent an array-based text encoding used by the test projection."""

    pass


class Message(BaseModel):
    """Represent a message with either supported test content encoding."""

    content: TextContent | BlockContent


class RequestProjection(BaseModel):
    """Represent the test request containing one guarded message."""

    messages: list[Message]


def request_locations():
    """Declare guarded text locations for the two test representations."""
    return GuardedTextAlternatives(
        locations=(
            GuardedTextLocation(
                role="user",
                object_path=("messages", 0),
                member="content",
                allows_replacement=False,
            ),
            GuardedTextLocation(
                role="user",
                object_path=("messages", 0, "content", 0),
                member="text",
                allows_replacement=False,
            ),
        )
    )


@pytest.mark.parametrize(
    ("projection", "payload"),
    [
        (
            RequestProjection(messages=[Message(content=TextContent("hello"))]),
            {"messages": [{"content": "hello"}]},
        ),
        (
            RequestProjection(messages=[Message(content=BlockContent([TextBlock(type="text", text="hello")]))]),
            {"messages": [{"content": [{"type": "text", "text": "hello"}]}]},
        ),
    ],
)
def test_guarded_text_alternatives_select_one_root_model_representation(projection, payload):
    """Alternative locations select the one representation accepted by the projection."""
    target = request_locations().locate(projection, payload)

    assert target.message.content == "hello"


def test_guarded_array_selection_requires_exactly_one_discriminator_match():
    """Array selection fails unless exactly one item matches its discriminator."""
    selection = GuardedArraySelection(match=(("type", "text"),), cardinality="exactly_one")

    assert selection.select([{"type": "metadata"}, {"type": "text", "text": "answer"}]) == {
        "type": "text",
        "text": "answer",
    }
    with pytest.raises(ValueError, match="exactly one"):
        selection.select([{"type": "metadata"}])
    with pytest.raises(ValueError, match="exactly one"):
        selection.select([{"type": "text"}, {"type": "text"}])


@pytest.mark.parametrize(
    ("expected", "received"),
    [
        (True, 1),
        (1, True),
        (False, 0),
        (0, False),
    ],
)
def test_guarded_array_selection_distinguishes_boolean_and_integer_discriminators(expected, received):
    """Boolean and integer discriminators do not match across JSON scalar types."""
    selection = GuardedArraySelection(match=(("value", expected),), cardinality="exactly_one")

    with pytest.raises(ValueError, match="exactly one"):
        selection.select([{"value": received}])


def _coverage(**fields):
    return ProjectionFieldCoverage(**{name: frozenset(value) for name, value in fields.items()})


class _Content(BaseModel):
    text: str


class _Root(BaseModel):
    items: list[_Content]


def test_projection_contract_metadata_rejects_inconsistent_declarations():
    """Coverage and contract metadata reject ambiguous or empty declarations."""
    with pytest.raises(ValueError, match="multiple classifications"):
        _coverage(guarded_fields={"text"}, opaque_fields={"text"})
    with pytest.raises(ValueError, match="non-empty"):
        ProjectionModelContract(model=_Content, coverage=_coverage(), source_schema=" ")
    with pytest.raises(ValueError, match="identifier"):
        PayloadProjectionContract(" ", "request", PayloadCapabilityProfile.SINGLE_TEXT_V1, _coverage())
    entry = ProjectionModelContract(model=_Content, coverage=_coverage(guarded_fields={"text"}))
    with pytest.raises(ValueError, match="each content model once"):
        PayloadProjectionContract(
            "test", "request", PayloadCapabilityProfile.SINGLE_TEXT_V1, _coverage(), content_models=(entry, entry)
        )


def test_projection_contract_validation_reports_each_mismatch():
    """A payload model must carry a contract for its direction that covers every field."""

    def bound(contract):
        model = type("Bound", (_Root,), {})
        model.projection_contract = contract
        return model

    def contract(root, content):
        return PayloadProjectionContract(
            "test",
            "request",
            PayloadCapabilityProfile.SINGLE_TEXT_V1,
            root,
            content_models=(ProjectionModelContract(model=_Content, coverage=content),),
        )

    covered = contract(_coverage(guarded_fields={"items"}), _coverage(guarded_fields={"text"}))
    assert validate_payload_projection_contract(bound(covered), "request") is covered
    with pytest.raises(ValueError, match="must declare a projection contract"):
        validate_payload_projection_contract(_Root, "request")
    with pytest.raises(ValueError, match="expected 'response'"):
        validate_payload_projection_contract(bound(covered), "response")
    with pytest.raises(ValueError, match="omits root fields"):
        validate_payload_projection_contract(
            bound(contract(_coverage(), _coverage(guarded_fields={"text"}))), "request"
        )
    with pytest.raises(ValueError, match="omits _Content fields"):
        validate_payload_projection_contract(
            bound(contract(_coverage(guarded_fields={"items"}), _coverage())), "request"
        )


def test_guarded_schema_error_describes_location_without_echoing_input():
    """Projection errors name the failing location but never the provider value."""
    with pytest.raises(ValidationError) as nested:
        _Root.model_validate({"items": [{"text": 1}]})
    with pytest.raises(ValidationError) as root:
        _Content.model_validate("secret")

    assert guarded_schema_error(nested.value, "request") == (
        "The request does not match the guarded content schema at items.0.text: Input should be a valid string."
    )
    message = guarded_schema_error(root.value, "request")
    assert message.startswith("The request does not match the guarded content schema: ")
    assert "secret" not in message


def test_message_target_reads_and_replaces_only_permitted_text():
    """Targets expose string text and replace it only when the contract allows."""
    payload = {"content": "hello"}
    target = GuardedMessageTarget("user", payload, "content", allows_replacement=True)
    target.replace_content("redacted")
    assert payload == {"content": "redacted"}
    with pytest.raises(ValueError, match="does not support replacement"):
        GuardedMessageTarget("user", payload, "content", allows_replacement=False).replace_content("x")
    with pytest.raises(TypeError, match="must be a string"):
        GuardedMessageTarget("user", {"content": None}, "content", allows_replacement=False).message


class _Block(BaseModel):
    type: str
    text: str | None = None


class _Blocks(BaseModel):
    blocks: list[_Block]


_SELECT_TEXT = GuardedArraySelection(match=(("type", "text"),), cardinality="exactly_one")


def test_text_location_follows_array_selections_in_payloads_and_projections():
    """Discriminated array segments resolve in both the raw payload and the projection."""
    location = GuardedTextLocation("user", ("blocks", _SELECT_TEXT), "text", allows_replacement=False)
    payload = {"blocks": [{"type": "image"}, {"type": "text", "text": "hello"}]}
    projection = _Blocks.model_validate(payload)

    assert location.locate(payload).message.content == "hello"
    location.validate_projection(projection)
    assert location.matches_projection(projection) is True
    assert location.matches_projection(_Blocks(blocks=[_Block(type="image")])) is False


def test_text_location_rejects_paths_that_do_not_resolve_to_text():
    """Location mismatches fail instead of selecting a different provider member."""
    location = GuardedTextLocation("user", ("blocks", 0), "text", allows_replacement=False)
    with pytest.raises(TypeError, match="does not match the provider payload"):
        location.locate({"blocks": {"text": "hello"}})
    with pytest.raises(TypeError, match="must resolve to a provider object"):
        location.locate({"blocks": ["hello"]})
    with pytest.raises(ValueError, match="does not match the provider projection"):
        GuardedTextLocation("user", ("items", "text"), "text", allows_replacement=False).validate_projection(
            _Root(items=[])
        )
    with pytest.raises(ValueError, match="does not resolve to projected text"):
        GuardedTextLocation("user", ("items",), "text", allows_replacement=False).validate_projection(_Root(items=[]))
    with pytest.raises(ValueError, match="does not resolve to projected text"):
        location.validate_projection(_Blocks(blocks=[_Block(type="text")]))
    GuardedTextLocation("user", ("blocks", 0), "text", allows_replacement=False, allows_empty=True).validate_projection(
        _Blocks(blocks=[_Block(type="text")])
    )
    assert (
        GuardedTextLocation("user", (1,), "text", allows_replacement=False).matches_projection(_Root(items=[])) is False
    )


def test_text_alternatives_require_exactly_one_matching_representation():
    """Alternatives validate only when one representation matches the projection."""
    alternatives = request_locations()
    alternatives.validate_projection(RequestProjection(messages=[Message(content=TextContent("hello"))]))
    with pytest.raises(ValueError, match="exactly one matching representation"):
        alternatives.validate_projection(RequestProjection(messages=[]))


class _AlternativeRequest(GuardedRequestModel):
    messages: list[Message]
    stream: bool = False
    guarded_text_location = request_locations()
    stream_selector_field = "stream"


class _UnboundRequest(GuardedRequestModel):
    messages: list[Message]


def test_payload_models_report_and_use_their_bindings():
    """Payload models expose whether they declare text and response-mode bindings."""
    payload = {"messages": [{"content": "hello"}], "stream": True}
    request = _AlternativeRequest.model_validate(payload)

    assert _AlternativeRequest.has_guarded_text_binding() is True
    assert _AlternativeRequest.has_response_mode_binding() is True
    assert request.locate_guarded_message(payload).message.content == "hello"
    assert request.streams_response is True

    unbound = _UnboundRequest.model_validate(payload)
    assert _UnboundRequest.has_guarded_text_binding() is False
    assert _UnboundRequest.has_response_mode_binding() is False
    with pytest.raises(TypeError, match="does not declare a guarded text location"):
        unbound.locate_guarded_message(payload)
    with pytest.raises(TypeError, match="does not declare a stream selector field"):
        unbound.streams_response
