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

from pathlib import Path
from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import ConfigDict, Field, ValidationError

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    EXTENSION,
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    export_payload_schema,
    guarded,
    payload_contract,
    policy_json_schema,
    text_location,
)
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy


def test_changed_python_policy_changes_coverage_and_export():
    class Message(PolicyModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(source="Message", opaque=("provider_id",))
        text: Annotated[str, guarded("user", replaceable=False, min_length=2)]
        role: Annotated[Literal["user"], constrained()]
        tools: Annotated[None, disabled("core_capability.tool_content")] = None

    class Request(PolicyModel):
        messages: Annotated[list[Message], guarded(min_length=1, max_length=1)]

    contract = payload_contract(Request, projection_id="test.request", direction="request")
    assert contract.content_models[0].coverage.opaque_fields == frozenset({"provider_id"})
    assert text_location(Request).allows_replacement is False
    exported = export_payload_schema(Request, projection_id="test.request")
    message = exported["properties"]["messages"]["items"]
    assert message["properties"]["text"]["minLength"] == 2
    assert message[EXTENSION]["source"] == "#/components/schemas/Message"
    assert message["properties"]["tools"]["default"] is None


@pytest.mark.parametrize("extra", ["ignore", "forbid", None])
def test_policy_models_reject_conflicting_extra_configuration(extra):
    """Object policy must see all extras before deciding whether to accept them."""
    with pytest.raises(ValueError, match="requires extra='allow'"):

        class Invalid(PolicyModel):
            model_config = ConfigDict(extra=extra)
            policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("provider_id",))
            text: Annotated[str, guarded("user")]


@pytest.mark.parametrize("export", [False, True])
def test_object_policy_uses_model_identity_not_schema_titles(export, export_policy_validator):
    """Display titles and field titles cannot hide or swap object policies."""

    class Left(PolicyModel):
        model_config = ConfigDict(title="Shared display title")
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("left_metadata",))
        text: Annotated[str, guarded("user")]

    class Right(PolicyModel):
        model_config = ConfigDict(title="Shared display title")
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("right_metadata",))
        value: Annotated[str, constrained()]

    class Root(PolicyModel):
        model_config = ConfigDict(title="Readable request")
        left: Annotated[Left, guarded(), Field(title="Friendly field")]
        right: Annotated[Right, constrained()]

    schema = export_payload_schema(Root, projection_id="test.request") if export else policy_json_schema(Root)
    assert schema["additionalProperties"] is False
    assert schema[EXTENSION]["unknown_fields"] == "forbid"
    for field, expected in (("left", "left_metadata"), ("right", "right_metadata")):
        node = schema["properties"][field]
        assert node["additionalProperties"] is False
        assert node[EXTENSION]["reject_case_aliases"] is True
        assert expected in node["properties"]
    assert "right_metadata" not in schema["properties"]["left"]["properties"]
    payload = {"left": {"text": "q", "left_metadata": 1}, "right": {"value": "v", "right_metadata": 2}}
    validator = export_policy_validator(schema, UnknownContentFieldPolicy.FORBID)
    assert validator.is_valid(payload)
    Root.model_validate(payload)
    payload["left"]["unreviewed"] = "x"
    assert not validator.is_valid(payload)
    with pytest.raises(ValidationError):
        Root.model_validate(payload)


def test_missing_policy_fails_at_class_definition():
    with pytest.raises(ValueError, match="missing field policy"):

        class Invalid(PolicyModel):
            text: str


def test_export_rejects_overlapping_unions_instead_of_changing_their_meaning():
    class Invalid(PolicyModel):
        text: Annotated[str | Literal["special"], constrained()]

    with pytest.raises(ValueError, match="disjoint nullable"):
        export_payload_schema(Invalid, projection_id="test.request")


def test_missing_disabled_default_fails_at_class_definition():
    with pytest.raises(ValueError, match="optional and null-only"):

        class Invalid(PolicyModel):
            tools: Annotated[None, disabled("core_capability.tool_content")]


def test_opaque_overlap_fails_at_class_definition():
    with pytest.raises(ValueError, match="overlaps declared"):

        class Invalid(PolicyModel):
            policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("text",))
            text: Annotated[str, guarded("user")]


def test_duplicate_opaque_inventory_fails_at_class_definition():
    with pytest.raises(ValueError, match="duplicate opaque"):

        class Invalid(PolicyModel):
            policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("id", "id"))


@pytest.mark.parametrize("helper", [guarded, constrained])
def test_helpers_do_not_hide_defaults(helper):
    with pytest.raises(ValueError, match="defaults explicitly"):
        helper(default=False)


@pytest.mark.parametrize("helper", [guarded, constrained])
@pytest.mark.parametrize("constraint", ["ge", "le", "gt", "lt", "multiple_of", "strict"])
def test_helpers_reject_constraints_the_contract_cannot_express(helper, constraint):
    with pytest.raises(ValueError, match="not expressible"):
        helper(**{constraint: 1})


def test_policy_models_are_closed_unless_configurable_fields_are_allowed():
    class Message(PolicyModel, GuardedContentModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("model",), unknown_fields="configurable")
        content: Annotated[str, guarded("user")]

    class Root(PolicyModel, GuardedContentModel):
        content: Annotated[str, guarded("user")]

    allow = UnknownContentFieldPolicy.ALLOW
    Message.model_validate({"content": "q", "model": "m"})
    for model in (Message, Root):
        with pytest.raises(ValidationError, match="unreviewed fields are forbidden: unreviewed"):
            model.model_validate({"content": "q", "unreviewed": "x"})
    Message.validate_payload({"content": "q", "unreviewed": "x"}, unknown_content_fields=allow)
    with pytest.raises(ValidationError, match="unreviewed fields are forbidden"):
        Root.validate_payload({"content": "q", "unreviewed": "x"}, unknown_content_fields=allow)
    for variant in ({"Content": "x"}, {"MODEL": "x"}, {"cOnTeNt": "x"}):
        with pytest.raises(ValidationError, match="only by case"):
            Message.validate_payload({"content": "q", **variant}, unknown_content_fields=allow)


@pytest.mark.parametrize(
    "declare",
    [
        lambda: disabled("tools are off"),
        lambda: disabled(" "),
        lambda: constrained(reason="Core_capability.tools"),
        lambda: guarded("assistant", replacement_reason="annotated text"),
    ],
)
def test_helpers_reject_unstructured_reasons(declare):
    with pytest.raises(ValueError, match="structured reason"):
        declare()


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (lambda: ObjectPolicy(source="#/components/schemas/Message"), "bare component"),
        (lambda: ObjectPolicy(source="Message Schema"), "bare component"),
        (lambda: ObjectPolicy(opaque=("*",)), "wildcards"),
        (lambda: ObjectPolicy(opaque=("",)), "wildcards"),
    ],
)
def test_object_policy_rejects_values_the_contract_cannot_represent(policy, message):
    with pytest.raises(ValueError, match=message):
        policy()


def test_replacement_policy_requires_subject():
    with pytest.raises(ValueError, match="requires a subject"):
        guarded(replaceable=True)


def test_text_binding_derivation_does_not_generate_schemas(monkeypatch):
    """Unrelated unions and schema-export availability cannot change a text path."""
    import nemoguardrails.server.experimental.provider.projection_policy as policies

    class Request(PolicyModel):
        text: Annotated[str, guarded("user", min_length=2)]
        option: Annotated[str | int, constrained()]

    def unexpected_schema(*args, **kwargs):
        raise AssertionError("Runtime binding derivation must not generate schemas")

    monkeypatch.setattr(policies, "export_payload_schema", unexpected_schema)
    monkeypatch.setattr(policies, "policy_json_schema", unexpected_schema)
    monkeypatch.setattr(Request, "model_json_schema", unexpected_schema)

    location = text_location(Request)
    assert location.object_path == ()
    assert location.member == "text"
    assert location.role == "user"
    assert location.allows_empty is False


def test_extraction_rejects_non_singleton_arrays():
    class Message(PolicyModel):
        text: Annotated[str, guarded("user")]

    class Request(PolicyModel):
        messages: Annotated[list[Message], guarded()]

    with pytest.raises(ValueError, match="exactly one array item"):
        text_location(Request)


def test_extraction_rejects_recursive_guarded_paths():
    """Recursive guarded containers fail explicitly rather than recursing forever."""

    class Node(PolicyModel):
        children: Annotated["list[Node]", guarded(min_length=1, max_length=1)]

    Node.model_rebuild()
    with pytest.raises(ValueError, match="Recursive guarded text paths"):
        text_location(Node)


def test_extraction_rejects_unknown_replacement_blocker():
    class Message(PolicyModel):
        text: Annotated[str, guarded("assistant", replaceable=True, blocked_by="annotatons")]
        annotations: Annotated[list[str] | None, constrained()] = None

    with pytest.raises(ValueError, match="'annotatons' is not a field"):
        text_location(Message)


def test_extraction_accepts_opaque_replacement_blocker():
    class Message(PolicyModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("citations",))
        text: Annotated[str, guarded("assistant", replaceable=True, blocked_by="citations")]

    assert text_location(Message).replacement_blocked_by == "citations"


def test_extraction_allows_empty_text_only_where_the_subject_type_does():
    class RequiredText(PolicyModel):
        text: Annotated[str, guarded("user", min_length=1)]

    class NullableText(PolicyModel):
        text: Annotated[str | None, guarded("assistant")]

    assert text_location(RequiredText).allows_empty is False
    assert text_location(NullableText).allows_empty is True

    class Message(PolicyModel):
        text: Annotated[str, guarded("user")]

    assert text_location(Message).allows_empty is True


def test_extraction_rejects_multiple_subjects():
    class Request(PolicyModel):
        first: Annotated[str, guarded("user")]
        second: Annotated[str, guarded("user")]

    with pytest.raises(ValueError, match="exactly one guarded subject"):
        text_location(Request)


def test_export_does_not_require_contract_yaml(monkeypatch):
    class Request(PolicyModel):
        text: Annotated[str, guarded("user")]

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Contract export must not read YAML policy")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert export_payload_schema(Request, projection_id="test.request")["type"] == "object"


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize(
    ("member", "forbid", "allow"),
    [
        ("key", True, True),
        ("Key", False, False),
        ("KEY", False, False),
        ("meßage", False, False),
        ("meſsage", False, False),
        ("meſſage", False, False),
        ("Message\n", False, True),
        ("key\n", False, True),
        ("namK", False, True),
    ],
)
def test_export_reference_matches_unicode_casefold_and_exact_names(
    member, forbid, allow, policy, export_policy_validator
):
    class Content(PolicyModel, GuardedContentModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("key",), unknown_fields="configurable")
        message: Annotated[str, guarded("user")]

    document = {"message": "q", member: "x"}
    exported = export_payload_schema(Content, projection_id="test.casefold")
    derived = export_policy_validator(exported, policy).is_valid(document)
    expected = allow if policy == UnknownContentFieldPolicy.ALLOW else forbid
    assert derived is expected
    if expected:
        Content.validate_payload(document, unknown_content_fields=policy)
    else:
        with pytest.raises(ValidationError):
            Content.validate_payload(document, unknown_content_fields=policy)


def test_policy_models_reject_aliases_and_unpolicied_nested_models():
    from pydantic import BaseModel

    with pytest.raises(ValueError, match="aliases are not supported"):

        class Aliased(PolicyModel):
            text: Annotated[str, guarded("user"), Field(alias="Text")]

    class Plain(BaseModel):
        text: str

    class Root(PolicyModel):
        items: Annotated[list[Plain], guarded(min_length=1, max_length=1)]

    with pytest.raises(ValueError, match="Nested models must declare field policies"):
        payload_contract(Root, projection_id="test", direction="request")


def test_model_graph_visits_shared_models_once_and_rejects_name_collisions():
    from nemoguardrails.server.experimental.provider.projection_policy import model_graph

    class Part(PolicyModel):
        text: Annotated[str, guarded("user")]

    class Root(PolicyModel):
        first: Annotated[Part, guarded()]
        second: Annotated[Part | None, constrained()] = None

    assert list(model_graph(Root)) == ["Root", "Part"]

    def other_part() -> type[PolicyModel]:
        class Part(PolicyModel):
            value: Annotated[str, constrained()]

        return Part

    class Colliding(PolicyModel):
        first: Annotated[Part, guarded()]
        second: Annotated[other_part() | None, constrained()] = None

    with pytest.raises(ValueError, match="Model names must be unique"):
        model_graph(Colliding)


def test_export_rejects_recursive_models():
    class Node(PolicyModel):
        text: Annotated[str, guarded("user")]
        children: Annotated["list[Node]", constrained()] = []

    Node.model_rebuild()
    with pytest.raises(ValueError, match="Recursive models"):
        export_payload_schema(Node, projection_id="test")


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[int, guarded("user")],
        Annotated[list[Annotated[str, guarded("user")]], guarded(min_length=1, max_length=1)],
    ],
)
def test_extraction_requires_a_named_string_subject(annotation):
    root = type(
        "Root",
        (PolicyModel,),
        {"__annotations__": {"value": annotation}, "__module__": __name__},
    )

    with pytest.raises(ValueError, match="named string field"):
        text_location(root)


@pytest.mark.parametrize("policy", list(UnknownContentFieldPolicy))
@pytest.mark.parametrize("member", ["key", "Straße", "Key", "STRASSE", "Content", "future", "Content\n"])
def test_explicit_export_member_rules_match_runtime(member, policy, export_policy_validator):
    from nemoguardrails.server.experimental.provider.payload import GuardedContentModel, GuardedPayloadModel

    class Message(PolicyModel, GuardedContentModel):
        policy: ClassVar[ObjectPolicy] = ObjectPolicy(opaque=("key", "Straße"), unknown_fields="configurable")
        content: Annotated[str, guarded("user", min_length=1)]

    class Request(PolicyModel, GuardedPayloadModel):
        message: Annotated[Message, guarded()]

    document = {"message": {"content": "q", member: "opaque"}}
    exported = export_payload_schema(Request, projection_id="example.request")
    expected = member in {"key", "Straße"} or (
        policy == UnknownContentFieldPolicy.ALLOW and member in {"future", "Content\n"}
    )
    assert export_policy_validator(exported, policy).is_valid(document) is expected
    if expected:
        Request.validate_payload(document, unknown_content_fields=policy)
    else:
        with pytest.raises(ValidationError):
            Request.validate_payload(document, unknown_content_fields=policy)
    if policy == UnknownContentFieldPolicy.ALLOW and member in {"Key", "STRASSE", "Content"}:
        del exported["properties"]["message"][EXTENSION]["reject_case_aliases"]
        assert export_policy_validator(exported, policy).is_valid(document)
