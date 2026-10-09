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

import json
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import Field, ValidationError

from nemoguardrails.server.experimental.provider.payload import GuardedContentModel
from nemoguardrails.server.experimental.provider.projection_policy import (
    CONTRACT_VERSION,
    EXTENSION,
    ObjectPolicy,
    PolicyModel,
    constrained,
    disabled,
    export_payload_schema,
    field_coverage,
    guarded,
    payload_contract,
    text_location,
)
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    PAYLOAD_CONTRACT as REQUEST_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    ChatCompletionsGuardedRequest,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_projection import (
    ChatCompletionsGuardedRequestProjection,
    ChatCompletionsUserMessageProjection,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    PAYLOAD_CONTRACT as RESPONSE_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    ChatCompletionsGuardedResponse,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsAssistantMessageProjection,
    ChatCompletionsGuardedResponseProjection,
)


def test_explicit_defaults_are_visible_to_python_and_pydantic():
    message = ChatCompletionsUserMessageProjection(content="question", role="user")
    request = ChatCompletionsGuardedRequestProjection(messages=[message])
    assert request.n == 1
    assert request.stream is False
    assert request.audio is None
    for name, expected in (("n", 1), ("stream", False), ("audio", None)):
        field = ChatCompletionsGuardedRequestProjection.model_fields[name]
        assert not field.is_required()
        assert field.default is expected


@pytest.mark.parametrize("value", [None, []])
def test_annotations_accept_only_null_or_empty(value):
    message = ChatCompletionsAssistantMessageProjection(role="assistant", content="answer", annotations=value)
    assert message.annotations == value
    with pytest.raises(ValidationError):
        ChatCompletionsAssistantMessageProjection(role="assistant", content="answer", annotations=[{"type": "x"}])


def test_n_accepts_the_integer_one():
    request = ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "n": 1})
    assert request.n == 1


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, 2, None])
def test_n_rejects_coercible_and_other_values(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "n": value})


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_stream_stays_strict(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "stream": value})


@pytest.mark.parametrize("value", [[], {}, False, "audio"])
def test_disabled_fields_reject_every_non_null_value(value):
    with pytest.raises(ValidationError):
        ChatCompletionsGuardedRequest.model_validate({"messages": [{"role": "user", "content": "q"}], "audio": value})


def test_bindings_derive_coverage_and_targets_from_the_typed_models():
    assert REQUEST_CONTRACT.root == field_coverage(ChatCompletionsGuardedRequestProjection)
    assert RESPONSE_CONTRACT.root == field_coverage(ChatCompletionsGuardedResponseProjection)
    assert ChatCompletionsGuardedRequest.guarded_text_location == text_location(ChatCompletionsGuardedRequestProjection)
    assert ChatCompletionsGuardedResponse.guarded_text_location == text_location(
        ChatCompletionsGuardedResponseProjection
    )
    response_content = RESPONSE_CONTRACT.content_models[-1]
    assert response_content.model is ChatCompletionsAssistantMessageProjection
    assert response_content.coverage.local_extension_fields == frozenset({"reasoning_content"})


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


def test_export_preserves_nullable_annotation_schema():
    exported = export_payload_schema(
        ChatCompletionsGuardedResponseProjection, projection_id=RESPONSE_CONTRACT.projection_id
    )
    annotations = exported["properties"]["choices"]["items"]["properties"]["message"]["properties"]["annotations"]
    assert annotations["default"] is None
    assert {"type": "null"} in annotations["oneOf"]
    assert {"type": "array", "items": {}, "maxItems": 0} in annotations["oneOf"]


def test_export_lists_reviewed_opaque_names_as_properties():
    request = export_payload_schema(ChatCompletionsGuardedRequestProjection, projection_id="test.request")
    assert request["properties"]["model"] == {EXTENSION: {"classification": "opaque"}}
    assert "opaque_fields" not in request[EXTENSION]
    validator = Draft202012Validator(request)
    message = {"role": "user", "content": "q"}
    assert validator.is_valid({"messages": [message], "model": "m", "temperature": 0.2})
    assert not validator.is_valid({"messages": [message], "future": 1})


def test_export_follows_each_object_unknown_field_policy():
    request = export_payload_schema(ChatCompletionsGuardedRequestProjection, projection_id="test.request")
    response = export_payload_schema(ChatCompletionsGuardedResponseProjection, projection_id="test.response")
    choice = response["properties"]["choices"]["items"]
    configurable = (request["properties"]["messages"]["items"], choice, choice["properties"]["message"])
    for content in configurable:
        assert content["additionalProperties"] is True
        assert content[EXTENSION]["unknown_fields"] == "configurable"
    for root in (request, response):
        assert root["additionalProperties"] is False
        assert "unknown_fields" not in root[EXTENSION]


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


def test_extraction_rejects_non_singleton_arrays():
    class Message(PolicyModel):
        text: Annotated[str, guarded("user")]

    class Request(PolicyModel):
        messages: Annotated[list[Message], guarded()]

    with pytest.raises(ValueError, match="exactly one array item"):
        text_location(Request)


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
    assert ChatCompletionsGuardedResponse.guarded_text_location.allows_empty is True
    assert ChatCompletionsGuardedRequest.guarded_text_location.allows_empty is False

    class Message(PolicyModel):
        text: Annotated[str, guarded("user")]

    assert text_location(Message).allows_empty is True


def test_extraction_rejects_multiple_subjects():
    class Request(PolicyModel):
        first: Annotated[str, guarded("user")]
        second: Annotated[str, guarded("user")]

    with pytest.raises(ValueError, match="exactly one guarded subject"):
        text_location(Request)


def test_field_validation_matches_original_unannotated_declarations():
    from itertools import product
    from typing import Any

    from pydantic import BaseModel, Field, StrictBool

    class OriginalRequestFields(BaseModel):
        n: Literal[1] = 1
        stream: StrictBool = False
        audio: None = None

    class OriginalResponseFields(BaseModel):
        annotations: Annotated[list[Any] | None, Field(max_length=0)] = None
        content: str | None
        logprobs: None = None

    def validated(model, payload):
        try:
            return model.model_validate(payload).model_dump()
        except ValidationError:
            return "rejected"

    values = [None, True, False, 0, 1, 1.0, 2, "1", "true", "", [], {}, ["x"]]
    # n is intentionally stricter than Literal[1]; test_n_rejects_coercible_and_other_values covers it.
    for stream, audio in product(values, repeat=2):
        fields = {"n": 1, "stream": stream, "audio": audio}
        old = validated(OriginalRequestFields, fields)
        new = validated(
            ChatCompletionsGuardedRequestProjection,
            {
                "messages": [{"role": "user", "content": "q"}],
                **fields,
            },
        )
        if isinstance(new, dict):
            new = {name: new[name] for name in fields}
        assert old == new, fields

    for annotations, content, logprobs in product(values + ["answer"], repeat=3):
        fields = {"annotations": annotations, "content": content, "logprobs": logprobs}
        old = validated(OriginalResponseFields, fields)
        new = validated(
            ChatCompletionsGuardedResponseProjection,
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content, "annotations": annotations},
                        "logprobs": logprobs,
                    }
                ],
            },
        )
        if isinstance(new, dict):
            choice = new["choices"][0]
            new = {name: choice["message"][name] for name in ("annotations", "content")}
            new["logprobs"] = choice["logprobs"]
        assert old == new, fields


def test_export_does_not_require_contract_yaml(monkeypatch):
    from pathlib import Path

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Contract export must not read YAML policy")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    assert (
        export_payload_schema(ChatCompletionsGuardedRequestProjection, projection_id="test.request")["type"] == "object"
    )


GUARD_CONTRACT_SCHEMA = (
    Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts/guard-contract.schema.json"
)


def _contract_errors(request: type[PolicyModel], response: type[PolicyModel]) -> list[str]:
    """Wrap two payload exports in a minimal operation contract and validate it."""
    contract = {
        "version": CONTRACT_VERSION,
        "operationId": "createChatCompletion",
        "profile": "single_text.v1",
        "request": export_payload_schema(request, projection_id=REQUEST_CONTRACT.projection_id),
        "response": export_payload_schema(response, projection_id=RESPONSE_CONTRACT.projection_id),
        "integration": {
            "endpoint": {
                "unsupported_request_code": "unsupported_request",
                "unsupported_response_code": "unsupported_response",
            }
        },
    }
    validator = Draft202012Validator(json.loads(GUARD_CONTRACT_SCHEMA.read_text(encoding="utf-8")))
    return [error.message for error in validator.iter_errors(contract)]


def test_chat_exports_conform_to_the_guard_contract_schema():
    assert _contract_errors(ChatCompletionsGuardedRequestProjection, ChatCompletionsGuardedResponseProjection) == []


def test_contract_schema_check_catches_unexportable_constraints():
    class Response(PolicyModel):
        text: Annotated[str, guarded("assistant")]
        count: Annotated[int, Field(ge=0), constrained()]

    assert _contract_errors(ChatCompletionsGuardedRequestProjection, Response)
