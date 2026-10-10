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

"""Verify that stream export describes the binding without constructing hooks."""

from dataclasses import replace

import pytest

from nemoguardrails.server.experimental.provider.contract_export import export_guard_contract
from nemoguardrails.server.experimental.provider.projection_policy import EXTENSION
from nemoguardrails.server.experimental.provider.sse import ServerSentEvent
from nemoguardrails.server.experimental.provider.stream import StreamBinding, StreamEventRole
from nemoguardrails.server.experimental.provider.stream_classifier import StreamEventRule
from nemoguardrails.server.experimental.provider.stream_policy import stream_shape_coverage
from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_classifier import STREAM_CLASSIFIER
from nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks import ChatCompletionsStreamHooks


def export_stream(classifier=STREAM_CLASSIFIER):
    binding = StreamBinding(classifier, ChatCompletionsStreamHooks)
    return export_guard_contract(replace(CHAT_COMPLETIONS_ENDPOINT, stream=binding))["stream"]


def with_rules(*rules, **transport):
    classifier = replace(STREAM_CLASSIFIER, rules=rules, **transport)
    coverage = stream_shape_coverage(rules, sentinels=classifier.sentinels, non_data_shape=classifier.non_data_shape)
    return replace(classifier, contract=replace(classifier.contract, shapes=coverage, fields=None))


def test_stream_export_describes_actual_rules_and_transport():
    stream = export_stream()
    assert stream["title"] == STREAM_CLASSIFIER.contract.projection_id
    assert stream[EXTENSION]["transport"] == {
        "require_sse_event": False,
        "non_data_shape": "[DONE]",
        "sentinels": {"[DONE]": "[DONE]"},
    }
    text, error = stream["oneOf"]
    assert text[EXTENSION]["event"] == {
        "classification": "guarded_delta",
        "variants": [
            {
                "source_schema": "CreateChatCompletionStreamResponse",
                "shape": "chat.completion.chunk:content",
                "match": {"object": "chat.completion.chunk"},
            }
        ],
        "missing_text": "opaque",
        "missing_text_shape": "chat.completion.chunk:metadata",
    }
    assert error[EXTENSION]["event"] == {
        "classification": "provider_error",
        "variants": [{"source_schema": "ErrorResponse", "shape": "error", "required_fields": ["error"]}],
    }
    choices = text["properties"]["choices"]
    assert choices["maxItems"] == 1
    delta = choices["items"]["properties"]["delta"]
    assert delta["properties"]["content"][EXTENSION]["subject"]["role"] == "assistant"
    assert choices["items"]["properties"]["logprobs"]["type"] == "null"


def test_buffered_only_export_preserves_payloads_and_labels():
    full = export_guard_contract(CHAT_COMPLETIONS_ENDPOINT)
    buffered = export_guard_contract(replace(CHAT_COMPLETIONS_ENDPOINT, stream=None))
    assert buffered == {key: value for key, value in full.items() if key != "stream"}


def test_binding_and_export_do_not_instantiate_hooks():
    def forbidden():
        raise AssertionError("Inspection must not construct protocol state")

    binding = StreamBinding(STREAM_CLASSIFIER, forbidden)
    endpoint = replace(CHAT_COMPLETIONS_ENDPOINT, stream=binding)
    assert export_guard_contract(endpoint)["stream"] == export_stream()
    with pytest.raises(AssertionError, match="protocol state"):
        binding.create_adapter()


def test_manual_classifier_is_runtime_only():
    class ManualClassifier:
        contract = STREAM_CLASSIFIER.contract

        def classify_event(self, event):
            return STREAM_CLASSIFIER.classify_event(event)

    binding = StreamBinding(ManualClassifier(), ChatCompletionsStreamHooks)
    endpoint = replace(CHAT_COMPLETIONS_ENDPOINT, stream=binding)
    event = ServerSentEvent.from_bytes(b"data: [DONE]\n\n")
    adapter = binding.create_adapter()
    assert adapter.classify_event(event).role is StreamEventRole.OPAQUE_METADATA
    adapter.validate_end_of_stream()
    with pytest.raises(TypeError, match="standard declarative"):
        export_guard_contract(endpoint)


def test_binding_validates_hooks_only_when_creating_an_adapter():
    binding = StreamBinding(STREAM_CLASSIFIER, lambda: object())
    with pytest.raises(TypeError, match="provider stream hooks"):
        binding.create_adapter()


@pytest.mark.parametrize("classifier,hooks", [(object(), ChatCompletionsStreamHooks), (STREAM_CLASSIFIER, None)])
def test_binding_rejects_invalid_declarations(classifier, hooks):
    with pytest.raises(TypeError, match="stream binding"):
        StreamBinding(classifier, hooks)


def test_endpoint_requires_an_explicit_stream_binding():
    with pytest.raises(TypeError, match="StreamBinding"):
        replace(CHAT_COMPLETIONS_ENDPOINT, stream=lambda: None)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"text_path": ("object",)}, "text binding differs"),
        ({"text_path": None}, "inconsistent text role"),
        ({"missing_text_value": "invisible"}, "must carry text"),
    ],
)
def test_export_revalidates_bound_rules(changes, message):
    rule = replace(STREAM_CLASSIFIER.rules[0], **changes)
    classifier = replace(STREAM_CLASSIFIER, rules=(rule, STREAM_CLASSIFIER.rules[1]))
    with pytest.raises(ValueError, match=message):
        export_stream(classifier)


def test_export_rejects_shape_inventory_drift():
    classifier = replace(STREAM_CLASSIFIER, rules=STREAM_CLASSIFIER.rules[:1])
    with pytest.raises(ValueError, match="coverage differs"):
        export_stream(classifier)


@pytest.mark.parametrize("value", [None, 0.0])
def test_export_rejects_unrepresentable_match_values(value):
    rule = replace(STREAM_CLASSIFIER.rules[0], match=(("object", value),))
    with pytest.raises(ValueError, match="unique match keys"):
        export_stream(with_rules(rule))


def test_export_rejects_duplicate_match_keys():
    rule = replace(STREAM_CLASSIFIER.rules[0], match=(("object", "one"), ("object", "two")))
    with pytest.raises(ValueError, match="unique match keys"):
        export_stream(with_rules(rule))


def test_export_rejects_custom_rule_matching():
    class CustomRule(StreamEventRule):
        def matches(self, payload):
            return True

    rule = CustomRule(
        shape="error",
        model=STREAM_CLASSIFIER.rules[1].model,
        role=StreamEventRole.PROVIDER_ERROR,
        required_fields=frozenset({"error"}),
    )
    with pytest.raises(TypeError, match="standard declarative event rules"):
        export_stream(with_rules(rule))


@pytest.mark.parametrize("role", [StreamEventRole.GUARDED_TEXT, StreamEventRole.TEXT_SNAPSHOT])
@pytest.mark.parametrize("fallback", ["reject", "empty"])
def test_export_describes_text_roles_and_supported_missing_text(role, fallback):
    rule = replace(
        STREAM_CLASSIFIER.rules[0],
        role=role,
        missing_text_role=role if fallback == "empty" else None,
        missing_text_shape=None,
        missing_text_value="" if fallback == "empty" else None,
    )
    exported = export_stream(with_rules(rule))["oneOf"][0][EXTENSION]["event"]
    assert exported["classification"] == ("snapshot" if role is StreamEventRole.TEXT_SNAPSHOT else "guarded_delta")
    assert exported["missing_text"] == fallback
    assert "missing_text_shape" not in exported


def test_export_describes_opaque_event_rules():
    rule = replace(STREAM_CLASSIFIER.rules[1], role=StreamEventRole.OPAQUE_METADATA)
    exported = export_stream(with_rules(rule))["oneOf"][0][EXTENSION]["event"]
    assert exported["classification"] == "opaque"
    assert "missing_text" not in exported


@pytest.mark.parametrize(
    "role,value",
    [
        (StreamEventRole.GUARDED_TEXT, "invented"),
        (StreamEventRole.TEXT_SNAPSHOT, ""),
        (StreamEventRole.PROVIDER_ERROR, None),
    ],
)
def test_export_rejects_lossy_missing_text_fallback(role, value):
    rule = replace(
        STREAM_CLASSIFIER.rules[0], missing_text_role=role, missing_text_shape="fallback", missing_text_value=value
    )
    with pytest.raises(ValueError, match="missing-text fallback"):
        export_stream(with_rules(rule))


def test_export_requires_explicit_source_provenance(monkeypatch):
    model = STREAM_CLASSIFIER.rules[0].model
    monkeypatch.setattr(model, "policy", replace(model.policy, source=None))
    with pytest.raises(ValueError, match="explicit policy source"):
        export_stream()


@pytest.mark.parametrize("sentinels", [((b" [DONE]", "[DONE]"),), ((b"[DONE]", "[DONE]"), (b"[DONE]", "[DONE]"))])
def test_export_rejects_noncanonical_or_duplicate_sentinels(sentinels):
    with pytest.raises(ValueError, match="unique nonempty canonical"):
        export_stream(replace(STREAM_CLASSIFIER, sentinels=sentinels))


def test_export_rejects_binary_sentinels():
    with pytest.raises(ValueError, match="UTF-8"):
        export_stream(replace(STREAM_CLASSIFIER, sentinels=((b"\xff", "[DONE]"),)))


def test_export_describes_data_discriminator_and_required_event():
    classifier = replace(STREAM_CLASSIFIER, event_type_field="object", require_event_type=True)
    transport = export_stream(classifier)[EXTENSION]["transport"]
    assert transport["require_sse_event"] is True
    assert transport["data_discriminator"] == "object"


def test_stream_exports_do_not_share_mutable_state():
    exported = export_stream()
    exported[EXTENSION]["transport"]["sentinels"].clear()
    exported["oneOf"][0]["properties"].clear()
    assert export_stream()[EXTENSION]["transport"]["sentinels"] == {"[DONE]": "[DONE]"}
    assert "choices" in export_stream()["oneOf"][0]["properties"]
