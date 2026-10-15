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

"""Independent pinned Chat specification and explicit synthetic examples.

This is test support, not a compiler or a generic payload generator. Provider
fixtures must validate against the source independently of our guard models.
"""

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

FIXTURE = Path(__file__).parent / "fixtures/pinned.openapi.yaml"
DESCRIPTOR = Path(__file__).parents[6] / "nemoguardrails/server/experimental/contracts/openai/source.yaml"


@lru_cache
def source_document():
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def resolve_pointer(document, pointer):
    """Resolve local JSON pointers; never retrieve a remote schema."""
    if not pointer.startswith("#/"):
        raise ValueError(f"Expected a local JSON pointer: {pointer}")
    value = document
    for token in pointer[2:].split("/"):
        value = value[token.replace("~1", "/").replace("~0", "~")]
    return value


def object_properties(document, schema):
    """Collect one selected object's properties, including references and allOf."""
    result = {}
    if "$ref" in schema:
        result.update(object_properties(document, resolve_pointer(document, schema["$ref"])))
    for part in schema.get("allOf", ()):
        result.update(object_properties(document, part))
    result.update(schema.get("properties", {}))
    return result


def normalize_nullable(schema):
    """Interpret legacy nullable annotations in this mixed-dialect source pin.

    The upstream document declares OpenAPI 3.1 but still uses nullable next to
    types, enums, and references. Preserve its bytes and adapt only validation;
    do not confuse ignored nullable annotations with intentional rejection.
    """
    if not isinstance(schema, dict):
        return schema
    result = deepcopy(schema)
    for keyword in ("properties", "patternProperties", "$defs", "dependentSchemas"):
        if keyword in result:
            result[keyword] = {name: normalize_nullable(value) for name, value in result[keyword].items()}
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if keyword in result:
            result[keyword] = [normalize_nullable(value) for value in result[keyword]]
    for keyword in ("items", "additionalProperties", "not", "contains", "if", "then", "else"):
        if keyword in result:
            result[keyword] = normalize_nullable(result[keyword])
    if result.pop("nullable", False):
        return {"anyOf": [result, {"type": "null"}]}
    return result


def source_validator(schema):
    """Validate against the pinned source, not a runtime-generated schema."""
    document = source_document()
    if isinstance(schema, str):
        schema = {"$ref": "#/components/schemas/" + schema}
    return Draft202012Validator(
        {
            **normalize_nullable(schema),
            "components": {
                "schemas": {
                    name: normalize_nullable(value) for name, value in document["components"]["schemas"].items()
                }
            },
        }
    )


def chat_request(**updates):
    return {"model": "test-model", "messages": [{"role": "user", "content": "question"}], **updates}


def chat_response(**updates):
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": "answer", "refusal": None},
            }
        ],
        **updates,
    }


def chat_chunk(content="answer", **updates):
    return {
        "id": "chatcmpl-conformance",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": None, "delta": {"content": content}}],
        **updates,
    }


def json_bytes(payload):
    """Keep formatting and Unicode visible to byte-preservation assertions."""
    return json.dumps(payload, ensure_ascii=False, indent=2).encode()


def sse_bytes(*events):
    return (
        b"".join(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n" for event in events)
        + b"data: [DONE]\r\n\r\n"
    )
