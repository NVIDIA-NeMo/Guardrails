# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for the OpenAI server schemas across the supported pydantic range.

``OpenAIChatMessage`` used ``BeforeValidator(..., json_schema_input_type=...)``,
an argument that only pydantic 2.9+ accepts, while the project declares
``pydantic>=2.5``. These tests pin down the behavior required by issue #2292:
the module must import and validate identically on every supported version,
and the richer JSON schema input type is preserved where pydantic supports it.
"""

from typing import List

import pydantic
import pytest
from pydantic import BeforeValidator, ValidationError

from nemoguardrails.server.schemas.openai import OpenAIChatMessage


def _supports_json_schema_input_type() -> bool:
    try:
        BeforeValidator(lambda value: value, json_schema_input_type=int)
    except TypeError:
        return False
    return True


def _make_request_model() -> type[pydantic.BaseModel]:
    class RequestModel(pydantic.BaseModel):
        messages: List[OpenAIChatMessage]

    return RequestModel


def test_accepts_valid_user_message():
    model = _make_request_model()
    parsed = model.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    assert parsed.messages[0] == {"role": "user", "content": "hi"}


def test_rejects_unknown_role():
    model = _make_request_model()
    with pytest.raises(ValidationError):
        model.model_validate({"messages": [{"role": "wizard", "content": "hi"}]})


def test_json_schema_generation_succeeds():
    # JSON schema generation must work on every supported pydantic version.
    schema = _make_request_model().model_json_schema()
    assert "messages" in schema["properties"]


@pytest.mark.skipif(
    not _supports_json_schema_input_type(),
    reason="pydantic < 2.9 does not support json_schema_input_type",
)
def test_json_schema_uses_rich_input_type_when_supported():
    schema = _make_request_model().model_json_schema()
    items = schema["properties"]["messages"]["items"]
    # The union input type is rendered either as a direct reference or as an
    # anyOf over the per-role message schemas defined in $defs.
    if "$ref" in items:
        assert items["$ref"].startswith("#/$defs/")
    else:
        assert "anyOf" in items
        assert all(branch["$ref"].startswith("#/$defs/") for branch in items["anyOf"])
