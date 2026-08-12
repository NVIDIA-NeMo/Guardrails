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

"""Verify AnthropicMessagesRequest stays in sync with the anthropic SDK.

The request schema is defined as an explicit Pydantic model for readability.
This test introspects the SDK's MessageCreateParams TypedDict and checks that
every field the SDK considers required is present in our model, and that we
haven't invented fields that don't exist upstream.
"""

import typing

import pytest

anthropic_types = pytest.importorskip("anthropic.types")

from anthropic.types import MessageCreateParams  # noqa: E402
from anthropic.types import message_create_params as mcp_module  # noqa: E402

from nemoguardrails.server.schemas.anthropic import AnthropicMessagesRequest  # noqa: E402

GUARDRAILS_ONLY_FIELDS = {"guardrails"}


def _get_sdk_fields() -> tuple[set[str], set[str]]:
    """Return (required, all) field names from the SDK's MessageCreateParams."""
    td = MessageCreateParams.__args__[0]
    resolved = typing.get_type_hints(td, localns=vars(mcp_module), include_extras=True)

    required = set()
    for name, annotation in resolved.items():
        origin = typing.get_origin(annotation)
        if origin is typing.Required:
            required.add(name)
        elif name in td.__required_keys__:
            required.add(name)

    return required, set(resolved.keys())


def test_all_sdk_required_fields_present():
    """Every field the SDK marks as Required must appear in our schema."""
    required, _ = _get_sdk_fields()
    our_fields = set(AnthropicMessagesRequest.model_fields.keys())
    missing = required - our_fields - GUARDRAILS_ONLY_FIELDS
    assert not missing, (
        f"AnthropicMessagesRequest is missing required SDK fields: {missing}. "
        f"Add them to nemoguardrails/server/schemas/anthropic.py."
    )


def test_no_invented_fields():
    """Our schema should not contain fields that don't exist in the SDK."""
    _, sdk_fields = _get_sdk_fields()
    our_fields = set(AnthropicMessagesRequest.model_fields.keys())
    invented = our_fields - sdk_fields - GUARDRAILS_ONLY_FIELDS
    assert not invented, (
        f"AnthropicMessagesRequest has fields not in the SDK: {invented}. "
        f"Remove them or add to GUARDRAILS_ONLY_FIELDS if intentional."
    )


def test_all_sdk_fields_present():
    """Every SDK field must appear in our schema — we forward everything to the model."""
    _, sdk_fields = _get_sdk_fields()
    our_fields = set(AnthropicMessagesRequest.model_fields.keys())
    missing = sdk_fields - our_fields - GUARDRAILS_ONLY_FIELDS
    assert not missing, (
        f"AnthropicMessagesRequest is missing SDK fields: {missing}. "
        f"Add them to nemoguardrails/server/schemas/anthropic.py so they are forwarded to the model."
    )
