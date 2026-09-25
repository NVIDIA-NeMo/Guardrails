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
from pydantic import BaseModel, RootModel

from nemoguardrails.server.experimental.provider.payload import (
    GuardedArraySelection,
    GuardedTextAlternatives,
    GuardedTextLocation,
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
