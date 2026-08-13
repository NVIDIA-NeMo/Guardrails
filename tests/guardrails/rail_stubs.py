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

"""Stand-ins for compiled rails, so a test about scheduling needs no paid vendor or NIM behind it.

``rails_compiled_as`` patches compilation rather than the compiled rails, so ``RailsManager``
reads these while deciding how to order and downgrade.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Iterator, Optional
from unittest.mock import patch

from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.compiled_rail import RailExecution
from nemoguardrails.guardrails.guardrails_types import RailResult
from nemoguardrails.manifests import RailDirection as SurfaceDirection


class StubRail:
    """Answers with one outcome, makes no model call, and keeps what it was handed.

    ``transform_target`` mirrors what a surface declares, not what the outcome does.
    """

    def __init__(self, outcome: Optional[RailOutcome] = None, transform_target: Optional[TransformTarget] = None):
        self.outcome = RailOutcome.allow() if outcome is None else outcome
        self.transform_target = transform_target
        self.seen_messages: list[dict] = []
        self.seen_bot_response: Optional[str] = None
        self.call_count = 0

    async def execute(self, messages: list[dict], bot_response: Optional[str] = None) -> RailExecution:
        self.seen_messages = [dict(message) for message in messages]
        self.seen_bot_response = bot_response
        self.call_count += 1
        return RailExecution(outcome=self.outcome)

    async def close(self) -> None:
        """A compiled rail may own an HTTP client; a stub has nothing to release."""


def rewriting_stub(text: str, direction: SurfaceDirection) -> StubRail:
    """A rail that both declares it may rewrite and does, to its direction's variable."""
    target = TransformTarget.USER_MESSAGE if direction is SurfaceDirection.INPUT else TransformTarget.BOT_MESSAGE
    return StubRail(RailOutcome.transform([(target, text)]), transform_target=target)


def declared_rewriter(direction: SurfaceDirection) -> StubRail:
    """A rail free to rewrite that does not, which is what most requests to one look like."""
    target = TransformTarget.USER_MESSAGE if direction is SurfaceDirection.INPUT else TransformTarget.BOT_MESSAGE
    return StubRail(transform_target=target)


def user_message_rewrite(text: str) -> RailResult:
    """The verdict input rails return when they rewrote the user message to *text*."""
    return RailResult(RailOutcome.transform([(TransformTarget.USER_MESSAGE, text)]))


def bot_message_rewrite(text: str) -> RailResult:
    """The verdict output rails return when they rewrote the response to *text*."""
    return RailResult(RailOutcome.transform([(TransformTarget.BOT_MESSAGE, text)]))


@contextmanager
def rails_compiled_as(rails: Mapping[str, StubRail]) -> Iterator[None]:
    """Compile the named flows to their stubs, and every other configured flow to a plain allow."""

    def _compile(flow: str, direction: SurfaceDirection, deps: Any, **kwargs: Any) -> StubRail:
        return rails.get(flow, StubRail())

    with patch("nemoguardrails.guardrails.rails_manager.compile_rail", side_effect=_compile):
        yield
