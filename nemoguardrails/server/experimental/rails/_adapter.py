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

"""Adapt Guardrails execution to the proxy content-checking boundary."""

from typing import Any, Protocol

from nemoguardrails import RailsConfig
from nemoguardrails.rails.llm.options import RailsResult, RailStatus, RailType
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckDecision,
    ContentCheckFailed,
    ContentInspectionPolicy,
    InputContentCheck,
    OutputContentCheck,
    StreamBufferingPolicy,
    UnsupportedContentCheckerConfiguration,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage


class _LegacyGuardrails(Protocol):
    config: RailsConfig

    async def check_async(
        self,
        messages: list[dict[str, Any]],
        rail_types: list[RailType] | None = None,
    ) -> RailsResult: ...

    async def shutdown(self) -> None: ...


def _message_document(message: GuardedMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}


def _rail_is_configured(rails: _LegacyGuardrails, rail_type: RailType) -> bool:
    rail_config = getattr(rails.config.rails, rail_type.value, None)
    flows = getattr(rail_config, "flows", None)
    return True if flows is None else bool(flows)


def _decision(result: RailsResult) -> ContentCheckDecision:
    if result.status == RailStatus.BLOCKED:
        return ContentBlocked(message=result.content, rule=result.rail)
    if result.status == RailStatus.MODIFIED:
        return ContentAllowed(replacement=result.content)
    return ContentAllowed()


class LegacyGuardrailsChecker:
    """Keep legacy rail configuration and results behind one proxy adapter."""

    def __init__(self, rails: _LegacyGuardrails):
        self._rails = rails

    def inspection_policy(self) -> ContentInspectionPolicy:
        rail_config = self._rails.config.rails
        unsupported = [
            rail_type
            for rail_type in ("retrieval", "tool_input", "tool_output")
            if getattr(getattr(rail_config, rail_type, None), "flows", None)
            or any(getattr(getattr(rail_config, rail_type, None), "per_tool", {}).values())
        ]
        if unsupported:
            raise UnsupportedContentCheckerConfiguration(
                f"These configured rail types are not supported by this reference path: {unsupported}."
            )

        inspect_input = _rail_is_configured(self._rails, RailType.INPUT)
        inspect_output = _rail_is_configured(self._rails, RailType.OUTPUT)
        stream_buffering = None
        if inspect_output:
            streaming = self._rails.config.rails.output.streaming
            if streaming.enabled:
                stream_buffering = StreamBufferingPolicy(
                    chunk_size=streaming.chunk_size,
                    context_size=streaming.context_size,
                    release_before_check=streaming.stream_first,
                )
        return ContentInspectionPolicy(
            inspect_input=inspect_input,
            inspect_output=inspect_output,
            stream_buffering=stream_buffering,
        )

    async def check_input(self, check: InputContentCheck) -> ContentCheckDecision:
        try:
            result = await self._rails.check_async(
                [_message_document(check.message)],
                rail_types=[RailType.INPUT],
            )
        except Exception as error:
            return ContentCheckFailed("The input Guardrails check failed.", error)
        return _decision(result)

    async def check_output(self, check: OutputContentCheck) -> ContentCheckDecision:
        try:
            result = await self._rails.check_async(
                [
                    _message_document(check.input_message),
                    {"role": "assistant", "content": check.output_content},
                ],
                rail_types=[RailType.OUTPUT],
            )
        except Exception as error:
            return ContentCheckFailed("The output Guardrails check failed.", error)
        return _decision(result)

    async def close(self) -> None:
        await self._rails.shutdown()
