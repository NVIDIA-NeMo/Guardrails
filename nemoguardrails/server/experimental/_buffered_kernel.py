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

"""Execute buffered guarded operations without owning provider transport."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, TypeVar

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentChecker,
    ContentCheckFailed,
    InputContentCheck,
    OutputContentCheck,
    UnsupportedContentModification,
    validate_content_check_decision,
    validate_content_checker,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation, GuardedMessageProjection
from nemoguardrails.server.experimental.provider.types import GuardedMessage

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
PayloadT = TypeVar("PayloadT")


class InspectionStage(str, Enum):
    """Identify the provider crossing that produced an enforcement outcome."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class OperationCompleted(Generic[ResponseT]):
    """Return the original provider response after required checks allow it."""

    response: ResponseT


@dataclass(frozen=True, slots=True)
class OperationBlocked:
    """Stop an operation at one inspected provider crossing."""

    stage: InspectionStage
    decision: ContentBlocked


@dataclass(frozen=True, slots=True)
class OperationCheckFailed:
    """Stop an operation when one required check cannot decide."""

    stage: InspectionStage
    failure: ContentCheckFailed


@dataclass(frozen=True, slots=True)
class OperationModificationUnsupported:
    """Stop an operation when a checker requests unsupported replacement."""

    stage: InspectionStage
    failure: UnsupportedContentModification


def _project_message(
    projection: GuardedMessageProjection[PayloadT],
    payload: PayloadT,
    expected_role: Literal["user", "assistant"],
) -> GuardedMessage:
    message = projection(payload)
    if not isinstance(message, GuardedMessage):
        raise TypeError("A guarded operation projection must return GuardedMessage.")
    if message.role != expected_role:
        raise ValueError(f"A guarded operation {expected_role} projection returned role {message.role!r}.")
    return message


def _stopped_operation(
    stage: InspectionStage,
    decision: object,
) -> OperationBlocked | OperationCheckFailed | OperationModificationUnsupported | None:
    try:
        validated = validate_content_check_decision(decision)
    except UnsupportedContentModification as failure:
        return OperationModificationUnsupported(stage, failure)
    except Exception as failure:
        return OperationCheckFailed(
            stage,
            ContentCheckFailed("The content checker returned an unsupported decision.", failure),
        )
    if isinstance(validated, ContentAllowed):
        return None
    if isinstance(validated, ContentBlocked):
        return OperationBlocked(stage, validated)
    return OperationCheckFailed(stage, validated)


async def _run_check(
    stage: InspectionStage,
    check: Callable[[], Awaitable[object]],
) -> OperationBlocked | OperationCheckFailed | OperationModificationUnsupported | None:
    try:
        decision = await check()
    except Exception as failure:
        return OperationCheckFailed(
            stage,
            ContentCheckFailed(f"The {stage.value} content check failed.", failure),
        )
    return _stopped_operation(stage, decision)


async def execute_buffered_operation(
    operation: BufferedGuardedOperation[RequestT, ResponseT],
    checker: ContentChecker,
    request: RequestT,
    dispatch: Callable[[RequestT], Awaitable[ResponseT]],
) -> OperationCompleted[ResponseT] | OperationBlocked | OperationCheckFailed | OperationModificationUnsupported:
    """Execute one buffered operation with exactly one statically bound checker."""

    validated = validate_content_checker(checker)
    validated_checker = validated.checker
    policy = validated.policy
    input_message = _project_message(operation.input_projection, request, "user")

    if policy.inspect_input:
        stopped = await _run_check(
            InspectionStage.INPUT,
            lambda: validated_checker.check_input(InputContentCheck(input_message)),
        )
        if stopped is not None:
            return stopped

    response = await dispatch(request)

    if policy.inspect_output:
        output_message = _project_message(operation.output_projection, response, "assistant")
        stopped = await _run_check(
            InspectionStage.OUTPUT,
            lambda: validated_checker.check_output(OutputContentCheck(input_message, output_message.content)),
        )
        if stopped is not None:
            return stopped

    return OperationCompleted(response)
