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

"""Execute buffered provider operations with optional content checks."""

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
    _ResolvedContentChecker,
    validate_content_check_decision,
    validate_content_checker,
)
from nemoguardrails.server.experimental._guarded_operation import (
    BufferedGuardedOperation,
    ContentInspectionNotApplicable,
    GuardedMessageProjection,
    UnsupportedGuardedPayload,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage

RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")
PayloadT = TypeVar("PayloadT")


class InspectionStage(str, Enum):
    """Identify whether an operation stopped during input or output checking."""

    INPUT = "input"
    OUTPUT = "output"


@dataclass(frozen=True, slots=True)
class OperationCompleted(Generic[ResponseT]):
    """Return the original provider response after required checks allow it."""

    response: ResponseT


@dataclass(frozen=True, slots=True)
class OperationBlocked:
    """Stop an operation when a content check blocks it."""

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


@dataclass(frozen=True, slots=True)
class OperationProjectionFailed:
    """Stop an operation when a provider payload cannot be inspected safely."""

    stage: InspectionStage
    failure: UnsupportedGuardedPayload


def _project_message(
    projection: GuardedMessageProjection[PayloadT],
    payload: PayloadT,
    expected_role: Literal["user", "assistant"],
) -> GuardedMessage | ContentInspectionNotApplicable:
    """Project and validate one result for content checking."""

    message = projection(payload)
    if isinstance(message, ContentInspectionNotApplicable):
        return message
    if not isinstance(message, GuardedMessage):
        raise TypeError("A guarded operation projection must return GuardedMessage or ContentInspectionNotApplicable.")
    if message.role != expected_role:
        raise ValueError(f"A guarded operation {expected_role} projection returned role {message.role!r}.")
    return message


def _stopped_operation(
    stage: InspectionStage,
    decision: object,
) -> OperationBlocked | OperationCheckFailed | OperationModificationUnsupported | None:
    """Convert a checker decision into an operation outcome."""

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
    """Run one checker call and convert errors into failure outcomes."""

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
    checker: ContentChecker | _ResolvedContentChecker,
    request: RequestT,
    dispatch: Callable[[RequestT], Awaitable[ResponseT]],
) -> (
    OperationCompleted[ResponseT]
    | OperationBlocked
    | OperationCheckFailed
    | OperationModificationUnsupported
    | OperationProjectionFailed
):
    """Execute one buffered operation with exactly one statically bound checker."""

    validated = checker if isinstance(checker, _ResolvedContentChecker) else validate_content_checker(checker)
    validated_checker = validated.checker
    policy = validated.policy
    try:
        input_message = _project_message(operation.input_projection, request, "user")
    except UnsupportedGuardedPayload as failure:
        return OperationProjectionFailed(InspectionStage.INPUT, failure)
    if isinstance(input_message, ContentInspectionNotApplicable):
        raise TypeError("A guarded operation input projection cannot be inapplicable.")

    if policy.inspect_input:
        stopped = await _run_check(
            InspectionStage.INPUT,
            lambda: validated_checker.check_input(InputContentCheck(input_message)),
        )
        if stopped is not None:
            return stopped

    response = await dispatch(request)

    if policy.inspect_output:
        try:
            output_message = _project_message(operation.output_projection, response, "assistant")
        except UnsupportedGuardedPayload as failure:
            return OperationProjectionFailed(InspectionStage.OUTPUT, failure)
        if isinstance(output_message, ContentInspectionNotApplicable):
            return OperationCompleted(response)
        stopped = await _run_check(
            InspectionStage.OUTPUT,
            lambda: validated_checker.check_output(OutputContentCheck(input_message, output_message.content)),
        )
        if stopped is not None:
            return stopped

    return OperationCompleted(response)
