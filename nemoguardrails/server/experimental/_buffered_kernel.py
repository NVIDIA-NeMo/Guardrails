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
    ContentInspectionPolicy,
    GuardedText,
    InputContentCheck,
    OutputContentCheck,
    validate_content_check_decision,
    validate_content_checker,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation, GuardedTextProjection

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


def _project_subject(
    projection: GuardedTextProjection[PayloadT],
    payload: PayloadT,
    expected_role: Literal["user", "assistant"],
) -> GuardedText:
    subject = projection(payload)
    if not isinstance(subject, GuardedText):
        raise TypeError("A guarded operation projection must return GuardedText.")
    if subject.role != expected_role:
        raise ValueError(f"A guarded operation {expected_role} projection returned role {subject.role!r}.")
    return subject


def _stopped_operation(
    stage: InspectionStage,
    decision: object,
) -> OperationBlocked | OperationCheckFailed | None:
    validated = validate_content_check_decision(decision)
    if isinstance(validated, ContentAllowed):
        return None
    if isinstance(validated, ContentBlocked):
        return OperationBlocked(stage, validated)
    return OperationCheckFailed(stage, validated)


async def execute_buffered_operation(
    operation: BufferedGuardedOperation[RequestT, ResponseT],
    checker: ContentChecker,
    request: RequestT,
    dispatch: Callable[[RequestT], Awaitable[ResponseT]],
) -> OperationCompleted[ResponseT] | OperationBlocked | OperationCheckFailed:
    """Execute one buffered operation with exactly one statically bound checker."""

    validated = validate_content_checker(checker)
    validated_checker = validated.checker
    policy = validated.policy
    input_subject = _project_input_when_required(operation, request, policy)

    if policy.inspect_input:
        if input_subject is None:
            raise RuntimeError("Input inspection requires a projected input subject.")
        stopped = _stopped_operation(
            InspectionStage.INPUT,
            await validated_checker.check_input(InputContentCheck(input_subject)),
        )
        if stopped is not None:
            return stopped

    response = await dispatch(request)

    if policy.inspect_output:
        if input_subject is None:
            raise RuntimeError("Output inspection requires a projected input subject.")
        output_subject = _project_subject(operation.output_projection, response, "assistant")
        stopped = _stopped_operation(
            InspectionStage.OUTPUT,
            await validated_checker.check_output(OutputContentCheck(input_subject, output_subject)),
        )
        if stopped is not None:
            return stopped

    return OperationCompleted(response)


def _project_input_when_required(
    operation: BufferedGuardedOperation[RequestT, ResponseT],
    request: RequestT,
    policy: ContentInspectionPolicy,
) -> GuardedText | None:
    if not policy.inspect_input and not policy.inspect_output:
        return None
    return _project_subject(operation.input_projection, request, "user")
