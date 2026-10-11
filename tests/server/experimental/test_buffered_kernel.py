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

"""Test buffered provider execution with optional content checks."""

from dataclasses import dataclass, field

import pytest

from nemoguardrails.server.experimental._buffered_kernel import (
    InspectionStage,
    OperationBlocked,
    OperationCheckFailed,
    OperationCompleted,
    OperationModificationUnsupported,
    OperationProjectionFailed,
    execute_buffered_operation,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
)
from nemoguardrails.server.experimental._guarded_operation import (
    BufferedGuardedOperation,
    ContentInspectionNotApplicable,
    UnsupportedGuardedPayload,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage


@dataclass
class Request:
    """Represent a provider request with checked text and opaque state."""

    text: str
    opaque: object = field(default_factory=object)


@dataclass
class Response:
    """Represent a provider response with checked text and opaque state."""

    text: str
    opaque: object = field(default_factory=object)


class StaticChecker:
    """Provide configurable checker results and observable calls."""

    def __init__(
        self,
        policy=ContentInspectionPolicy(True, True),
        input_decision=ContentAllowed(),
        output_decision=ContentAllowed(),
        input_error=None,
        output_error=None,
    ):
        self.policy = policy
        self.input_decision = input_decision
        self.output_decision = output_decision
        self.input_error = input_error
        self.output_error = output_error
        self.calls = []

    def inspection_policy(self):
        self.calls.append("policy")
        return self.policy

    async def check_input(self, check):
        self.calls.append(("input", check))
        if self.input_error is not None:
            raise self.input_error
        return self.input_decision

    async def check_output(self, check):
        self.calls.append(("output", check))
        if self.output_error is not None:
            raise self.output_error
        return self.output_decision


def projection_raising(failure):
    def project(_payload):
        raise failure

    return project


@pytest.fixture
def operation():
    return BufferedGuardedOperation(
        name="test.buffered",
        input_projection=lambda request: GuardedMessage("user", request.text),
        output_projection=lambda response: GuardedMessage("assistant", response.text),
    )


@pytest.mark.asyncio
async def test_no_inspection_validates_input_and_dispatches_original_values():
    """Preserve opaque request and response values when checks are disabled."""

    checker = StaticChecker(ContentInspectionPolicy(False, False))
    request = Request("question")
    response = Response("answer")
    dispatched = []
    projected = []

    def project_input(value):
        projected.append(value)
        return GuardedMessage("user", value.text)

    operation = BufferedGuardedOperation(
        name="test.transparent",
        input_projection=project_input,
        output_projection=lambda _response: pytest.fail("output must not be projected"),
    )

    async def dispatch(value):
        dispatched.append(value)
        return response

    result = await execute_buffered_operation(operation, checker, request, dispatch)

    assert result == OperationCompleted(response)
    assert result.response is response
    assert dispatched == [request]
    assert dispatched[0] is request
    assert projected == [request]
    assert checker.calls == ["policy"]


@pytest.mark.asyncio
async def test_input_and_output_checks_share_one_checker_and_context(operation):
    """Use one checker and carry input context to output checking."""

    checker = StaticChecker()
    request = Request("question")
    response = Response("answer")

    async def dispatch(value):
        assert value is request
        checker.calls.append("dispatch")
        return response

    result = await execute_buffered_operation(operation, checker, request, dispatch)

    assert result.response is response
    assert checker.calls[0] == "policy"
    assert checker.calls[1][0] == "input"
    assert checker.calls[1][1].message == GuardedMessage("user", "question")
    assert checker.calls[2] == "dispatch"
    assert checker.calls[3][0] == "output"
    assert checker.calls[3][1].input_message == GuardedMessage("user", "question")
    assert checker.calls[3][1].output_content == "answer"


@pytest.mark.asyncio
async def test_output_only_inspection_still_projects_effective_input_context(operation):
    """Project input context when only output checking is enabled."""

    checker = StaticChecker(ContentInspectionPolicy(False, True))
    response = Response("answer")

    async def dispatch(_request):
        return response

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert result.response is response
    assert [call if isinstance(call, str) else call[0] for call in checker.calls] == [
        "policy",
        "output",
    ]
    assert checker.calls[-1][1].input_message.content == "question"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "outcome_type"),
    [(ContentBlocked("blocked", "rule"), OperationBlocked), (ContentCheckFailed("failed"), OperationCheckFailed)],
)
async def test_input_stop_prevents_dispatch(operation, decision, outcome_type):
    """Stop before provider dispatch when the input check does not allow it."""

    checker = StaticChecker(input_decision=decision)

    async def dispatch(_request):
        pytest.fail("a stopped input must not be dispatched")

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, outcome_type)
    assert result.stage is InspectionStage.INPUT
    assert [call if isinstance(call, str) else call[0] for call in checker.calls] == [
        "policy",
        "input",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "outcome_type"),
    [(ContentBlocked("blocked", "rule"), OperationBlocked), (ContentCheckFailed("failed"), OperationCheckFailed)],
)
async def test_output_stop_does_not_return_provider_response(operation, decision, outcome_type):
    """Hide a provider response when the output check does not allow it."""

    checker = StaticChecker(output_decision=decision)
    response = Response("answer")
    dispatch_count = 0

    async def dispatch(_request):
        nonlocal dispatch_count
        dispatch_count += 1
        return response

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, outcome_type)
    assert result.stage is InspectionStage.OUTPUT
    assert not hasattr(result, "response")
    assert dispatch_count == 1


@pytest.mark.asyncio
async def test_input_modification_fails_before_dispatch(operation):
    """Reject input replacement before provider dispatch."""

    checker = StaticChecker(input_decision=ContentAllowed(replacement="changed"))

    async def dispatch(_request):
        pytest.fail("a modified input must not be dispatched")

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, OperationModificationUnsupported)
    assert result.stage is InspectionStage.INPUT


@pytest.mark.asyncio
async def test_output_modification_fails_without_returning_provider_response(operation):
    """Reject output replacement without returning the provider response."""

    checker = StaticChecker(output_decision=ContentAllowed(replacement="changed"))
    response = Response("answer")

    async def dispatch(_request):
        return response

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, OperationModificationUnsupported)
    assert result.stage is InspectionStage.OUTPUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("projection", "message"),
    [
        (lambda _request: "question", "must return GuardedMessage"),
        (lambda _request: GuardedMessage("assistant", "question"), "user projection"),
    ],
)
async def test_invalid_input_projection_fails_before_dispatch(projection, message):
    """Reject invalid projected input before provider dispatch."""

    operation = BufferedGuardedOperation(
        name="test.invalid_input",
        input_projection=projection,
        output_projection=lambda response: GuardedMessage("assistant", response.text),
    )

    async def dispatch(_request):
        pytest.fail("an invalid projection must not be dispatched")

    with pytest.raises((TypeError, ValueError), match=message):
        await execute_buffered_operation(operation, StaticChecker(), Request("question"), dispatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("projection", "message"),
    [
        (lambda _response: "answer", "must return GuardedMessage"),
        (lambda _response: GuardedMessage("user", "answer"), "assistant projection"),
    ],
)
async def test_invalid_output_projection_fails_after_dispatch(projection, message):
    """Reject invalid projected output after provider dispatch."""

    operation = BufferedGuardedOperation(
        name="test.invalid_output",
        input_projection=lambda request: GuardedMessage("user", request.text),
        output_projection=projection,
    )
    request = Request("question")
    response = Response("answer")
    dispatched = []

    async def dispatch(value):
        dispatched.append(value)
        return response

    with pytest.raises((TypeError, ValueError), match=message):
        await execute_buffered_operation(operation, StaticChecker(), request, dispatch)

    assert dispatched == [request]


@pytest.mark.asyncio
async def test_unsupported_input_payload_fails_before_dispatch():
    """Return an input projection failure before provider dispatch."""

    failure = UnsupportedGuardedPayload("unsupported request")
    operation = BufferedGuardedOperation(
        name="test.unsupported_input",
        input_projection=projection_raising(failure),
        output_projection=lambda response: GuardedMessage("assistant", response.text),
    )

    async def dispatch(_request):
        pytest.fail("an unsupported input must not be dispatched")

    result = await execute_buffered_operation(operation, StaticChecker(), Request("question"), dispatch)

    assert result == OperationProjectionFailed(InspectionStage.INPUT, failure)


@pytest.mark.asyncio
async def test_unsupported_output_payload_hides_provider_response():
    """Hide the provider response when output projection fails."""

    failure = UnsupportedGuardedPayload("unsupported response")
    response = Response("answer")
    operation = BufferedGuardedOperation(
        name="test.unsupported_output",
        input_projection=lambda request: GuardedMessage("user", request.text),
        output_projection=projection_raising(failure),
    )

    async def dispatch(_request):
        return response

    result = await execute_buffered_operation(operation, StaticChecker(), Request("question"), dispatch)

    assert result == OperationProjectionFailed(InspectionStage.OUTPUT, failure)
    assert not hasattr(result, "response")


@pytest.mark.asyncio
async def test_inapplicable_output_inspection_preserves_provider_response():
    """Preserve a response that does not contain content to check."""

    response = Response("provider error")
    checker = StaticChecker()
    operation = BufferedGuardedOperation(
        name="test.inapplicable_output",
        input_projection=lambda request: GuardedMessage("user", request.text),
        output_projection=lambda _response: ContentInspectionNotApplicable(),
    )

    async def dispatch(_request):
        return response

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert result == OperationCompleted(response)
    assert [call if isinstance(call, str) else call[0] for call in checker.calls] == ["policy", "input"]


@pytest.mark.asyncio
async def test_input_inspection_cannot_be_declared_inapplicable():
    """Reject an input projection without content to check."""

    operation = BufferedGuardedOperation(
        name="test.inapplicable_input",
        input_projection=lambda _request: ContentInspectionNotApplicable(),
        output_projection=lambda response: GuardedMessage("assistant", response.text),
    )

    async def dispatch(_request):
        pytest.fail("an inapplicable input must not be dispatched")

    with pytest.raises(TypeError, match="input projection cannot be inapplicable"):
        await execute_buffered_operation(operation, StaticChecker(), Request("question"), dispatch)


@pytest.mark.asyncio
async def test_unknown_checker_decision_fails_closed_before_dispatch(operation):
    """Convert an unknown input decision into a failed check."""

    checker = StaticChecker(input_decision=object())

    async def dispatch(_request):
        pytest.fail("an unknown checker decision must not be dispatched")

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, OperationCheckFailed)
    assert result.stage is InspectionStage.INPUT
    assert isinstance(result.failure.cause, TypeError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "checker"),
    [
        (InspectionStage.INPUT, StaticChecker(input_error=RuntimeError("input failed"))),
        (InspectionStage.OUTPUT, StaticChecker(output_error=RuntimeError("output failed"))),
    ],
)
async def test_checker_exceptions_become_stage_specific_failures(operation, stage, checker):
    """Attach the correct inspection stage to checker exceptions."""

    response = Response("answer")

    async def dispatch(_request):
        return response

    result = await execute_buffered_operation(operation, checker, Request("question"), dispatch)

    assert isinstance(result, OperationCheckFailed)
    assert result.stage is stage
    assert isinstance(result.failure.cause, RuntimeError)
