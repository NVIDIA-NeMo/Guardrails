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

from dataclasses import dataclass, field

import pytest

from nemoguardrails.server.experimental._buffered_kernel import (
    InspectionStage,
    OperationBlocked,
    OperationCheckFailed,
    OperationCompleted,
    execute_buffered_operation,
)
from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
    GuardedText,
    UnsupportedContentModification,
)
from nemoguardrails.server.experimental._guarded_operation import BufferedGuardedOperation


@dataclass
class Request:
    text: str
    opaque: object = field(default_factory=object)


@dataclass
class Response:
    text: str
    opaque: object = field(default_factory=object)


class StaticChecker:
    def __init__(
        self,
        policy=ContentInspectionPolicy(True, True),
        input_decision=ContentAllowed(),
        output_decision=ContentAllowed(),
    ):
        self.policy = policy
        self.input_decision = input_decision
        self.output_decision = output_decision
        self.calls = []

    def inspection_policy(self):
        self.calls.append("policy")
        return self.policy

    async def check_input(self, check):
        self.calls.append(("input", check))
        return self.input_decision

    async def check_output(self, check):
        self.calls.append(("output", check))
        return self.output_decision


@pytest.fixture
def operation():
    return BufferedGuardedOperation(
        name="test.buffered",
        input_projection=lambda request: GuardedText("user", request.text),
        output_projection=lambda response: GuardedText("assistant", response.text),
    )


@pytest.mark.asyncio
async def test_no_inspection_dispatches_original_values_without_projection():
    checker = StaticChecker(ContentInspectionPolicy(False, False))
    request = Request("question")
    response = Response("answer")
    dispatched = []
    operation = BufferedGuardedOperation(
        name="test.transparent",
        input_projection=lambda _request: pytest.fail("input must not be projected"),
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
    assert checker.calls == ["policy"]


@pytest.mark.asyncio
async def test_input_and_output_checks_share_one_checker_and_context(operation):
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
    assert checker.calls[1][1].subject == GuardedText("user", "question")
    assert checker.calls[2] == "dispatch"
    assert checker.calls[3][0] == "output"
    assert checker.calls[3][1].input_subject == GuardedText("user", "question")
    assert checker.calls[3][1].output_subject == GuardedText("assistant", "answer")


@pytest.mark.asyncio
async def test_output_only_inspection_still_projects_effective_input_context(operation):
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
    assert checker.calls[-1][1].input_subject.content == "question"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "outcome_type"),
    [(ContentBlocked("blocked", "rule"), OperationBlocked), (ContentCheckFailed("failed"), OperationCheckFailed)],
)
async def test_input_stop_prevents_dispatch(operation, decision, outcome_type):
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
    checker = StaticChecker(input_decision=ContentAllowed(replacement="changed"))

    async def dispatch(_request):
        pytest.fail("a modified input must not be dispatched")

    with pytest.raises(UnsupportedContentModification, match="not supported"):
        await execute_buffered_operation(operation, checker, Request("question"), dispatch)


@pytest.mark.asyncio
async def test_output_modification_fails_without_returning_provider_response(operation):
    checker = StaticChecker(output_decision=ContentAllowed(replacement="changed"))
    response = Response("answer")

    async def dispatch(_request):
        return response

    with pytest.raises(UnsupportedContentModification, match="not supported"):
        await execute_buffered_operation(operation, checker, Request("question"), dispatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("projection", "message"),
    [
        (lambda _request: "question", "must return GuardedText"),
        (lambda _request: GuardedText("assistant", "question"), "user projection"),
    ],
)
async def test_invalid_input_projection_fails_before_dispatch(projection, message):
    operation = BufferedGuardedOperation(
        name="test.invalid_input",
        input_projection=projection,
        output_projection=lambda response: GuardedText("assistant", response.text),
    )

    async def dispatch(_request):
        pytest.fail("an invalid projection must not be dispatched")

    with pytest.raises((TypeError, ValueError), match=message):
        await execute_buffered_operation(operation, StaticChecker(), Request("question"), dispatch)


@pytest.mark.asyncio
async def test_unknown_checker_decision_fails_closed_before_dispatch(operation):
    checker = StaticChecker(input_decision=object())

    async def dispatch(_request):
        pytest.fail("an unknown checker decision must not be dispatched")

    with pytest.raises(TypeError, match="unsupported decision"):
        await execute_buffered_operation(operation, checker, Request("question"), dispatch)
