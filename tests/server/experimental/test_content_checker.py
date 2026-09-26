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

"""Test private content-checker declarations and validation."""

import pytest

from nemoguardrails.server.experimental._content_checker import (
    ContentAllowed,
    ContentBlocked,
    ContentCheckFailed,
    ContentInspectionPolicy,
    InputContentCheck,
    InvalidContentChecker,
    OutputContentCheck,
    UnsupportedContentCheckerConfiguration,
    UnsupportedContentModification,
    validate_content_check_decision,
    validate_content_checker,
)
from nemoguardrails.server.experimental.provider.types import GuardedMessage


class StaticChecker:
    """Provide one deterministic checker for contract tests."""

    def __init__(self):
        self.policy_reads = 0

    def inspection_policy(self):
        self.policy_reads += 1
        return ContentInspectionPolicy(True, True)

    async def check_input(self, check):
        return ContentAllowed()

    async def check_output(self, check):
        return ContentAllowed()


def test_guarded_message_preserves_role_and_content():
    """Keep the provider-neutral role and content together."""

    assert GuardedMessage("user", "question").content == "question"
    assert GuardedMessage("assistant", "answer").content == "answer"


def test_output_check_preserves_source_contract():
    """Carry effective input context alongside generated output text."""

    check = OutputContentCheck(
        input_message=GuardedMessage("user", "question"),
        output_content="answer",
    )

    assert check.input_message.content == "question"
    assert check.output_content == "answer"


def test_source_unsupported_configuration_exception_is_available():
    """Keep unsupported checker configuration as a value error."""

    assert issubclass(UnsupportedContentCheckerConfiguration, ValueError)


def test_static_checker_is_validated_once_without_a_resolver():
    """Bind a statically supplied checker to one captured policy."""

    checker = StaticChecker()

    validated = validate_content_checker(checker)

    assert validated.checker is checker
    assert validated.policy == ContentInspectionPolicy(True, True)
    assert checker.policy_reads == 1


@pytest.mark.parametrize("missing_method", ["inspection_policy", "check_input", "check_output"])
def test_checker_validation_rejects_missing_methods(missing_method):
    """Reject objects missing any required checker operation."""

    checker = StaticChecker()
    setattr(checker, missing_method, None)

    with pytest.raises(InvalidContentChecker, match=missing_method):
        validate_content_checker(checker)


def test_checker_validation_rejects_unknown_policy():
    """Reject a checker whose policy has an unknown representation."""

    checker = StaticChecker()
    checker.inspection_policy = lambda: object()

    with pytest.raises(InvalidContentChecker, match="ContentInspectionPolicy"):
        validate_content_checker(checker)


@pytest.mark.parametrize(
    "decision",
    [ContentAllowed(), ContentBlocked("blocked", rule="policy"), ContentCheckFailed("failed")],
)
def test_supported_checker_decisions_are_valid(decision):
    """Accept every decision supported by the checker contract."""

    assert validate_content_check_decision(decision) is decision


def test_content_modification_is_explicitly_unsupported():
    """Reject a requested content replacement."""

    with pytest.raises(UnsupportedContentModification, match="not supported"):
        validate_content_check_decision(ContentAllowed(replacement="modified"))


def test_input_check_carries_the_guarded_message():
    """Pass the projected provider input to the checker unchanged."""

    message = GuardedMessage("user", "question")

    assert InputContentCheck(message).message is message
