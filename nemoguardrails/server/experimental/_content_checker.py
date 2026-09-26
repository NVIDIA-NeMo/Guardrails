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

"""Define private content-checking declarations for guarded operations."""

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast

from nemoguardrails.server.experimental.provider.types import GuardedMessage


class InvalidContentChecker(TypeError):
    """Report an object that does not satisfy the private checker boundary."""


class UnsupportedContentCheckerConfiguration(ValueError):
    """Report checker capabilities unsupported by the proxy pipeline."""


class UnsupportedContentModification(RuntimeError):
    """Report a checker modification that the proxy cannot safely apply."""


@dataclass(frozen=True, slots=True)
class ContentInspectionPolicy:
    """Record whether a checker examines input and output content."""

    inspect_input: bool
    inspect_output: bool

    def __post_init__(self) -> None:
        """Reject non-boolean inspection flags."""

        if not isinstance(self.inspect_input, bool) or not isinstance(self.inspect_output, bool):
            raise TypeError("Content inspection flags must be booleans.")


@dataclass(frozen=True, slots=True)
class InputContentCheck:
    """Carry a provider input subject to one checker."""

    message: GuardedMessage


@dataclass(frozen=True, slots=True)
class OutputContentCheck:
    """Carry generated text and its effective input context to one checker."""

    input_message: GuardedMessage
    output_content: str


@dataclass(frozen=True, slots=True)
class ContentAllowed:
    """Allow content without changing its provider representation."""

    replacement: str | None = None

    def __post_init__(self) -> None:
        """Reject replacement values that cannot represent text."""

        if self.replacement is not None and not isinstance(self.replacement, str):
            raise TypeError("A content replacement must be a string.")


@dataclass(frozen=True, slots=True)
class ContentBlocked:
    """Block content with a client-safe explanation and optional rule."""

    message: str
    rule: str | None = None

    def __post_init__(self) -> None:
        """Require a client-safe message and an optional non-empty rule."""

        if not isinstance(self.message, str) or not self.message:
            raise ValueError("A blocked-content message must be a non-empty string.")
        if self.rule is not None and (not isinstance(self.rule, str) or not self.rule):
            raise ValueError("A blocked-content rule must be a non-empty string when supplied.")


@dataclass(frozen=True, slots=True)
class ContentCheckFailed:
    """Report that checking could not produce a decision."""

    message: str
    cause: BaseException | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Require a message and preserve only exception-shaped causes."""

        if not isinstance(self.message, str) or not self.message:
            raise ValueError("A checker failure message must be a non-empty string.")
        if self.cause is not None and not isinstance(self.cause, BaseException):
            raise TypeError("A checker failure cause must be an exception when supplied.")


ContentCheckDecision: TypeAlias = ContentAllowed | ContentBlocked | ContentCheckFailed


class ContentChecker(Protocol):
    """Inspect provider-neutral input and output content."""

    def inspection_policy(self) -> ContentInspectionPolicy:
        """Return which input and output checks are enabled."""

        ...

    async def check_input(self, check: InputContentCheck) -> ContentCheckDecision:
        """Inspect one input message."""

        ...

    async def check_output(self, check: OutputContentCheck) -> ContentCheckDecision:
        """Inspect output text with its input context."""

        ...


@dataclass(frozen=True, slots=True)
class _ResolvedContentChecker:
    """Bind one checker to the policy validated for its execution."""

    checker: ContentChecker
    policy: ContentInspectionPolicy


def validate_content_checker(checker: object) -> _ResolvedContentChecker:
    """Validate one statically bound checker before operation execution."""

    for method_name in ("inspection_policy", "check_input", "check_output"):
        if not callable(getattr(checker, method_name, None)):
            raise InvalidContentChecker(f"The content checker must define callable {method_name}().")
    validated = cast(ContentChecker, checker)
    policy = validated.inspection_policy()
    if not isinstance(policy, ContentInspectionPolicy):
        raise InvalidContentChecker("The content checker must return a ContentInspectionPolicy.")
    return _ResolvedContentChecker(validated, policy)


def validate_content_check_decision(decision: object) -> ContentCheckDecision:
    """Validate one checker decision and reject unsupported modification."""

    if not isinstance(decision, (ContentAllowed, ContentBlocked, ContentCheckFailed)):
        raise TypeError("The content checker returned an unsupported decision.")
    if isinstance(decision, ContentAllowed) and decision.replacement is not None:
        raise UnsupportedContentModification("Content modification is not supported.")
    return decision
