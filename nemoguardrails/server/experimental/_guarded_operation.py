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

"""Describe buffered provider operations and their checked content."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from nemoguardrails.server.experimental.provider.types import GuardedMessage

PayloadT = TypeVar("PayloadT", contravariant=True)
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class UnsupportedGuardedPayload(ValueError):
    """Report a provider payload outside an operation's guarded projection."""

    def __init__(self, message: str, code: str | None = None):
        """Store the stable provider-specific failure code when one is available."""
        super().__init__(message)
        self.code = code


class InvalidGuardedPayload(UnsupportedGuardedPayload):
    """Report malformed provider data that cannot enter guarded validation."""


class UnsupportedGuardedRepresentation(UnsupportedGuardedPayload):
    """Report an HTTP representation that the guarded payload cannot inspect."""

    code: str

    def __init__(self, message: str, code: str):
        """Store the required code for an unsupported HTTP representation."""
        super().__init__(message, code)


@dataclass(frozen=True, slots=True)
class ContentInspectionNotApplicable:
    """Declare that one provider response does not contain guarded output."""


class GuardedMessageProjection(Protocol[PayloadT]):
    """Extract one message to check from a provider-owned value."""

    def __call__(self, payload: PayloadT) -> GuardedMessage | ContentInspectionNotApplicable:
        """Return the message selected from one payload, when present."""

        ...


@dataclass(frozen=True, slots=True)
class BufferedGuardedOperation(Generic[RequestT, ResponseT]):
    """Describe the checked input and output of one buffered operation."""

    name: str
    input_projection: GuardedMessageProjection[RequestT]
    output_projection: GuardedMessageProjection[ResponseT]

    def __post_init__(self) -> None:
        """Validate the operation identity and its provider projections."""

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A guarded operation name must be a non-empty dotted identifier.")
        if not all(part.isidentifier() for part in self.name.split(".")):
            raise ValueError("A guarded operation name must be a non-empty dotted identifier.")
        for label, projection in (
            ("input", self.input_projection),
            ("output", self.output_projection),
        ):
            if not isinstance(projection, Callable):
                raise TypeError(f"The guarded operation {label} projection must be callable.")
