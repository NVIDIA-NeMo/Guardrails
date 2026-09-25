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

"""Describe provider fields and API revisions bound outside JSON bodies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from urllib.parse import parse_qs


class ProviderTransportRequest(Protocol):
    """Expose the HTTP metadata used by provider transport bindings."""

    @property
    def headers(self) -> Sequence[tuple[bytes, bytes]]:
        """Return the request headers as raw name/value pairs."""

        ...

    @property
    def query(self) -> bytes:
        """Return the raw request query string."""

        ...


class TransportLocation(str, Enum):
    """Identify where provider protocol data appears in an HTTP request."""

    PATH = "path"
    QUERY = "query"
    HEADER = "header"


class ProviderBindingViolation(ValueError):
    """Report request metadata outside a guarded endpoint binding."""

    def __init__(self, message: str, *, code: str):
        """Store the stable failure code for provider-native error mapping."""
        super().__init__(message)
        self.code = code


class ApiRevisionConstraint(ABC):
    """Define which client-selected API revisions a binding accepts."""

    @abstractmethod
    def accepts(self, revision: str | None) -> bool:
        """Return whether a received revision is accepted."""

        ...

    @abstractmethod
    def openapi_schema(self) -> dict[str, object]:
        """Describe accepted revisions as an OpenAPI string schema."""

        ...

    @property
    @abstractmethod
    def example(self) -> str:
        """Return one accepted revision for examples and tests."""

        ...

    @abstractmethod
    def describe(self) -> str:
        """Describe accepted revisions in validation errors."""

        ...


@dataclass(frozen=True, slots=True)
class ExactApiRevision(ApiRevisionConstraint):
    """Accept one exact provider API revision."""

    value: str

    def __post_init__(self) -> None:
        """Require a non-empty revision."""

        if not self.value.strip():
            raise ValueError("An exact provider API revision must not be empty.")

    def accepts(self, revision: str | None) -> bool:
        """Return whether the received revision is the reviewed value."""

        return revision == self.value

    def openapi_schema(self) -> dict[str, object]:
        """Describe the exact revision as an OpenAPI string schema."""

        return {"type": "string", "const": self.value}

    @property
    def example(self) -> str:
        """Return the accepted revision."""

        return self.value

    def describe(self) -> str:
        """Describe the accepted revision in validation errors."""

        return self.value


@dataclass(frozen=True, slots=True)
class ProviderApiRevisionBinding:
    """Bind a client-selected provider API revision to HTTP metadata."""

    accepted: ApiRevisionConstraint
    location: TransportLocation
    transport_name: str
    error_code: str = "unsupported_provider_api_revision"

    def __post_init__(self) -> None:
        """Validate revision transport metadata."""

        if self.location is TransportLocation.PATH:
            raise ValueError("Path API versions belong to guarded endpoint routes, not revision bindings.")
        if not self.transport_name.strip() or not self.error_code.strip():
            raise ValueError("Provider API revision binding values must not be empty.")

    def validate(self, request: ProviderTransportRequest) -> None:
        """Validate a client-selected header or query revision."""

        if self.location is TransportLocation.HEADER:
            encoded_name = self.transport_name.lower().encode("ascii")
            values = [value.decode("latin-1") for name, value in request.headers if name.lower() == encoded_name]
            received = values[-1] if values else None
        else:
            values = parse_qs(request.query.decode("ascii"), keep_blank_values=True).get(self.transport_name, [])
            received = values[-1] if values else None
        if not self.accepted.accepts(received):
            raise ProviderBindingViolation(
                f"The guarded route requires {self.transport_name}: {self.accepted.describe()}.",
                code=self.error_code,
            )

    def openapi_parameter(self) -> dict[str, object]:
        """Return the required OpenAPI revision parameter."""

        return {
            "name": self.transport_name,
            "in": self.location.value,
            "required": True,
            "schema": self.accepted.openapi_schema(),
        }
