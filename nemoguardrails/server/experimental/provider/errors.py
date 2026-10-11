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

"""Bind provider-native runtime errors to their documented response models."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from nemoguardrails.server.experimental._http_kernel import OutcomeRenderer


@dataclass(frozen=True, slots=True)
class ProviderErrorResponse:
    """Describe one provider-native proxy error response."""

    status_code: int
    model: type[BaseModel]
    description: str


@dataclass(frozen=True, slots=True)
class ProviderErrorMapping:
    """Make one provider error authority serve runtime and OpenAPI."""

    renderer: OutcomeRenderer
    responses: tuple[ProviderErrorResponse, ...]

    def __post_init__(self) -> None:
        """Require one documented model for each response status code."""
        status_codes = [response.status_code for response in self.responses]
        if len(status_codes) != len(set(status_codes)):
            raise ValueError("Provider error response status codes must be unique.")

    @property
    def documented_responses(self) -> dict[int | str, dict[str, Any]]:
        """Return FastAPI response declarations from the shared error authority."""
        return {
            response.status_code: {
                "model": response.model,
                "description": response.description,
            }
            for response in self.responses
        }
