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

"""Declare provider endpoints handled by the guarded JSON pipeline."""

from dataclasses import dataclass

from nemoguardrails.server.experimental._http_kernel import GuardedOperationPath
from nemoguardrails.server.experimental.provider.payload import (
    GuardedPayloadModel,
    GuardedRequestModel,
    validate_payload_projection_contract,
)
from nemoguardrails.server.experimental.provider.transport import ProviderApiRevisionBinding


@dataclass(frozen=True, slots=True)
class GuardedJsonEndpoint:
    """Describe one provider route and the content fields it guards."""

    route_path: str
    operation_name: str
    operation: str
    unsupported_request_code: str
    unsupported_response_code: str
    guarded_request_model: type[GuardedRequestModel]
    guarded_response_model: type[GuardedPayloadModel]
    method: str = "POST"
    api_revision: ProviderApiRevisionBinding | None = None
    operation_paths: tuple[GuardedOperationPath, ...] = ()

    def __post_init__(self) -> None:
        """Validate the route, projections, and shared capability profile."""
        if not self.operation_name or not all(part.isidentifier() for part in self.operation_name.split(".")):
            raise ValueError("A guarded endpoint operation name must contain only dotted identifiers.")
        if not self.operation.strip():
            raise ValueError("A guarded endpoint operation label must not be empty.")
        if not self.unsupported_request_code.strip() or not self.unsupported_response_code.strip():
            raise ValueError("A guarded endpoint must declare non-empty unsupported-shape error codes.")
        if not self.method or self.method != self.method.upper():
            raise ValueError("A guarded endpoint method must be a non-empty uppercase name.")
        operation_paths = self.operation_paths or (GuardedOperationPath(self.route_path, frozenset({self.method})),)
        if not any(
            operation_path.matches(self.route_path) and self.method in operation_path.methods
            for operation_path in operation_paths
        ):
            raise ValueError("A guarded endpoint route and method must belong to one guarded operation path.")
        object.__setattr__(self, "operation_paths", operation_paths)
        if not self.guarded_request_model.has_guarded_text_binding():
            raise ValueError("The guarded request projection must declare its guarded text location.")
        if not self.guarded_request_model.has_response_mode_binding():
            raise ValueError("The guarded request projection must declare its provider response mode.")
        if not self.guarded_response_model.has_guarded_text_binding():
            raise ValueError("The guarded response projection must declare its guarded text location.")
        request_contract = validate_payload_projection_contract(self.guarded_request_model, "request")
        response_contract = validate_payload_projection_contract(self.guarded_response_model, "response")
        if request_contract.profile != response_contract.profile:
            raise ValueError("Guarded request and response capability profiles must match.")
