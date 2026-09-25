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

from dataclasses import dataclass

import pytest

from nemoguardrails.server.experimental.provider.transport import (
    ExactApiRevision,
    ProviderApiRevisionBinding,
    ProviderBindingViolation,
    TransportLocation,
)


@dataclass(frozen=True)
class RequestMetadata:
    """Provide the request metadata consumed by revision bindings."""

    headers: tuple[tuple[bytes, bytes], ...] = ()
    query: bytes = b""


@pytest.mark.parametrize(
    ("location", "transport_name", "metadata"),
    [
        (
            TransportLocation.HEADER,
            "provider-version",
            RequestMetadata(headers=((b"provider-version", b"2026-09-25"),)),
        ),
        (
            TransportLocation.QUERY,
            "api-version",
            RequestMetadata(query=b"api-version=2026-09-25"),
        ),
    ],
)
def test_provider_api_revision_binding_accepts_exact_header_or_query_revision(location, transport_name, metadata):
    """An exact provider revision may be bound to a header or query value."""
    binding = ProviderApiRevisionBinding(
        accepted=ExactApiRevision("2026-09-25"),
        location=location,
        transport_name=transport_name,
    )

    binding.validate(metadata)


def test_provider_api_revision_binding_rejects_missing_revision_with_stable_code():
    """A missing required revision reports a stable provider error code."""
    binding = ProviderApiRevisionBinding(
        accepted=ExactApiRevision("2026-09-25"),
        location=TransportLocation.HEADER,
        transport_name="provider-version",
    )

    with pytest.raises(ProviderBindingViolation) as exc_info:
        binding.validate(RequestMetadata())

    assert exc_info.value.code == "unsupported_provider_api_revision"
    assert binding.openapi_parameter() == {
        "name": "provider-version",
        "in": "header",
        "required": True,
        "schema": {"type": "string", "const": "2026-09-25"},
    }


def test_provider_api_revision_binding_rejects_path_location():
    """Path revisions belong to route ownership rather than metadata bindings."""
    with pytest.raises(ValueError, match="belong to guarded endpoint routes"):
        ProviderApiRevisionBinding(
            accepted=ExactApiRevision("v1"),
            location=TransportLocation.PATH,
            transport_name="api_version",
        )
