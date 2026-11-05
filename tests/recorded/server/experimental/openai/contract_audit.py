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

"""Strictly audit proxy exchanges independently of runtime compatibility policy."""

from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from nemoguardrails.server.experimental._json_payload import parse_json_object
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.payload import guarded_schema_error
from nemoguardrails.server.experimental.provider.sse import iter_sse_events
from nemoguardrails.server.experimental.provider.stream import StreamEventRole, UnsupportedProviderStream
from nemoguardrails.server.experimental.provider.types import UnknownContentFieldPolicy
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
    LITELLM_CHAT_COMPLETIONS_ENDPOINT,
)
from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import LITELLM_RESPONSES_ENDPOINT
from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT

CASSETTES = Path(__file__).parent / "cassettes"


def recorded_endpoint(path: Path, uri: str) -> GuardedJsonEndpoint:
    """Use the recording suite's explicit provider and operation selections."""
    directory = path.parent.name
    route = urlsplit(uri).path
    fixed_bindings = {
        "test_chat_completions": ("/v1/chat/completions", LITELLM_CHAT_COMPLETIONS_ENDPOINT),
        "test_responses": ("/v1/responses", RESPONSES_ENDPOINT),
        "test_litellm_responses": ("/v1/responses", LITELLM_RESPONSES_ENDPOINT),
    }
    if directory in fixed_bindings:
        expected_route, endpoint = fixed_bindings[directory]
        if route == expected_route:
            return endpoint
    elif directory == "test_forward_compatibility" and path.name.startswith("test_native_responses_"):
        if route == "/v1/responses":
            return RESPONSES_ENDPOINT
    elif directory == "test_gateway_routes" or (
        directory == "test_forward_compatibility" and path.name.startswith("test_gateway_")
    ):
        operations = {
            "/v1/chat/completions": LITELLM_CHAT_COMPLETIONS_ENDPOINT,
            "/v1/responses": LITELLM_RESPONSES_ENDPOINT,
        }
        if route in operations:
            return operations[route]
    raise ValueError(f"No reviewed proxy binding for recording {path.name!r} at {directory!r}")


async def audit_exchange(path: Path, uri: str, request_body: bytes | str, response: dict) -> None:
    """Report contract drift without echoing captured provider content."""
    try:
        await _audit_exchange(path, uri, request_body, response)
    except ValidationError as error:
        raise AssertionError(f"{path.name}: {guarded_schema_error(error, 'recorded payload')}") from None
    except UnsupportedProviderStream as error:
        raise AssertionError(f"{path.name}: {error}") from None


async def _audit_exchange(path: Path, uri: str, request_body: bytes | str, response: dict) -> None:
    """Audit the complete request and successful response, including stream state."""
    endpoint = recorded_endpoint(path, uri)
    if isinstance(request_body, str):
        request_body = request_body.encode()
    request = parse_json_object(request_body)
    projection = endpoint.guarded_request_model.validate_payload(
        request, unknown_content_fields=UnknownContentFieldPolicy.FORBID
    )
    if not 200 <= response["status"]["code"] < 300:
        return
    wire = response["body"]["string"]
    if isinstance(wire, str):
        wire = wire.encode()
    if not projection.streams_response:
        endpoint.guarded_response_model.validate_payload(
            parse_json_object(wire), unknown_content_fields=UnknownContentFieldPolicy.FORBID
        )
        return
    if endpoint.stream is None:
        raise ValueError("A streaming recording requires a reviewed stream binding")
    adapter = endpoint.stream.with_unknown_fields(UnknownContentFieldPolicy.FORBID).create_adapter()
    observed_text = ""

    async def chunks():
        yield wire

    async for event in iter_sse_events(chunks(), max_event_bytes=1024 * 1024):
        classified = adapter.classify_event(event)
        if classified.role is StreamEventRole.GUARDED_TEXT:
            observed_text += classified.text or ""
        elif classified.role is StreamEventRole.TEXT_SNAPSHOT and classified.text != observed_text:
            raise ValueError("Recorded snapshot differs from the preceding text deltas")
    adapter.validate_end_of_stream()
