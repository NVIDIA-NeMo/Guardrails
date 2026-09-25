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

"""Compose provider declarations with the guarded buffered HTTP pipeline."""

from dataclasses import dataclass, replace

from pydantic import ValidationError

from nemoguardrails.server.experimental._guarded_operation import (
    BufferedGuardedOperation,
    ContentInspectionNotApplicable,
    InvalidGuardedPayload,
    UnsupportedGuardedPayload,
    UnsupportedGuardedRepresentation,
)
from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpRequest,
    BufferedHttpResponse,
    GuardedHttpOperation,
    GuardedOperationPath,
)
from nemoguardrails.server.experimental._json_payload import InvalidJson, UnsupportedJsonShape, parse_json_object
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.errors import ProviderErrorMapping
from nemoguardrails.server.experimental.provider.payload import GuardedMessageTarget, guarded_schema_error
from nemoguardrails.server.experimental.provider.transport import ProviderBindingViolation
from nemoguardrails.server.experimental.provider.types import GuardedMessage, JsonObject


@dataclass(slots=True)
class GuardableProviderRequest:
    """Hold one original HTTP request and its validated exact guarded target."""

    request: BufferedHttpRequest
    raw_body: bytes
    payload: JsonObject
    target: GuardedMessageTarget
    streaming: bool


@dataclass(frozen=True, slots=True)
class PreparedProviderRequest:
    """Hold the exact provider request state after input inspection."""

    body: bytes
    input_message: GuardedMessage
    body_modified: bool
    streaming: bool


def _header(headers: tuple[tuple[bytes, bytes], ...], name: bytes) -> str | None:
    """Return the last value of a case-insensitive HTTP header."""
    values = [value.decode("latin-1") for key, value in headers if key.lower() == name]
    return values[-1] if values else None


def _media_type(value: str | None) -> str | None:
    """Return a normalized media type without parameters."""
    return value.split(";", 1)[0].strip().lower() if value is not None else None


def _require_json_representation(request: BufferedHttpRequest) -> None:
    """Require an unencoded JSON representation for guarded request parsing."""
    if _media_type(_header(request.headers, b"content-type")) != "application/json":
        raise UnsupportedGuardedRepresentation(
            "Content-Type must be application/json.",
            "unsupported_media_type",
        )
    content_encoding = _header(request.headers, b"content-encoding")
    if content_encoding is not None and content_encoding.strip().lower() not in {"", "identity"}:
        raise UnsupportedGuardedRepresentation(
            "Encoded request bodies are not supported by the guarded path.",
            "unsupported_content_encoding",
        )


def _prepare_guardable_request(
    request: BufferedHttpRequest,
    endpoint: GuardedJsonEndpoint,
) -> GuardableProviderRequest:
    """Validate the request and locate its exact guarded text target."""
    if endpoint.api_revision is not None:
        try:
            endpoint.api_revision.validate(request)
        except ProviderBindingViolation as error:
            raise UnsupportedGuardedPayload(str(error), error.code) from error
    _require_json_representation(request)
    try:
        payload = parse_json_object(request.body)
    except InvalidJson as error:
        raise InvalidGuardedPayload("The request body must be valid JSON.") from error
    except UnsupportedJsonShape as error:
        raise UnsupportedGuardedPayload(str(error)) from error
    try:
        projection = endpoint.guarded_request_model.validate_payload(payload)
        target = projection.locate_guarded_message(payload)
    except ValidationError as error:
        raise UnsupportedGuardedPayload(guarded_schema_error(error, "request")) from error
    if projection.streams_response:
        raise UnsupportedGuardedPayload("The guarded endpoint does not support streaming responses yet.")
    return GuardableProviderRequest(
        request=request,
        raw_body=request.body,
        payload=payload,
        target=target,
        streaming=False,
    )


def _prepare_provider_request(request: GuardableProviderRequest) -> PreparedProviderRequest:
    """Preserve the original provider body after input inspection."""
    return PreparedProviderRequest(
        body=request.raw_body,
        input_message=request.target.message,
        body_modified=False,
        streaming=request.streaming,
    )


def create_buffered_guarded_http_operation(
    endpoint: GuardedJsonEndpoint,
    errors: ProviderErrorMapping,
) -> GuardedHttpOperation:
    """Create one buffered HTTP operation from the provider endpoint contract."""

    def prepare_request(request: BufferedHttpRequest) -> GuardableProviderRequest:
        """Prepare one buffered HTTP request for guarded execution."""
        return _prepare_guardable_request(request, endpoint)

    def project_request(payload: GuardableProviderRequest) -> GuardedMessage:
        """Select guarded input text from a prepared provider request."""
        return payload.target.message

    def forward_request(request: GuardableProviderRequest) -> BufferedHttpRequest:
        """Return the provider request that should be dispatched."""
        prepared = _prepare_provider_request(request)
        return replace(request.request, body=prepared.body)

    def project_response(
        payload: BufferedHttpResponse,
    ) -> GuardedMessage | ContentInspectionNotApplicable:
        """Select guarded output text from a successful JSON response."""
        if not 200 <= payload.status_code < 300:
            return ContentInspectionNotApplicable()
        if _media_type(_header(payload.headers, b"content-type")) != "application/json":
            raise UnsupportedGuardedPayload("The successful provider response must use application/json.")
        content_encoding = _header(payload.headers, b"content-encoding")
        if content_encoding is not None and content_encoding.strip().lower() not in {"", "identity"}:
            raise UnsupportedGuardedPayload("Encoded successful provider responses are not supported.")
        try:
            document = parse_json_object(payload.body)
            projection = endpoint.guarded_response_model.validate_payload(document)
            return projection.locate_guarded_message(document).message
        except ValidationError as error:
            raise UnsupportedGuardedPayload(guarded_schema_error(error, "successful response")) from error
        except (InvalidJson, UnsupportedJsonShape) as error:
            raise UnsupportedGuardedPayload(str(error)) from error

    openapi_extra = {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": endpoint.guarded_request_model.model_json_schema(),
                }
            },
        }
    }
    if endpoint.api_revision is not None:
        openapi_extra["parameters"] = [endpoint.api_revision.openapi_parameter()]

    return GuardedHttpOperation(
        operation_path=GuardedOperationPath(endpoint.route_path, frozenset({endpoint.method})),
        operation=BufferedGuardedOperation[GuardableProviderRequest, BufferedHttpResponse](
            name=endpoint.operation_name,
            input_projection=project_request,
            output_projection=project_response,
        ),
        prepare_request=prepare_request,
        forward_request=forward_request,
        documented_responses=errors.documented_responses,
        guarded_operation_paths=endpoint.operation_paths,
        openapi_extra=openapi_extra,
    )
