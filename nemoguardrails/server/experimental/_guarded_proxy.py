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

"""Compose provider declarations with the guarded HTTP pipeline."""

from dataclasses import dataclass

from pydantic import ValidationError

from nemoguardrails.server.experimental._buffered_kernel import (
    InspectionStage,
    OperationProjectionFailed,
    PreparedOperationInput,
)
from nemoguardrails.server.experimental._guarded_operation import (
    BufferedGuardedOperation,
    ContentInspectionNotApplicable,
    InvalidGuardedPayload,
    UnsupportedGuardedPayload,
    UnsupportedGuardedRepresentation,
)
from nemoguardrails.server.experimental._guarded_stream import (
    StreamInspectionUnsupported,
    UnsupportedStreamInspection,
)
from nemoguardrails.server.experimental._http_kernel import (
    BufferedHttpRequest,
    BufferedHttpResponse,
    GuardedHttpOperation,
    GuardedOperationPath,
)
from nemoguardrails.server.experimental._json_payload import InvalidJson, UnsupportedJsonShape, parse_json_object
from nemoguardrails.server.experimental._streaming_http import StreamingHttpDispatch, execute_streaming_http
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.errors import ProviderErrorMapping
from nemoguardrails.server.experimental.provider.payload import GuardedMessageTarget, guarded_schema_error
from nemoguardrails.server.experimental.provider.projection_policy import PolicyModel, policy_json_schema
from nemoguardrails.server.experimental.provider.transport import ProviderBindingViolation
from nemoguardrails.server.experimental.provider.types import GuardedMessage


@dataclass(slots=True)
class GuardableProviderRequest:
    """Hold one original HTTP request and its validated exact guarded target."""

    request: BufferedHttpRequest
    target: GuardedMessageTarget
    streaming: bool = False


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
        raise UnsupportedGuardedPayload(str(error), endpoint.unsupported_request_code) from error
    try:
        projection = endpoint.guarded_request_model.validate_payload(payload)
        target = projection.locate_guarded_message(payload)
    except ValidationError as error:
        raise UnsupportedGuardedPayload(
            guarded_schema_error(error, "request"), endpoint.unsupported_request_code
        ) from error
    streaming = projection.streams_response
    if streaming and endpoint.stream_adapter_factory is None:
        raise UnsupportedGuardedPayload(
            "The guarded endpoint does not support streaming responses.", endpoint.unsupported_request_code
        )
    return GuardableProviderRequest(
        request=request,
        target=target,
        streaming=streaming,
    )


def create_guarded_http_operation(
    endpoint: GuardedJsonEndpoint,
    errors: ProviderErrorMapping,
    *,
    stream_dispatch: StreamingHttpDispatch | None = None,
    max_stream_event_bytes: int,
    max_pending_stream_bytes: int,
) -> GuardedHttpOperation:
    """Create one buffered-and-streaming operation from an endpoint contract."""

    def prepare_request(request: BufferedHttpRequest) -> GuardableProviderRequest:
        """Prepare one buffered HTTP request for guarded execution."""
        return _prepare_guardable_request(request, endpoint)

    def project_request(payload: GuardableProviderRequest) -> GuardedMessage:
        """Select guarded input text from a prepared provider request."""
        return payload.target.message

    def forward_request(request: GuardableProviderRequest) -> BufferedHttpRequest:
        """Forward the original HTTP request without reserializing its body."""
        return request.request

    def project_response(
        payload: BufferedHttpResponse,
    ) -> GuardedMessage | ContentInspectionNotApplicable:
        """Select guarded output text from a successful JSON response."""
        if not 200 <= payload.status_code < 300:
            return ContentInspectionNotApplicable()
        if _media_type(_header(payload.headers, b"content-type")) != "application/json":
            raise UnsupportedGuardedPayload(
                "The successful provider response must use application/json.", endpoint.unsupported_response_code
            )
        content_encoding = _header(payload.headers, b"content-encoding")
        if content_encoding is not None and content_encoding.strip().lower() not in {"", "identity"}:
            raise UnsupportedGuardedPayload(
                "Encoded successful provider responses are not supported.", endpoint.unsupported_response_code
            )
        try:
            document = parse_json_object(payload.body)
            projection = endpoint.guarded_response_model.validate_payload(document)
            target = projection.locate_guarded_message(document)
            # Validation has already rejected content in any other reviewed
            # field, so a response without text has nothing to inspect.
            if not target.has_text:
                return ContentInspectionNotApplicable()
            return target.message
        except ValidationError as error:
            raise UnsupportedGuardedPayload(
                guarded_schema_error(error, "successful response"), endpoint.unsupported_response_code
            ) from error
        except (InvalidJson, UnsupportedJsonShape) as error:
            raise UnsupportedGuardedPayload(str(error), endpoint.unsupported_response_code) from error

    request_model = endpoint.guarded_request_model
    request_schema = (
        policy_json_schema(request_model)
        if issubclass(request_model, PolicyModel)
        else request_model.model_json_schema()
    )

    async def handle_prepared_request(
        prepared_input: PreparedOperationInput[GuardableProviderRequest],
    ):
        guardable = prepared_input.request
        if not guardable.streaming:
            return None
        if stream_dispatch is None:
            return errors.renderer(
                OperationProjectionFailed(
                    InspectionStage.INPUT,
                    UnsupportedGuardedPayload("The guarded endpoint does not have streaming dispatch configured."),
                )
            )
        streaming_policy = prepared_input.resolved.policy.stream_buffering
        if prepared_input.resolved.policy.inspect_output and streaming_policy is None:
            return errors.renderer(
                StreamInspectionUnsupported(
                    UnsupportedStreamInspection("Guarded output streaming requires a buffering policy.")
                )
            )
        adapter_factory = endpoint.stream_adapter_factory
        if adapter_factory is None:
            raise AssertionError("A streaming request requires an endpoint stream adapter.")
        return await execute_streaming_http(
            guardable.request,
            dispatch=stream_dispatch,
            checker=prepared_input.resolved.checker,
            streaming_policy=streaming_policy,
            input_message=guardable.target.message,
            adapter=adapter_factory(),
            render_outcome=errors.renderer,
            max_event_bytes=max_stream_event_bytes,
            max_pending_bytes=max_pending_stream_bytes,
        )

    success_content = {"application/json": {"schema": {}}}
    if endpoint.stream_adapter_factory is not None and stream_dispatch is not None:
        success_content["text/event-stream"] = {"schema": {"type": "string"}}
    documented_responses = errors.documented_responses
    documented_responses[200] = {
        "description": "Provider-native successful response.",
        "content": success_content,
    }
    openapi_extra = {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": request_schema,
                }
            },
        }
    }
    if endpoint.api_revision is not None:
        openapi_extra["parameters"] = [endpoint.api_revision.openapi_parameter()]

    operation: BufferedGuardedOperation[GuardableProviderRequest, BufferedHttpResponse] = BufferedGuardedOperation(
        name=endpoint.operation_name,
        input_projection=project_request,
        output_projection=project_response,
    )

    return GuardedHttpOperation(
        operation_path=GuardedOperationPath(endpoint.route_path, frozenset({endpoint.method})),
        operation=operation,
        prepare_request=prepare_request,
        forward_request=forward_request,
        prepared_request_handler=handle_prepared_request,
        documented_responses=documented_responses,
        guarded_operation_paths=endpoint.operation_paths,
        openapi_extra=openapi_extra,
    )
