# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional, Union

__all__ = [
    "ConfigurationError",
    "InvalidModelConfigurationError",
    "InvalidRailsConfigurationError",
    "InvalidStateError",
    "LLMCallException",
    "LLMClientError",
    "LLMAuthenticationError",
    "LLMRateLimitError",
    "LLMBadRequestError",
    "LLMContextWindowError",
    "LLMUnsupportedParamsError",
    "LLMServerError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "LLMResponseValidationError",
    "StreamingNotSupportedError",
]


class ConfigurationError(ValueError):
    """
    Base class for Guardrails Configuration validation errors.
    """

    pass


class InvalidModelConfigurationError(ConfigurationError):
    """Raised when a guardrail configuration's model is invalid."""

    pass


class InvalidRailsConfigurationError(ConfigurationError):
    """Raised when rails configuration is invalid.

    Examples:
        - Input/output rail references a model that doesn't exist in config
        - Rail references a flow that doesn't exist
        - Missing required prompt template
        - Invalid rail parameters
    """

    pass


class StreamingNotSupportedError(InvalidRailsConfigurationError):
    """Raised when streaming is requested but not supported by the configuration."""

    pass


class InvalidStateError(ValueError):
    """Raised when a caller-supplied `state` argument is not valid public input.

    The serialized Colang 2.0 runtime State carries trusted control-plane fields
    (`flow_configs`, `rails_config`, in-flight flow execution) and must not come
    from an untrusted caller. Stateful 2.x execution uses `process_events_async`,
    which keeps a live `State` object in the trusted Python process.
    """

    pass


class LLMCallException(Exception):
    """A wrapper around the LLM call invocation exception.

    This is used to propagate the exception out of the `generate_async` call. The default behavior is to
    catch it and return an "Internal server error." message.
    """

    inner_exception: Union[BaseException, str]
    detail: Optional[str]

    def __init__(self, inner_exception: Union[BaseException, str], detail: Optional[str] = None):
        """Initialize LLMCallException.

        Args:
            inner_exception: The original exception that occurred
            detail: Optional context to prepend (for example, the model name or endpoint)
        """
        message = f"{detail or 'LLM Call Exception'}: {str(inner_exception)}"
        super().__init__(message)

        self.inner_exception = inner_exception
        self.detail = detail


class LLMClientError(Exception):
    """Base class for LLM client errors.

    ``status_code`` holds the HTTP response status when one was received,
    or ``0`` when no response arrived (client-side timeout or network
    error). Callers should branch on exception class rather than
    ``status_code`` to distinguish HTTP vs network failures, the type
    hierarchy is the authoritative discriminator.
    """

    def __init__(
        self,
        status_code: int,
        error_message: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        param: Optional[str] = None,
        body: Optional[dict] = None,
        response_headers: Optional[dict] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.status_code = status_code
        self.error_message = error_message
        self.error_type = error_type
        self.error_code = error_code
        self.param = param
        self.body = body
        self.response_headers = response_headers
        self.model_name = model_name
        self.provider_name = provider_name
        self.base_url = base_url
        super().__init__(f"[{status_code}] {error_message}" if status_code > 0 else error_message)

    def __str__(self) -> str:
        parts = []
        if self.model_name:
            parts.append(f"model={self.model_name}")
        if self.provider_name:
            parts.append(f"provider={self.provider_name}")
        if self.base_url:
            parts.append(f"endpoint={self.base_url}")
        context = f" ({', '.join(parts)})" if parts else ""
        prefix = f"[{self.status_code}]" if self.status_code > 0 else ""
        return f"{prefix}{context} {self.error_message}".strip()


class LLMAuthenticationError(LLMClientError):
    pass


class LLMRateLimitError(LLMClientError):
    def __init__(
        self,
        status_code: int,
        error_message: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        param: Optional[str] = None,
        body: Optional[dict] = None,
        response_headers: Optional[dict] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        base_url: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
    ):
        super().__init__(
            status_code,
            error_message,
            error_type,
            error_code,
            param,
            body,
            response_headers,
            model_name,
            provider_name,
            base_url,
        )
        self.retry_after_seconds = retry_after_seconds


class LLMBadRequestError(LLMClientError):
    pass


class LLMContextWindowError(LLMBadRequestError):
    pass


class LLMUnsupportedParamsError(LLMBadRequestError):
    pass


class LLMServerError(LLMClientError):
    pass


class LLMTimeoutError(LLMClientError):
    pass


class LLMConnectionError(LLMClientError):
    pass


class LLMResponseValidationError(LLMClientError):
    def __init__(
        self,
        message: str,
        response_data: Optional[dict] = None,
        model_name: Optional[str] = None,
        provider_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.response_data = response_data
        super().__init__(
            status_code=0,
            error_message=message,
            body=response_data,
            model_name=model_name,
            provider_name=provider_name,
            base_url=base_url,
        )
