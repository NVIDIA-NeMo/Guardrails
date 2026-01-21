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
    "LLMCallException",
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
