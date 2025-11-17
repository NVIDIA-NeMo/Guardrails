# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Custom exceptions for NeMo Guardrails.

These exceptions represent errors that the SDK detects and raises.
Provider errors (authentication, network, rate limits, etc.) are NOT wrapped
and should bubble up unchanged to the consuming application.
"""

from typing import Any, Dict, Optional


class NemoGuardrailsError(Exception):
    """Base exception for all NeMo Guardrails SDK errors.

    Attributes:
        message: Human-readable error message
        context: Additional context about the error (model names, config details, etc.)
    """

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            context: Additional context (e.g., model name, provider, file path, etc.)
        """
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        """Return a formatted error message with context."""
        parts = [super().__str__()]

        if self.context:
            context_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        return "\n".join(parts)


# ============================================================================
# Configuration Errors (Creation-Time)
# ============================================================================


class ConfigurationError(NemoGuardrailsError):
    """Base class for configuration-related errors.

    These errors occur when loading or validating guardrail configurations,
    typically during RailsConfig loading or LLMRails initialization.
    """

    pass


class InvalidModelConfigurationError(ConfigurationError):
    """Raised when model configuration is invalid.

    Examples:
        - model.model is empty or missing
        - model.engine is invalid
        - Invalid model parameters
        - Model type not recognized
        - Base URL is malformed
    """

    pass


class InvalidConfigurationFileError(ConfigurationError):
    """Raised when configuration files cannot be successfullyparsed.

    Examples:
        - Invalid YAML syntax in config.yml
        - Invalid Colang syntax in .co files
        - Circular import dependencies
        - File not found
    """

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


# ============================================================================
# Runtime Errors (Inference-Time)
# ============================================================================


class ModelAuthenticationError(NemoGuardrailsError):
    """Raised at inference-time when an authentication error occurs when calling the model."""

    pass


class LLMInvocationError(NemoGuardrailsError):
    """Base class for runtime errors that occur during inference.

    These errors occur after configuration has been loaded and validated,
    during the actual execution of guardrails.
    """

    pass


class MalformedLLMResponseError(LLMInvocationError):
    """Raised when the SDK cannot parse an LLM response.

    Examples:
        - NemoGuard JSON parsing failures
        - Structured output doesn't match expected format
        - LLM returns invalid JSON when JSON is required
        - Response missing required fields
    """

    pass


class InvalidConfigurationError(LLMInvocationError):
    """Raised when an invalid configuration is detected at runtime.

    Examples:
        - Dynamic configuration issues discovered during inference
        - Rail references a flow or action that doesn't exist (only discovered at runtime)
        - Configuration state becomes invalid during execution
    """

    pass


class ActionExecutionError(LLMInvocationError):
    """Raised when a custom action fails during execution.

    Examples:
        - User-defined custom action raised an exception
        - Action returned invalid format
        - Action parameter validation failed at runtime
        - Action depends on external resource that's unavailable
    """

    pass
