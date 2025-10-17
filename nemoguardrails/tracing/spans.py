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

"""Span models for NeMo Guardrails tracing system."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from nemoguardrails.tracing.constants import (
    CommonAttributes,
    GenAIAttributes,
    GuardrailsAttributes,
)


class SpanKind(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"


class SpanEvent(BaseModel):
    """Event that can be attached to a span."""

    name: str = Field(description="Event name (e.g., 'gen_ai.user.message')")
    timestamp: float = Field(description="Timestamp when the event occurred (relative)")
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Event attributes"
    )
    body: Optional[Dict[str, Any]] = Field(
        default=None, description="Event body for structured data"
    )


class SpanLegacy(BaseModel):
    """Simple span model (v1) for basic tracing."""

    span_id: str = Field(description="The id of the span.")
    name: str = Field(description="A human-readable name for the span.")
    parent_id: Optional[str] = Field(
        default=None, description="The id of the parent span."
    )
    resource_id: Optional[str] = Field(
        default=None, description="The id of the resource."
    )
    start_time: float = Field(description="The start time of the span.")
    end_time: float = Field(description="The end time of the span.")
    duration: float = Field(description="The duration of the span in seconds.")
    metrics: Dict[str, Union[int, float]] = Field(
        default_factory=dict, description="The metrics recorded during the span."
    )


class BaseSpan(BaseModel, ABC):
    """Base span with common fields across all span types."""

    span_id: str = Field(description="Unique identifier for this span")
    name: str = Field(description="Human-readable name for the span")
    parent_id: Optional[str] = Field(default=None, description="ID of the parent span")

    start_time: float = Field(
        description="Start time relative to trace start (seconds)"
    )
    end_time: float = Field(description="End time relative to trace start (seconds)")
    duration: float = Field(description="Duration of the span in seconds")

    span_kind: SpanKind = Field(description="OpenTelemetry span kind")

    events: List[SpanEvent] = Field(
        default_factory=list,
        description="Events attached to this span following OpenTelemetry conventions",
    )

    error: Optional[bool] = Field(default=None, description="Whether an error occurred")
    error_type: Optional[str] = Field(
        default=None, description="Type of error (e.g., exception class name)"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message or description"
    )

    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom attributes not covered by typed fields",
    )

    @abstractmethod
    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert typed fields to legacy OpenTelemetry attributes dictionary.

        Returns:
            Dict containing OTel semantic convention attributes.
        """
        pass

    def _base_attributes(self) -> Dict[str, Any]:
        """Get common attributes for all span types."""
        attributes = {
            CommonAttributes.SPAN_KIND: self.span_kind,
        }

        # TODO: for future release, consider adding:
        # if self.error is not None:
        #     attributes["error"] = self.error
        # if self.error_type is not None:
        #     attributes["error.type"] = self.error_type
        # if self.error_message is not None:
        #     attributes["error.message"] = self.error_message

        attributes.update(self.custom_attributes)

        return attributes


class InteractionSpan(BaseSpan):
    """Top-level span for a guardrails interaction (server span)."""

    span_kind: SpanKind = SpanKind.SERVER

    operation_name: str = Field(
        default="guardrails", description="Operation name for this interaction"
    )
    service_name: str = Field(default="nemo_guardrails", description="Service name")

    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    request_id: Optional[str] = Field(default=None, description="Request identifier")

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OTel attributes."""
        attributes = self._base_attributes()

        attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation_name
        attributes["service.name"] = self.service_name

        if self.user_id is not None:
            attributes["user.id"] = self.user_id
        if self.session_id is not None:
            attributes["session.id"] = self.session_id
        if self.request_id is not None:
            attributes["request.id"] = self.request_id

        return attributes


class RailSpan(BaseSpan):
    """Span for a guardrail execution (internal span)."""

    span_kind: SpanKind = SpanKind.INTERNAL
    # rail-specific attributes
    rail_type: str = Field(description="Type of rail (e.g., input, output, dialog)")
    rail_name: str = Field(description="Name of the rail (e.g., check_jailbreak)")
    rail_stop: Optional[bool] = Field(
        default=None, description="Whether the rail stopped execution"
    )
    rail_decisions: Optional[List[str]] = Field(
        default=None, description="Decisions made by the rail"
    )

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OTel attributes."""
        attributes = self._base_attributes()

        attributes[GuardrailsAttributes.RAIL_TYPE] = self.rail_type
        attributes[GuardrailsAttributes.RAIL_NAME] = self.rail_name

        if self.rail_stop is not None:
            attributes[GuardrailsAttributes.RAIL_STOP] = self.rail_stop
        if self.rail_decisions is not None:
            attributes[GuardrailsAttributes.RAIL_DECISIONS] = self.rail_decisions

        return attributes


class ActionSpan(BaseSpan):
    """Span for an action execution (internal span)."""

    span_kind: SpanKind = SpanKind.INTERNAL
    # action-specific attributes
    action_name: str = Field(description="Name of the action being executed")
    action_params: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters passed to the action"
    )
    has_llm_calls: bool = Field(
        default=False, description="Whether this action made LLM calls"
    )
    llm_calls_count: int = Field(
        default=0, description="Number of LLM calls made by this action"
    )

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OTel attributes."""
        attributes = self._base_attributes()

        attributes[GuardrailsAttributes.ACTION_NAME] = self.action_name
        attributes[GuardrailsAttributes.ACTION_HAS_LLM_CALLS] = self.has_llm_calls
        attributes[GuardrailsAttributes.ACTION_LLM_CALLS_COUNT] = self.llm_calls_count

        # add action parameters as individual attributes
        for param_name, param_value in self.action_params.items():
            if isinstance(param_value, (str, int, float, bool)):
                attributes[
                    f"{GuardrailsAttributes.ACTION_PARAM_PREFIX}{param_name}"
                ] = param_value

        return attributes


class LLMSpan(BaseSpan):
    """Span for an LLM API call (client span)."""

    span_kind: SpanKind = SpanKind.CLIENT
    provider_name: str = Field(
        description="LLM provider name (e.g., openai, anthropic)"
    )
    request_model: str = Field(description="Model requested (e.g., gpt-4)")
    response_model: str = Field(
        description="Model that responded (usually same as request_model)"
    )
    operation_name: str = Field(
        description="Operation name (e.g., chat.completions, embeddings)"
    )

    usage_input_tokens: Optional[int] = Field(
        default=None, description="Number of input tokens"
    )
    usage_output_tokens: Optional[int] = Field(
        default=None, description="Number of output tokens"
    )
    usage_total_tokens: Optional[int] = Field(
        default=None, description="Total number of tokens"
    )

    # Request parameters
    temperature: Optional[float] = Field(
        default=None, description="Temperature parameter"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(default=None, description="Top-p parameter")
    top_k: Optional[int] = Field(default=None, description="Top-k parameter")
    frequency_penalty: Optional[float] = Field(
        default=None, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=None, description="Presence penalty"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Stop sequences"
    )

    response_id: Optional[str] = Field(default=None, description="Response identifier")
    response_finish_reasons: Optional[List[str]] = Field(
        default=None, description="Finish reasons for each choice"
    )

    cache_hit: bool = Field(
        default=False,
        description="Whether this LLM response was served from application cache",
    )

    def to_otel_attributes(self) -> Dict[str, Any]:
        """Convert to OTel attributes."""
        attributes = self._base_attributes()

        attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] = self.provider_name
        attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] = self.request_model
        attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] = self.response_model
        attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] = self.operation_name

        if self.usage_input_tokens is not None:
            attributes[
                GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
            ] = self.usage_input_tokens
        if self.usage_output_tokens is not None:
            attributes[
                GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS
            ] = self.usage_output_tokens
        if self.usage_total_tokens is not None:
            attributes[
                GenAIAttributes.GEN_AI_USAGE_TOTAL_TOKENS
            ] = self.usage_total_tokens

        if self.temperature is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] = self.temperature
        if self.max_tokens is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] = self.max_tokens
        if self.top_p is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] = self.top_p
        if self.top_k is not None:
            attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] = self.top_k
        if self.frequency_penalty is not None:
            attributes[
                GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY
            ] = self.frequency_penalty
        if self.presence_penalty is not None:
            attributes[
                GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY
            ] = self.presence_penalty
        if self.stop_sequences is not None:
            attributes[
                GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES
            ] = self.stop_sequences

        if self.response_id is not None:
            attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] = self.response_id
        if self.response_finish_reasons is not None:
            attributes[
                GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
            ] = self.response_finish_reasons

        attributes[GuardrailsAttributes.LLM_CACHE_HIT] = self.cache_hit

        return attributes


TypedSpan = Union[InteractionSpan, RailSpan, ActionSpan, LLMSpan]

SpanOpentelemetry = TypedSpan


def is_opentelemetry_span(span: Any) -> bool:
    """Check if an object is a typed span (V2).

    Args:
        span: Object to check

    Returns:
        True if the object is a typed span, False otherwise
    """
    # Python 3.9 compatibility: cannot use isinstance with Union types
    return isinstance(span, (InteractionSpan, RailSpan, ActionSpan, LLMSpan))
