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

"""Span extraction logic for different span versions."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from nemoguardrails.rails.llm.options import ActivatedRail
from nemoguardrails.tracing.constants import (
    EventNames,
    GuardrailsEventNames,
    GuardrailsEventTypes,
    OperationNames,
    SpanNames,
    SpanTypes,
    SystemConstants,
)
from nemoguardrails.tracing.spans import (
    ActionSpan,
    InteractionSpan,
    LLMSpan,
    RailSpan,
    SpanEvent,
    SpanLegacy,
    SpanOpentelemetry,
)
from nemoguardrails.utils import new_uuid


class SpanExtractor(ABC):
    """Base class for span extractors."""

    @abstractmethod
    def extract_spans(self, activated_rails: List[ActivatedRail]) -> List[Union[SpanLegacy, SpanOpentelemetry]]:
        """Extract spans from activated rails."""
        ...


class SpanExtractorV1(SpanExtractor):
    """Extract v1 spans (legacy format)."""

    def extract_spans(self, activated_rails: List[ActivatedRail]) -> List[Union[SpanLegacy, SpanOpentelemetry]]:
        """Extract v1 spans from activated rails."""
        spans: List[Union[SpanLegacy, SpanOpentelemetry]] = []
        if not activated_rails:
            return spans

        ref_time = activated_rails[0].started_at or 0.0

        # Create interaction span
        interaction_span = SpanLegacy(
            span_id=new_uuid(),
            name=SpanTypes.INTERACTION,  # V1 uses legacy naming
            start_time=(activated_rails[0].started_at or 0.0) - ref_time,
            end_time=(activated_rails[-1].finished_at or 0.0) - ref_time,
            duration=(activated_rails[-1].finished_at or 0.0) - (activated_rails[0].started_at or 0.0),
        )

        interaction_span.metrics.update(
            {
                "interaction_total": 1,
                "interaction_seconds_avg": interaction_span.duration,
                "interaction_seconds_total": interaction_span.duration,
            }
        )
        spans.append(interaction_span)

        # Process rails and actions
        for activated_rail in activated_rails:
            rail_span = SpanLegacy(
                span_id=new_uuid(),
                name="rail: " + activated_rail.name,
                parent_id=interaction_span.span_id,
                start_time=(activated_rail.started_at or 0.0) - ref_time,
                end_time=(activated_rail.finished_at or 0.0) - ref_time,
                duration=activated_rail.duration or 0.0,
            )
            spans.append(rail_span)

            for action in activated_rail.executed_actions:
                action_span = SpanLegacy(
                    span_id=new_uuid(),
                    name="action: " + action.action_name,
                    parent_id=rail_span.span_id,
                    start_time=(action.started_at or 0.0) - ref_time,
                    end_time=(action.finished_at or 0.0) - ref_time,
                    duration=action.duration or 0.0,
                )

                base_metric_name = f"action_{action.action_name}"
                action_span.metrics.update(
                    {
                        f"{base_metric_name}_total": 1,
                        f"{base_metric_name}_seconds_avg": action.duration or 0.0,
                        f"{base_metric_name}_seconds_total": action.duration or 0.0,
                    }
                )
                spans.append(action_span)

                # Process LLM calls
                for llm_call in action.llm_calls:
                    model_name = llm_call.llm_model_name or SystemConstants.UNKNOWN
                    llm_span = SpanLegacy(
                        span_id=new_uuid(),
                        name="LLM: " + model_name,
                        parent_id=action_span.span_id,
                        start_time=(llm_call.started_at or 0.0) - ref_time,
                        end_time=(llm_call.finished_at or 0.0) - ref_time,
                        duration=llm_call.duration or 0.0,
                    )

                    base_metric_name = f"llm_call_{model_name.replace('/', '_')}"
                    llm_span.metrics.update(
                        {
                            f"{base_metric_name}_total": 1,
                            f"{base_metric_name}_seconds_avg": llm_call.duration or 0.0,
                            f"{base_metric_name}_seconds_total": llm_call.duration or 0.0,
                            f"{base_metric_name}_prompt_tokens_total": llm_call.prompt_tokens or 0,
                            f"{base_metric_name}_completion_tokens_total": llm_call.completion_tokens or 0,
                            f"{base_metric_name}_tokens_total": llm_call.total_tokens or 0,
                        }
                    )
                    spans.append(llm_span)

        return spans


class SpanExtractorV2(SpanExtractor):
    """Extract v2 spans with OpenTelemetry semantic conventions."""

    def __init__(self, events: Optional[List[dict]] = None, enable_content_capture: bool = False):
        """Initialize with optional events for extracting user/bot messages.

        Args:
            events: Internal events from InteractionLog
            enable_content_capture: Whether to include potentially sensitive content in events
        """
        self.internal_events = events or []
        self.enable_content_capture = enable_content_capture

    def extract_spans(self, activated_rails: List[ActivatedRail]) -> List[Union[SpanLegacy, SpanOpentelemetry]]:
        """Extract v2 spans from activated rails with OpenTelemetry attributes."""
        spans: List[Union[SpanLegacy, SpanOpentelemetry]] = []
        ref_time = activated_rails[0].started_at or 0.0

        interaction_span = InteractionSpan(
            span_id=new_uuid(),
            name=SpanNames.GUARDRAILS_REQUEST,
            start_time=(activated_rails[0].started_at or 0.0) - ref_time,
            end_time=(activated_rails[-1].finished_at or 0.0) - ref_time,
            duration=(activated_rails[-1].finished_at or 0.0) - (activated_rails[0].started_at or 0.0),
            operation_name=OperationNames.GUARDRAILS,
            service_name=SystemConstants.SYSTEM_NAME,
        )
        spans.append(interaction_span)

        for activated_rail in activated_rails:
            # Create typed RailSpan
            rail_span = RailSpan(
                span_id=new_uuid(),
                name=SpanNames.GUARDRAILS_RAIL,  # Low-cardinality name
                parent_id=interaction_span.span_id,
                start_time=(activated_rail.started_at or 0.0) - ref_time,
                end_time=(activated_rail.finished_at or 0.0) - ref_time,
                duration=activated_rail.duration or 0.0,
                rail_type=activated_rail.type,
                rail_name=activated_rail.name,
                rail_stop=(activated_rail.stop if activated_rail.stop is not None else None),
                rail_decisions=(activated_rail.decisions if activated_rail.decisions else None),
            )
            spans.append(rail_span)

            for action in activated_rail.executed_actions:
                # Create typed ActionSpan
                action_span = ActionSpan(
                    span_id=new_uuid(),
                    name=SpanNames.GUARDRAILS_ACTION,
                    parent_id=rail_span.span_id,
                    start_time=(action.started_at or 0.0) - ref_time,
                    end_time=(action.finished_at or 0.0) - ref_time,
                    duration=action.duration or 0.0,
                    action_name=action.action_name,
                    has_llm_calls=len(action.llm_calls) > 0,
                    llm_calls_count=len(action.llm_calls),
                    action_params={
                        k: v for k, v in (action.action_params or {}).items() if isinstance(v, (str, int, float, bool))
                    },
                    # TODO: There is no error field in ExecutedAction. The fields below are defined on BaseSpan but
                    #  will never be set if using an ActivatedRail object to populate an ActivatedRail object.
                    error=None,
                    error_type=None,
                    error_message=None,
                )
                spans.append(action_span)

                for llm_call in action.llm_calls:
                    model_name = llm_call.llm_model_name or SystemConstants.UNKNOWN

                    provider_name = llm_call.llm_provider_name or SystemConstants.UNKNOWN

                    # use the specific task name as operation name (custom operation)
                    # this provides better observability for NeMo Guardrails specific tasks
                    operation_name = llm_call.task or OperationNames.COMPLETION

                    # follow OpenTelemetry convention: span name = "{operation} {model}"
                    span_name = f"{operation_name} {model_name}"

                    # extract request parameters from raw_response if available
                    temperature = None
                    max_tokens = None
                    top_p = None
                    response_id = None
                    finish_reasons = None

                    if llm_call.raw_response:
                        response_id = llm_call.raw_response.get("id")
                        finish_reasons = self._extract_finish_reasons(llm_call.raw_response)
                        temperature = llm_call.raw_response.get("temperature")
                        max_tokens = llm_call.raw_response.get("max_tokens")
                        top_p = llm_call.raw_response.get("top_p")

                    cache_hit = hasattr(llm_call, "from_cache") and llm_call.from_cache

                    llm_span = LLMSpan(
                        span_id=new_uuid(),
                        name=span_name,
                        parent_id=action_span.span_id,
                        start_time=(llm_call.started_at or 0.0) - ref_time,
                        end_time=(llm_call.finished_at or 0.0) - ref_time,
                        duration=llm_call.duration or 0.0,
                        provider_name=provider_name,
                        request_model=model_name,
                        response_model=model_name,
                        operation_name=operation_name,
                        usage_input_tokens=llm_call.prompt_tokens,
                        usage_output_tokens=llm_call.completion_tokens,
                        usage_total_tokens=llm_call.total_tokens,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        response_id=response_id,
                        response_finish_reasons=finish_reasons,
                        cache_hit=cache_hit,
                        # TODO: add error to LLMCallInfo for future release
                        # error=(
                        #     True
                        #     if hasattr(llm_call, "error") and llm_call.error
                        #     else None
                        # ),
                        # error_type=(
                        #     type(llm_call.error).__name__
                        #     if hasattr(llm_call, "error") and llm_call.error
                        #     else None
                        # ),
                        # error_message=(
                        #     str(llm_call.error)
                        #     if hasattr(llm_call, "error") and llm_call.error
                        #     else None
                        # ),
                    )

                    llm_events = self._extract_llm_events(llm_call, llm_span.start_time)
                    llm_span.events.extend(llm_events)

                    spans.append(llm_span)

        # Add conversation events to the interaction span
        if self.internal_events:
            interaction_events = self._extract_conversation_events(ref_time)
            interaction_span.events.extend(interaction_events)

        return spans

    def _extract_llm_events(self, llm_call, start_time: float) -> List[SpanEvent]:
        """Extract OpenTelemetry GenAI message events from an LLM call."""
        events = []

        # TODO: Update to use newer gen_ai.user.message and gen_ai.assistant.message events
        # Currently using deprecated gen_ai.content.prompt and gen_ai.content.completion for simplicity
        if llm_call.prompt:
            # per OTel spec: content should NOT be captured by default
            body = {"content": llm_call.prompt} if self.enable_content_capture else {}
            events.append(
                SpanEvent(
                    name=EventNames.GEN_AI_CONTENT_PROMPT,
                    timestamp=start_time,
                    body=body,
                )
            )

        if llm_call.completion:
            # per OTel spec: content should NOT be captured by default
            body = {"content": llm_call.completion} if self.enable_content_capture else {}
            events.append(
                SpanEvent(
                    name=EventNames.GEN_AI_CONTENT_COMPLETION,
                    timestamp=start_time + (llm_call.duration or 0),
                    body=body,
                )
            )

        return events

    def _extract_conversation_events(self, ref_time: float) -> List[SpanEvent]:
        """Extract guardrails-specific conversation events from internal events.

        NOTE: These are NeMo Guardrails internal events, NOT OpenTelemetry GenAI events.
        We use guardrails-specific namespacing to avoid confusion with OTel GenAI semantic conventions.
        """
        events = []

        for event in self.internal_events:
            event_type = event.get("type", "")
            body = dict()
            event_timestamp = self._get_event_timestamp(event, ref_time)

            if event_type == GuardrailsEventTypes.UTTERANCE_USER_ACTION_FINISHED:
                if self.enable_content_capture:
                    body["content"] = event.get("final_transcript", "")
                body["type"] = event_type
                events.append(
                    SpanEvent(
                        name=GuardrailsEventNames.UTTERANCE_USER_FINISHED,
                        timestamp=event_timestamp,
                        body=body,
                    )
                )

            elif event_type == GuardrailsEventTypes.USER_MESSAGE:
                if self.enable_content_capture:
                    body["content"] = event.get("text", "")
                body["type"] = event_type
                events.append(
                    SpanEvent(
                        name=GuardrailsEventNames.USER_MESSAGE,
                        timestamp=event_timestamp,
                        body=body,
                    )
                )

            elif event_type == GuardrailsEventTypes.START_UTTERANCE_BOT_ACTION:
                if self.enable_content_capture:
                    body["content"] = event.get("script", "")
                body["type"] = event_type
                events.append(
                    SpanEvent(
                        name=GuardrailsEventNames.UTTERANCE_BOT_STARTED,
                        timestamp=event_timestamp,
                        body=body,
                    )
                )
            elif event_type == GuardrailsEventTypes.UTTERANCE_BOT_ACTION_FINISHED:
                if self.enable_content_capture:
                    body["content"] = event.get("final_script", "")
                body["type"] = event_type
                body["is_success"] = event.get("is_success", True)
                events.append(
                    SpanEvent(
                        name=GuardrailsEventNames.UTTERANCE_BOT_FINISHED,
                        timestamp=event_timestamp,
                        body=body,
                    )
                )

        return events

    def _get_event_timestamp(self, event: dict, ref_time: float) -> float:
        """Extract timestamp from event or use reference time.

        Args:
            event: The internal event dictionary
            ref_time: Reference time to use as fallback (trace start time)

        Returns:
            Timestamp in seconds relative to trace start
        """
        event_created_at = event.get("event_created_at")
        if event_created_at:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(event_created_at)
                absolute_timestamp = dt.timestamp()
                return absolute_timestamp - ref_time
            except (ValueError, AttributeError):
                pass

        # fallback: use reference time (event at start of trace)
        return 0.0

    def _extract_finish_reasons(self, raw_response: dict) -> Optional[List[str]]:
        """Extract finish reasons from raw LLM response."""
        if not raw_response:
            return None

        finish_reasons = []

        if "finish_reason" in raw_response:
            finish_reasons.append(raw_response["finish_reason"])

        if not finish_reasons and raw_response:
            finish_reasons = ["stop"]

        return finish_reasons if finish_reasons else None


from nemoguardrails.tracing.span_format import SpanFormat, validate_span_format  # noqa: E402


def create_span_extractor(
    span_format: str = "legacy",
    events: Optional[List[dict]] = None,
    enable_content_capture: bool = True,
) -> SpanExtractor:
    """Create a span extractor based on format and configuration.

    Args:
        span_format: Format of span extractor ('legacy' or 'opentelemetry')
        events: Internal events for OpenTelemetry extractor
        enable_content_capture: Whether to capture content in spans

    Returns:
        Configured span extractor instance

    Raises:
        ValueError: If span_format is not supported
    """
    format_enum = validate_span_format(span_format)

    if format_enum == SpanFormat.LEGACY:
        return SpanExtractorV1()  # TODO: Rename to SpanExtractorLegacy
    elif format_enum == SpanFormat.OPENTELEMETRY:
        return SpanExtractorV2(  # TODO: Rename to SpanExtractorOTel
            events=events,
            enable_content_capture=enable_content_capture,
        )
    else:
        # This should never happen due to validation, but keeps type checker happy
        raise ValueError(f"Unknown span format: {span_format}")
