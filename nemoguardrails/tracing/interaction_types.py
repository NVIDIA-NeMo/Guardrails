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

"""Core models for the tracing system."""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from nemoguardrails.rails.llm.options import ActivatedRail, GenerationLog
from nemoguardrails.tracing.span_extractors import SpanExtractor, create_span_extractor
from nemoguardrails.tracing.spans import SpanLegacy, SpanOpentelemetry


class InteractionLog(BaseModel):
    """Detailed log about the execution of an interaction."""

    id: str = Field(description="A human readable id of the interaction.")

    activated_rails: List[ActivatedRail] = Field(
        default_factory=list, description="Details about the activated rails."
    )
    events: List[dict] = Field(
        default_factory=list,
        description="The full list of events recorded during the interaction.",
    )
    trace: List[Union[SpanLegacy, SpanOpentelemetry]] = Field(
        default_factory=list, description="Detailed information about the execution."
    )


class InteractionOutput(BaseModel):
    """Simple model for interaction output used in tracer."""

    id: str = Field(description="A human readable id of the interaction.")
    input: Any = Field(description="The input for the interaction.")
    output: Optional[Any] = Field(
        default=None, description="The output of the interaction."
    )


def extract_interaction_log(
    interaction_output: InteractionOutput,
    generation_log: GenerationLog,
    span_format: str = "opentelemetry",
    enable_content_capture: bool = False,
) -> InteractionLog:
    """Extracts an `InteractionLog` object from an `GenerationLog` object.

    Args:
        interaction_output: The interaction output
        generation_log: The generation log
        span_format: Span format to use ("legacy" or "opentelemetry")
        enable_content_capture: Whether to include content in trace events
    """
    internal_events = generation_log.internal_events

    span_extractor: SpanExtractor = create_span_extractor(
        span_format=span_format,
        events=internal_events,
        enable_content_capture=enable_content_capture,
    )

    spans = span_extractor.extract_spans(generation_log.activated_rails)

    return InteractionLog(
        id=interaction_output.id,
        activated_rails=generation_log.activated_rails,
        events=generation_log.internal_events or [],
        trace=spans,
    )
