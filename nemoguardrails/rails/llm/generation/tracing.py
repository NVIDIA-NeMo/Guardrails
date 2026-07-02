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

"""Generation tracing option preparation and log restoration."""

from dataclasses import dataclass
from typing import Any, List, Optional

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationLogOptions,
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter

__all__ = [
    "GenerationTracingState",
    "create_startup_tracing_adapters",
    "export_generation_trace",
    "prepare_generation_tracing",
    "restore_generation_trace_log",
]


def create_startup_tracing_adapters(config: RailsConfig) -> list[InteractionLogAdapter] | None:
    """Build the tracing log adapters configured at startup (None when tracing is off)."""
    if not config.tracing:
        return None

    from nemoguardrails.tracing import create_log_adapters

    return create_log_adapters(config.tracing)


@dataclass
class GenerationTracingState:
    gen_options: Optional[GenerationOptions]
    original_log_options: Optional[GenerationLogOptions]


def prepare_generation_tracing(
    *,
    tracing_enabled: bool,
    gen_options: Optional[GenerationOptions],
) -> GenerationTracingState:
    """Prepare generation options needed for tracing without mutating callers."""
    if not tracing_enabled:
        return GenerationTracingState(
            gen_options=gen_options,
            original_log_options=None,
        )

    if gen_options is None:
        tracing_options = GenerationOptions()
    else:
        tracing_options = gen_options.model_copy(deep=True)

    original_log_options = tracing_options.log.model_copy(deep=True)
    tracing_options.log.activated_rails = True
    tracing_options.log.llm_calls = True
    tracing_options.log.internal_events = True

    return GenerationTracingState(
        gen_options=tracing_options,
        original_log_options=original_log_options,
    )


async def export_generation_trace(
    *,
    tracing_config: Any,
    log_adapters: Optional[List[Any]],
    messages: Optional[List[dict]],
    response: GenerationResponse,
) -> None:
    """Export tracing data for a completed generation response."""
    # Lazy import to avoid circular dependencies through eval/tracing modules.
    from nemoguardrails.tracing import Tracer

    span_format = getattr(tracing_config, "span_format", "opentelemetry")
    enable_content_capture = getattr(tracing_config, "enable_content_capture", False)
    tracer = Tracer(
        input=messages,
        response=response,
        adapters=log_adapters,
        span_format=span_format,
        enable_content_capture=enable_content_capture,
    )
    await tracer.export_async()


def restore_generation_trace_log(
    *,
    response: GenerationResponse,
    original_log_options: Optional[GenerationLogOptions],
) -> None:
    """Restore the response log fields requested by the caller before tracing."""
    if original_log_options is None:
        return

    if not any(
        (
            original_log_options.internal_events,
            original_log_options.activated_rails,
            original_log_options.llm_calls,
            original_log_options.colang_history,
        )
    ):
        response.log = None
        return

    if response.log is None:
        return

    if not original_log_options.internal_events:
        response.log.internal_events = []
    if not original_log_options.activated_rails:
        response.log.activated_rails = []
    if not original_log_options.llm_calls:
        response.log.llm_calls = []
