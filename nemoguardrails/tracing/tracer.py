# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import uuid
from contextlib import AsyncExitStack
from typing import List, Optional

from nemoguardrails.rails.llm.config import TracingConfig
from nemoguardrails.rails.llm.options import GenerationLog, GenerationResponse
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing.adapters.registry import LogAdapterRegistry
from nemoguardrails.tracing.interaction_types import (
    InteractionLog,
    InteractionOutput,
    extract_interaction_log,
)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Tracer:
    def __init__(
        self,
        input,
        response: GenerationResponse,
        adapters: Optional[List[InteractionLogAdapter]] = None,
        span_format: str = "opentelemetry",
        enable_content_capture: bool = False,
    ):
        self._interaction_output = InteractionOutput(
            id=new_uuid(), input=input[-1]["content"], output=response.response
        )
        self._generation_log = response.log
        self.adapters = []
        if self._generation_log is None:
            raise RuntimeError("Generation log is missing.")

        self.adapters = adapters or []
        self._span_format = span_format
        self._enable_content_capture = enable_content_capture

    def generate_interaction_log(
        self,
        interaction_output: Optional[InteractionOutput] = None,
        generation_log: Optional[GenerationLog] = None,
    ) -> InteractionLog:
        """Generates an InteractionLog from the given interaction output and generation log."""
        if interaction_output is None:
            interaction_output = self._interaction_output

        if generation_log is None:
            generation_log = self._generation_log

        # At this point generation_log should not be None since it comes from self._generation_log
        if generation_log is None:
            raise Exception("Can't generate interaction log without Generation log")

        interaction_log = extract_interaction_log(
            interaction_output,
            generation_log,
            span_format=self._span_format,
            enable_content_capture=self._enable_content_capture,
        )
        return interaction_log

    def add_adapter(self, adapter: InteractionLogAdapter):
        """Adds an adapter to the tracer."""
        self.adapters.append(adapter)

    def export(self):
        """Exports the interaction log using the configured adapters."""
        interaction_log = self.generate_interaction_log()
        for adapter in self.adapters:
            adapter.transform(interaction_log)

    async def export_async(self):
        """Exports the interaction log using the configured adapters."""
        interaction_log = self.generate_interaction_log()

        async with AsyncExitStack() as stack:
            for adapter in self.adapters:
                await stack.enter_async_context(adapter)

            # Transform the interaction logs asynchronously with use of all adapters
            tasks = [
                adapter.transform_async(interaction_log) for adapter in self.adapters
            ]
            await asyncio.gather(*tasks)


def create_log_adapters(config: TracingConfig) -> List[InteractionLogAdapter]:
    adapters = []
    if config.enabled:
        adapter_configs = config.adapters
        if adapter_configs:
            for adapter_config in adapter_configs:
                log_adapter_cls = LogAdapterRegistry().get(adapter_config.name)
                log_adapter_args = adapter_config.model_dump()
                log_adapter_args.pop("name", None)
                log_adapter = log_adapter_cls(**log_adapter_args)
                adapters.append(log_adapter)
    return adapters
