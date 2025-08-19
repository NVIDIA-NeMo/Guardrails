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

import pytest

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import GenerationOptions
from nemoguardrails.tracing import SpanOpentelemetry, create_span_extractor
from nemoguardrails.tracing.spans import LLMSpan, is_opentelemetry_span
from tests.utils import FakeLLM


@pytest.fixture
def v2_config():
    return RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: openai
    model: gpt-4

tracing:
  enabled: true
  span_format: opentelemetry
  adapters: []
"""
    )


@pytest.fixture
def v1_config():
    return RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: openai
    model: gpt-4

tracing:
  enabled: true
  span_format: legacy
  adapters: []
"""
    )


@pytest.fixture
def default_config():
    return RailsConfig.from_content(
        yaml_content="""
models:
  - type: main
    engine: openai
    model: gpt-4

tracing:
  enabled: true
  adapters: []
"""
    )


def test_span_v2_configuration(v2_config):
    assert v2_config.tracing.span_format == "opentelemetry"

    llm = FakeLLM(responses=["Hello! I'm here to help."])
    _rails = LLMRails(config=v2_config, llm=llm)

    extractor = create_span_extractor(span_format="opentelemetry")
    assert extractor.__class__.__name__ == "SpanExtractorV2"


@pytest.mark.asyncio
async def test_v2_spans_generated_with_events(v2_config):
    llm = FakeLLM(responses=["  express greeting", "Hello! How can I help you today?"])

    rails = LLMRails(config=v2_config, llm=llm)

    options = GenerationOptions(
        log={"activated_rails": True, "internal_events": True, "llm_calls": True}
    )

    response = await rails.generate_async(
        messages=[{"role": "user", "content": "Hello!"}], options=options
    )

    assert response.response is not None
    assert response.log is not None

    from nemoguardrails.tracing.interaction_types import (
        InteractionOutput,
        extract_interaction_log,
    )

    interaction_output = InteractionOutput(
        id="test", input="Hello!", output=response.response
    )

    interaction_log = extract_interaction_log(interaction_output, response.log)

    assert len(interaction_log.trace) > 0

    for span in interaction_log.trace:
        assert is_opentelemetry_span(span)

    interaction_span = next(
        (s for s in interaction_log.trace if s.name == "guardrails.request"), None
    )
    assert interaction_span is not None

    llm_spans = [s for s in interaction_log.trace if isinstance(s, LLMSpan)]
    assert len(llm_spans) > 0

    for llm_span in llm_spans:
        assert hasattr(llm_span, "provider_name")
        assert hasattr(llm_span, "request_model")

        attrs = llm_span.to_otel_attributes()
        assert "gen_ai.provider.name" in attrs
        assert "gen_ai.request.model" in attrs

        assert hasattr(llm_span, "events")
        assert len(llm_span.events) > 0


def test_v1_backward_compatibility(v1_config):
    assert v1_config.tracing.span_format == "legacy"

    llm = FakeLLM(responses=["Hello!"])
    _rails = LLMRails(config=v1_config, llm=llm)

    extractor = create_span_extractor(span_format="legacy")
    assert extractor.__class__.__name__ == "SpanExtractorV1"


def test_default_span_format(default_config):
    assert default_config.tracing.span_format == "opentelemetry"


def test_span_format_configuration_direct():
    extractor_legacy = create_span_extractor(span_format="legacy")
    assert extractor_legacy.__class__.__name__ == "SpanExtractorV1"

    extractor_otel = create_span_extractor(span_format="opentelemetry")
    assert extractor_otel.__class__.__name__ == "SpanExtractorV2"

    with pytest.raises(ValueError) as exc_info:
        create_span_extractor(span_format="invalid")
    assert "Invalid span format" in str(exc_info.value)
