#!/usr/bin/env python3
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

"""
Complete working example of NeMo Guardrails with OpenTelemetry tracing.

This example uses the ConsoleSpanExporter so you can see traces immediately
without needing to set up any external infrastructure.

Usage:
    pip install nemoguardrails[tracing] opentelemetry-sdk
    python working_example.py
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from nemoguardrails import LLMRails, RailsConfig


def setup_opentelemetry():
    """Configure OpenTelemetry SDK with console output."""

    print("Setting up OpenTelemetry...")

    # configure resource (metadata about your service)
    resource = Resource.create(
        {
            "service.name": "nemo-guardrails-example",
            "service.version": "1.0.0",
            "deployment.environment": "development",
        },
        schema_url="https://opentelemetry.io/schemas/1.26.0",
    )

    # set up the tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # configure console exporter (prints traces to stdout)
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    tracer_provider.add_span_processor(span_processor)

    print(" OpenTelemetry configured with ConsoleSpanExporter")
    print(" Traces will be printed to the console below\n")


def create_guardrails_config():
    """Create a simple guardrails configuration with tracing enabled."""

    return RailsConfig.from_content(
        colang_content="""
        define user express greeting
            "hello"
            "hi"
            "hey"

        define flow
            user express greeting
            bot express greeting

        define bot express greeting
            "Hello! I'm a guardrails-enabled assistant."
            "Hi there! How can I help you today?"
        """,
        config={
            "models": [
                {
                    "type": "main",
                    "engine": "openai",
                    "model": "gpt-4o",
                }
            ],
            "tracing": {"enabled": True, "adapters": [{"name": "OpenTelemetry"}]},
            # Note: The following old-style configuration is deprecated and will be ignored:
            # "tracing": {
            #     "enabled": True,
            #     "adapters": [{
            #         "name": "OpenTelemetry",
            #         "service_name": "my-service",      # DEPRECATED - configure in Resource
            #         "exporter": "console",             # DEPRECATED - configure SDK
            #         "resource_attributes": {           # DEPRECATED - configure in Resource
            #             "env": "production"
            #         }
            #     }]
            # }
        },
    )


def main():
    """Main function demonstrating NeMo Guardrails with OpenTelemetry."""
    print(" NeMo Guardrails + OpenTelemetry Example")
    print("=" * 50)

    # step 1: configure OpenTelemetry (APPLICATION'S RESPONSIBILITY)
    setup_opentelemetry()

    # step 2: create guardrails configuration
    print(" Creating guardrails configuration...")
    config = create_guardrails_config()
    rails = LLMRails(config)
    print(" Guardrails configured with tracing enabled\n")

    # step 3: test the guardrails with tracing
    print(" Testing guardrails (traces will appear below)...")
    print("-" * 50)

    # this will create spans that get exported to the console
    response = rails.generate(
        messages=[{"role": "user", "content": "What can you do?"}]
    )

    print(f"User: What can you do?")
    print(f"Bot: {response.response}")
    print("-" * 50)

    # force export any remaining spans
    print("\n Flushing remaining traces...")
    trace.get_tracer_provider().force_flush(1000)

    print("\n Example completed!")
    print("\n Tips:")
    print("   - Traces were printed above (look for JSON output)")
    print("   - In production, replace ConsoleSpanExporter with OTLP/Jaeger")
    print("   - The spans show the internal flow of guardrails processing")


if __name__ == "__main__":
    main()
