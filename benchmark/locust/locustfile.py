#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Locust load test file for NeMo Guardrails OpenAI-compatible server.

This file defines the load test behavior. It can be run directly with:
    locust -f locustfile.py --host http://localhost:8000

Or via the Typer CLI wrapper:
    python -m benchmark.locust config.yaml
"""

import os

from locust import HttpUser, constant, task


class GuardrailsUser(HttpUser):
    """
    Simulated user that continuously sends chat completion requests to the
    NeMo Guardrails server.

    Each user will continuously send requests with no wait time between them
    (continuous load).
    """

    # No wait time between requests (continuous hammering)
    wait_time = constant(0)

    def on_start(self):
        """Called when a simulated user starts.
        Uses environment variables to pass the Guardrails config_id, model, and message"""
        # Get configuration from environment variables set by the CLI wrapper
        self.config_id = os.getenv("LOCUST_CONFIG_ID", "default")
        self.model = os.getenv("LOCUST_MODEL", "mock-llm")
        self.message = os.getenv("LOCUST_MESSAGE", "Hello, what can you do?")

    @task
    def chat_completion(self):
        """
        Send a Guardrails chat completion request (/v1/chat/completions)
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.message}],
            "guardrails": {"config_id": self.config_id},
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=60,
            catch_response=True,
            name="/v1/chat/completions",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got status code {response.status_code}: {response.text}")
