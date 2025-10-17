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

import asyncio
import importlib
import json
import os
import tempfile
import unittest

from nemoguardrails.tracing import InteractionLog, SpanLegacy
from nemoguardrails.tracing.adapters.filesystem import FileSystemAdapter
from nemoguardrails.tracing.spans import (
    ActionSpan,
    InteractionSpan,
    LLMSpan,
    RailSpan,
    SpanEvent,
)


class TestFileSystemAdapter(unittest.TestCase):
    def setUp(self):
        # creating a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = os.path.join(self.temp_dir.name, "trace.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization_default_path(self):
        adapter = FileSystemAdapter()
        self.assertEqual(adapter.filepath, "./.traces/trace.jsonl")

    def test_initialization_custom_path(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        self.assertEqual(adapter.filepath, self.filepath)
        self.assertTrue(os.path.exists(os.path.dirname(self.filepath)))

    def test_transform(self):
        adapter = FileSystemAdapter(filepath=self.filepath)

        #  Mock the InteractionLog
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={},
                )
            ],
        )

        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            self.assertEqual(log_dict["trace_id"], "test_id")
            self.assertEqual(len(log_dict["spans"]), 1)
            self.assertEqual(log_dict["spans"][0]["name"], "test_span")

    @unittest.skipIf(
        importlib.util.find_spec("aiofiles") is None, "aiofiles is not installed"
    )
    def test_transform_async(self):
        async def run_test():
            adapter = FileSystemAdapter(filepath=self.filepath)

            # Mock the InteractionLog
            interaction_log = InteractionLog(
                id="test_id",
                activated_rails=[],
                events=[],
                trace=[
                    SpanLegacy(
                        name="test_span",
                        span_id="span_1",
                        parent_id=None,
                        start_time=0.0,
                        end_time=1.0,
                        duration=1.0,
                        metrics={},
                    )
                ],
            )

            await adapter.transform_async(interaction_log)

            with open(self.filepath, "r") as f:
                content = f.read()
                log_dict = json.loads(content.strip())
                self.assertEqual(log_dict["trace_id"], "test_id")
                self.assertEqual(len(log_dict["spans"]), 1)
                self.assertEqual(log_dict["spans"][0]["name"], "test_span")

        asyncio.run(run_test())

    def test_schema_version(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_id",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="test_span",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    metrics={},
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            self.assertEqual(log_dict["schema_version"], "1.0")

    def test_span_legacy_with_metrics(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_trace",
            activated_rails=[],
            events=[],
            trace=[
                SpanLegacy(
                    name="llm_call",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.5,
                    duration=1.5,
                    metrics={
                        "input_tokens": 10,
                        "output_tokens": 20,
                        "total_tokens": 30,
                    },
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            span = log_dict["spans"][0]
            self.assertEqual(span["span_type"], "SpanLegacy")
            self.assertIn("metrics", span)
            self.assertEqual(span["metrics"]["input_tokens"], 10)
            self.assertEqual(span["metrics"]["output_tokens"], 20)
            self.assertEqual(span["metrics"]["total_tokens"], 30)

    def test_interaction_span_with_events(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        events = [
            SpanEvent(
                name="gen_ai.content.prompt",
                timestamp=0.1,
                attributes={"gen_ai.prompt": "Hello, how are you?"},
            ),
            SpanEvent(
                name="gen_ai.content.completion",
                timestamp=1.9,
                attributes={"gen_ai.completion": "I'm doing well, thank you!"},
            ),
        ]
        interaction_log = InteractionLog(
            id="test_trace",
            activated_rails=[],
            events=[],
            trace=[
                InteractionSpan(
                    name="interaction",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=2.0,
                    duration=2.0,
                    span_kind="server",
                    request_model="gpt-4",
                    events=events,
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            span = log_dict["spans"][0]
            self.assertEqual(span["span_type"], "InteractionSpan")
            self.assertEqual(span["span_kind"], "server")
            self.assertIn("events", span)
            self.assertEqual(len(span["events"]), 2)
            self.assertEqual(span["events"][0]["name"], "gen_ai.content.prompt")
            self.assertEqual(span["events"][0]["timestamp"], 0.1)
            self.assertIn("attributes", span)
            self.assertIn("gen_ai.operation.name", span["attributes"])

    def test_rail_span_with_attributes(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_trace",
            activated_rails=[],
            events=[],
            trace=[
                RailSpan(
                    name="check_jailbreak",
                    span_id="span_1",
                    parent_id="parent_span",
                    start_time=0.5,
                    end_time=1.0,
                    duration=0.5,
                    span_kind="internal",
                    rail_type="input",
                    rail_name="check_jailbreak",
                    rail_stop=False,
                    rail_decisions=["allow"],
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            span = log_dict["spans"][0]
            self.assertEqual(span["span_type"], "RailSpan")
            self.assertEqual(span["span_kind"], "internal")
            self.assertEqual(span["parent_id"], "parent_span")
            self.assertIn("attributes", span)
            self.assertEqual(span["attributes"]["rail.type"], "input")
            self.assertEqual(span["attributes"]["rail.name"], "check_jailbreak")
            self.assertEqual(span["attributes"]["rail.stop"], False)
            self.assertEqual(span["attributes"]["rail.decisions"], ["allow"])

    def test_action_span_with_error(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_trace",
            activated_rails=[],
            events=[],
            trace=[
                ActionSpan(
                    name="execute_action",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=0.5,
                    duration=0.5,
                    span_kind="internal",
                    action_name="fetch_data",
                    action_params={"url": "https://api.example.com"},
                    error=True,
                    error_type="ConnectionError",
                    error_message="Failed to connect to API",
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            span = log_dict["spans"][0]
            self.assertEqual(span["span_type"], "ActionSpan")
            self.assertIn("error", span)
            self.assertEqual(span["error"]["occurred"], True)
            self.assertEqual(span["error"]["type"], "ConnectionError")
            self.assertEqual(span["error"]["message"], "Failed to connect to API")
            self.assertIn("attributes", span)
            self.assertEqual(span["attributes"]["action.name"], "fetch_data")

    def test_llm_span_with_custom_attributes(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_trace",
            activated_rails=[],
            events=[],
            trace=[
                LLMSpan(
                    name="llm_api_call",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                    span_kind="client",
                    provider_name="openai",
                    operation_name="chat.completions",
                    request_model="gpt-4",
                    temperature=0.7,
                    response_model="gpt-4-0613",
                    usage_input_tokens=50,
                    usage_output_tokens=100,
                    custom_attributes={"custom_key": "custom_value"},
                )
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            span = log_dict["spans"][0]
            self.assertEqual(span["span_type"], "LLMSpan")
            self.assertEqual(span["span_kind"], "client")
            self.assertIn("attributes", span)
            self.assertEqual(span["attributes"]["gen_ai.request.model"], "gpt-4")
            self.assertEqual(span["attributes"]["gen_ai.request.temperature"], 0.7)
            self.assertEqual(span["attributes"]["gen_ai.response.model"], "gpt-4-0613")
            self.assertEqual(span["attributes"]["gen_ai.usage.input_tokens"], 50)
            self.assertEqual(span["attributes"]["gen_ai.usage.output_tokens"], 100)
            self.assertIn("custom_attributes", span)
            self.assertEqual(span["custom_attributes"]["custom_key"], "custom_value")

    def test_mixed_span_types(self):
        adapter = FileSystemAdapter(filepath=self.filepath)
        interaction_log = InteractionLog(
            id="test_mixed",
            activated_rails=[],
            events=[],
            trace=[
                InteractionSpan(
                    name="interaction",
                    span_id="span_1",
                    parent_id=None,
                    start_time=0.0,
                    end_time=3.0,
                    duration=3.0,
                    span_kind="server",
                    request_model="gpt-4",
                ),
                RailSpan(
                    name="check_jailbreak",
                    span_id="span_2",
                    parent_id="span_1",
                    start_time=0.5,
                    end_time=1.0,
                    duration=0.5,
                    span_kind="internal",
                    rail_type="input",
                    rail_name="check_jailbreak",
                    rail_stop=False,
                ),
                SpanLegacy(
                    name="legacy_span",
                    span_id="span_3",
                    parent_id="span_1",
                    start_time=1.5,
                    end_time=2.5,
                    duration=1.0,
                    metrics={"tokens": 25},
                ),
            ],
        )
        adapter.transform(interaction_log)

        with open(self.filepath, "r") as f:
            content = f.read()
            log_dict = json.loads(content.strip())
            self.assertEqual(len(log_dict["spans"]), 3)

            self.assertEqual(log_dict["spans"][0]["span_type"], "InteractionSpan")
            self.assertIn("span_kind", log_dict["spans"][0])
            self.assertIn("attributes", log_dict["spans"][0])

            self.assertEqual(log_dict["spans"][1]["span_type"], "RailSpan")
            self.assertEqual(log_dict["spans"][1]["parent_id"], "span_1")

            self.assertEqual(log_dict["spans"][2]["span_type"], "SpanLegacy")
            self.assertIn("metrics", log_dict["spans"][2])
            self.assertNotIn("span_kind", log_dict["spans"][2])

    @unittest.skipIf(
        importlib.util.find_spec("aiofiles") is None, "aiofiles is not installed"
    )
    def test_transform_async_with_otel_spans(self):
        async def run_test():
            adapter = FileSystemAdapter(filepath=self.filepath)
            interaction_log = InteractionLog(
                id="test_async_otel",
                activated_rails=[],
                events=[],
                trace=[
                    InteractionSpan(
                        name="interaction",
                        span_id="span_1",
                        parent_id=None,
                        start_time=0.0,
                        end_time=2.0,
                        duration=2.0,
                        span_kind="server",
                        request_model="gpt-4",
                        events=[
                            SpanEvent(
                                name="test_event",
                                timestamp=1.0,
                                attributes={"key": "value"},
                            )
                        ],
                    )
                ],
            )

            await adapter.transform_async(interaction_log)

            with open(self.filepath, "r") as f:
                content = f.read()
                log_dict = json.loads(content.strip())
                self.assertEqual(log_dict["schema_version"], "2.0")
                self.assertEqual(log_dict["trace_id"], "test_async_otel")
                span = log_dict["spans"][0]
                self.assertEqual(span["span_type"], "InteractionSpan")
                self.assertIn("events", span)
                self.assertEqual(len(span["events"]), 1)

        asyncio.run(run_test())
