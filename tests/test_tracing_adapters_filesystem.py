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
import importlib
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from nemoguardrails.eval.models import Span
from nemoguardrails.tracing import InteractionLog
from nemoguardrails.tracing.adapters.filesystem import FileSystemAdapter


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
                Span(
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
                    Span(
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
