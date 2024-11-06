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
import unittest
from unittest.mock import AsyncMock, MagicMock

from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.rails.llm.config import TracingConfig
from nemoguardrails.rails.llm.options import (
    ActivatedRail,
    ExecutedAction,
    GenerationLog,
    GenerationResponse,
)
from nemoguardrails.tracing.adapters.base import InteractionLogAdapter
from nemoguardrails.tracing.tracer import Tracer, new_uuid


class TestTracer(unittest.TestCase):
    def test_new_uuid(self):
        uuid_str = new_uuid()
        self.assertIsInstance(uuid_str, str)
        self.assertEqual(len(uuid_str), 36)  # UUID length

    def test_tracer_initialization(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=GenerationLog())
        tracer = Tracer(input=input_data, response=response)
        self.assertEqual(tracer._interaction_output.input, "test input")
        self.assertEqual(tracer._interaction_output.output, "test response")
        self.assertEqual(tracer._generation_log, response.log)

    def test_tracer_initialization_missing_log(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=None)
        with self.assertRaises(RuntimeError):
            Tracer(input=input_data, response=response)

    def test_generate_interaction_log(self):
        input_data = [{"content": "test input"}]

        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=[],
                executed_actions=[],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer = Tracer(input=input_data, response=response)
        interaction_log = tracer.generate_interaction_log()
        self.assertIsNotNone(interaction_log)

    def test_add_adapter(self):
        input_data = [{"content": "test input"}]
        response = GenerationResponse(response="test response", log=GenerationLog())
        tracer = Tracer(input=input_data, response=response)
        adapter = MagicMock(spec=InteractionLogAdapter)
        tracer.add_adapter(adapter)
        self.assertIn(adapter, tracer.adapters)

    def test_export(self):
        input_data = [{"content": "test input"}]

        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=["dummy_decision"],
                executed_actions=[
                    ExecutedAction(
                        action_name="dummy_action",
                        action_params={},
                        return_value=None,
                        llm_calls=[
                            LLMCallInfo(
                                task="dummy_task",
                                duration=1.0,
                                total_tokens=10,
                                prompt_tokens=5,
                                completion_tokens=5,
                                started_at=0.0,
                                finished_at=1.0,
                                prompt="dummy_prompt",
                                completion="dummy_completion",
                                raw_response={
                                    "token_usage": {
                                        "total_tokens": 10,
                                        "completion_tokens": 5,
                                        "prompt_tokens": 5,
                                    },
                                    "model_name": "dummy_model",
                                },
                                llm_model_name="dummy_model",
                            )
                        ],
                        started_at=0.0,
                        finished_at=1.0,
                        duration=1.0,
                    )
                ],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response_non_empty = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer_non_empty = Tracer(input=input_data, response=response_non_empty)
        adapter_non_empty = MagicMock(spec=InteractionLogAdapter)
        tracer_non_empty.add_adapter(adapter_non_empty)
        tracer_non_empty.export()
        adapter_non_empty.transform.assert_called_once()

    def test_export_async(self):
        input_data = [{"content": "test input"}]
        activated_rails = [
            ActivatedRail(
                type="dummy_type",
                name="dummy_name",
                decisions=["dummy_decision"],
                executed_actions=[
                    ExecutedAction(
                        action_name="dummy_action",
                        action_params={},
                        return_value=None,
                        llm_calls=[
                            LLMCallInfo(
                                task="dummy_task",
                                duration=1.0,
                                total_tokens=10,
                                prompt_tokens=5,
                                completion_tokens=5,
                                started_at=0.0,
                                finished_at=1.0,
                                prompt="dummy_prompt",
                                completion="dummy_completion",
                                raw_response={
                                    "token_usage": {
                                        "total_tokens": 10,
                                        "completion_tokens": 5,
                                        "prompt_tokens": 5,
                                    },
                                    "model_name": "dummy_model",
                                },
                                llm_model_name="dummy_model",
                            )
                        ],
                        started_at=0.0,
                        finished_at=1.0,
                        duration=1.0,
                    )
                ],
                stop=False,
                additional_info=None,
                started_at=0.0,
                finished_at=1.0,
                duration=1.0,
            )
        ]

        response_non_empty = GenerationResponse(
            response="test response",
            log=GenerationLog(activated_rails=activated_rails, internal_events=[]),
        )
        tracer_non_empty = Tracer(input=input_data, response=response_non_empty)
        adapter_non_empty = AsyncMock(spec=InteractionLogAdapter)
        adapter_non_empty.__aenter__ = AsyncMock(return_value=adapter_non_empty)
        adapter_non_empty.__aexit__ = AsyncMock(return_value=None)
        tracer_non_empty.add_adapter(adapter_non_empty)

        asyncio.run(tracer_non_empty.export_async())
        adapter_non_empty.transform_async.assert_called_once()


if __name__ == "__main__":
    unittest.main()
