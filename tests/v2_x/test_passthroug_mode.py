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
import logging
import unittest

from nemoguardrails import RailsConfig
from tests.utils import TestChat

colang_content = '''
    import core
    import passthrough

    flow main
        activate llm continuation
        activate greeting
        activate other reactions

    flow greeting
        user expressed greeting
        bot say "Hello world!"

    flow other reactions
        user expressed to be bored
        bot say "No problem!"

    flow user expressed greeting
        """"User expressed greeting in any way or form."""
        user said "hi"

    flow user expressed to be bored
        """"User expressed to be bored."""
        user said "This is boring"
    '''

yaml_content = """
colang_version: "2.x"
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo-instruct

    """


config = RailsConfig.from_content(colang_content, yaml_content)


class TestPassthroughLLMActionLogging(unittest.IsolatedAsyncioTestCase):
    def test_passthrough_llm_action_not_invoked_via_logs(self):
        chat = TestChat(
            config,
            llm_completions=["user expressed greeting"],
        )
        rails = chat.app

        logger = logging.getLogger("nemoguardrails.colang.v2_x.runtime.statemachine")

        with self.assertLogs(logger, level="INFO") as log:
            messages = [{"role": "user", "content": "hi"}]
            response = rails.generate(messages=messages)
            # Check that 'StartPassthroughLLMAction' is not in the logs
            passthrough_invoked = any(
                "PassthroughLLMActionFinished" in message for message in log.output
            )
            self.assertFalse(
                passthrough_invoked, "PassthroughLLMAction was invoked unexpectedly."
            )

            self.assertIn("content", response)
            self.assertIsInstance(response["content"], str)

    @unittest.skip(
        reason="Github issue https://github.com/NVIDIA/NeMo-Guardrails/issues/1378"
    )
    def test_passthrough_llm_action_invoked_via_logs(self):
        chat = TestChat(
            config,
            llm_completions=["user asked about capabilites", "a random text from llm"],
        )
        rails = chat.app

        logger = logging.getLogger("nemoguardrails.colang.v2_x.runtime.statemachine")

        with self.assertLogs(logger, level="INFO") as log:
            messages = [{"role": "user", "content": "What can you do?"}]
            response = rails.generate(messages=messages)
            # Check that 'StartPassthroughLLMAction' is in the logs
            passthrough_invoked = any(
                "StartPassthroughLLMAction" in message for message in log.output
            )
            self.assertTrue(
                passthrough_invoked, "PassthroughLLMAction was not invoked."
            )

            self.assertIn("content", response)
            self.assertIsInstance(response["content"], str)
