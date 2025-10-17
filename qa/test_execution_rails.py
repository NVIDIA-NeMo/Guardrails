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
import os

import pytest

from .utils import ExampleConfigChatterTestCase

QA_MODE = os.environ.get("QA")


class TestExecutionRails(ExampleConfigChatterTestCase):
    example_name = "execution_rails"
    config_name = "_deprecated/execution_rails"

    @pytest.mark.skipif(not QA_MODE, reason="Not in QA mode.")
    @pytest.mark.unit
    def test_execution_rails(self):
        """Test the execution_rails example"""
        # Define Q&A messages
        messages = {
            "Hi there!": [
                "Hi! How can I assist you today?",
                "Hello again! How can I help you?",
                "Hello again! What can I help you with today?",
                "Hello again! What can I do for you?",
                "Hello! What can I help you with today?",
                "Hello again! How can I help you today?",
                "Hello again! How can I be of assistance?",
                "Hello there! How can I help you today?",
                "Hello! How can I assist you today?",
                "Hey there!",
                "Hi! How can I help you today?",
                "Hello! How can I help you today?",
                "Hello, how can I help you today?",
                "Hello there! How can I help you?",
            ],
            "How can you help?": [
                "I am an AI assistant that helps answer mathematical questions. My core mathematical skills are powered by wolfram alpha.",
                "How are you feeling today?",
            ],
            "What is 434 + 56*7.5?": [
                "434 + 56*7.5 is equal to 854.",
                "The result is 854.",
                "The result of 434 + 56*7.5 is 854.",
                "The answer is 854.",
                "434 + 56 * 7.5 is equal to 854.",
            ],
        }

        self.run_test(messages)
