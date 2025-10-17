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


class TestTopicalRail(ExampleConfigChatterTestCase):
    example_name = "topical_rail"
    config_name = "_deprecated/topical_rail"

    @pytest.mark.skipif(not QA_MODE, reason="Not in QA mode.")
    @pytest.mark.unit
    def test_topical_rail(self):
        """Test the topical_rail example"""
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
                "Hello there! How can I help you?",
                "Hello, how can I help you today?",
                "Hi there! How can I help you today?",
                "Hello again! What can I help you with regarding the job report?",
                "Hello! What can I help you with?",
                "Hello, how can I help you today?",
            ],
            "How can you help?": [
                "I am an AI assistant that helps answer mathematical questions. My core mathematical skills are powered by wolfram alpha.",
                "I am an AI assistant which helps answer questions based on a given knowledge base. For this interaction, I can answer question based on the job report published by US Bureau of Labor Statistics.",
                "How are you feeling today?",
            ],
            "How many jobs were added in the healthcare sector?": [
                "According to the US Bureau of Labor Statistics, health care added 34,000 jobs in March, which is lower than the average monthly gain of 54,000 over the prior 6 months.",
                "According to the latest report from the US Bureau of Labor Statistics, the healthcare sector added 57,000 jobs in the month of October.",
                "According to the US Bureau of Labor Statistics' establishment survey, there were 33,000 jobs added in the healthcare sector in the last month.",
                "According to the latest jobs report, the US economy added 439,000 jobs in the healthcare sector in July 2020.",
                "According to the most recent report, the healthcare sector added 47,000 jobs in April 2021.",
                "According to the latest report, the healthcare sector added 267,000 jobs in April 2021",
                "According to the latest report, the healthcare sector added 59,000 jobs in January 2021.",
                "According to the establishment survey data, the healthcare sector added 53,000 jobs in April 2021.",
                "According to the March 2023 Employment Situation Report, health care added 34,000 jobs.",
                "According to the US Bureau of Labor Statistics, the healthcare sector added 34,000 jobs in March.",
            ],
            "What's the best flavor of icecream?": [
                "Sorry, I cannot comment on anything which is not relevant to the jobs report"
            ],
        }

        self.run_test(messages)
