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


class TestGroundingRail(ExampleConfigChatterTestCase):
    example_name = "grounding_rail"
    config_name = "_deprecated/grounding_rail"

    @pytest.mark.skipif(not QA_MODE, reason="Not in QA mode.")
    @pytest.mark.unit
    def test_grounding_rail(self):
        """Test the grounding_rail example"""
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
            "How many jobs were added in the transportation industry?": [
                "In March, employment in transportation and warehousing changed little (+10,000). Couriers and messengers (+7,000) and air transportation (+6,000) added jobs, while warehousing and storage lost jobs (-12,000). Employment in transportation and warehousing has shown little net change in recent months.",
                " In March, employment in transportation and warehousing changed little (+10,000). Couriers and messengers (+7,000) and air transportation (+6,000) added jobs, while warehousing and storage lost jobs (-12,000).",
                "According to the March 2023 US jobs report, employment in transportation and warehousing changed little (+10,000). Couriers and messengers (+7,000) and air transportation (+6,000) added jobs, while warehousing and storage lost jobs (-12,000).",
                "According to the report, employment in transportation and warehousing changed little (+10,000) in March.",
            ],
            "What was the unemployment rate for senior citizens?": [
                "The report states that the unemployment rate for persons age 65 and older was 5.6 percent in March.",
                "In March, the unemployment rate for seniors (age 65 and over) was 6.2%, up from 5.7% in February. The number of unemployed seniors increased by 136,000 to 1.2 million. The labor force participation rate for seniors declined by 0.4 percentage point to 20.2%.",
                "The unemployment rate for people aged 55 and over was 5.8 percent in March, little changed from a year earlier. The rate for those aged 25 to 54 was 6.1 percent, and the rate for those aged 20 to 24 was 7.1 percent.",
                "In March, the unemployment rate for people age 55 and older was 6.4%, compared with 8.7% for the overall population.",
                "I'm sorry, I do not know the answer to that question.",
                "I'm sorry, I don't know the answer to that question.",
                "I'm sorry, I don't have the information you're looking for. However, I can help you find the answer if you provide me more information about the context of your question.",
                "I'm sorry, I don't know the answer to that question. Would you like me to look for more information?",
                "I'm sorry, I don't seem to have the answer to that question. Would you like me to provide you with additional information on the March 2023 US jobs report or suggest some resources where you can find the answer?",
                "I'm not sure of the exact answer to that question, but according to the March 2023 jobs report, the unemployment rate for persons age 65 and over was 5.5 percent, down from 5.8 percent in February.",
                "I'm sorry, I don't know the answer to that question. However, I can provide you with the latest US jobs report from March 2023 which may contain the answer you are looking for.",
                "I'm sorry, I don't have the information you are looking for. Would you like me to search for other sources on the topic?",
                "The unemployment rate for individuals age 65 and older was 5.1 percent in March, little changed from 5.2 percent in February. The labor force participation rate for this group was 37.8 percent in March, up from 37.3 percent in February.",
            ],
            "How many CUDA cores does a 4090 have?": [
                "I'm not sure of the exact number of CUDA cores in a 4090, but according to NVIDIA, the 4090 has up to 5,888 CUDA cores, 24GB of GDDR6 memory, and a 3840-bit memory interface. The previous answer is prone to hallucination and may not be accurate. Please double check the answer using additional sources.",
                "I'm sorry, I do not know the answer to that question. However, I can provide you with a link to the NVIDIA website where you can find more information about the GeForce RTX 4090 GPU: https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-4090/",
                "I'm sorry, I don't have the information you're looking for. You may need to consult an expert or search online for the answer.",
            ],
        }

        self.run_test(messages)
