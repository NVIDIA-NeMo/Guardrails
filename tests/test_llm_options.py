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

"""Tests for LLM isolation with models that don't have model_kwargs field."""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationLog, GenerationStats


def test_generation_log_print_summary(capsys):
    """Test printing rais stats with dummy data"""

    stats = GenerationStats(
        input_rails_duration=1.0,
        dialog_rails_duration=2.0,
        generation_rails_duration=3.0,
        output_rails_duration=4.0,
        total_duration=10.0,  # Sum of all previous rail durations
        llm_calls_duration=8.0,  # Less than total duration
        llm_calls_count=4,  # Input, dialog, generation and output calls
        llm_calls_total_prompt_tokens=1000,
        llm_calls_total_completion_tokens=2000,
        llm_calls_total_tokens=3000,  # Sum of prompt and completion tokens
    )

    generation_log = GenerationLog(activated_rails=[], stats=stats)

    generation_log.print_summary()
    capture = capsys.readouterr()
    capture_lines = capture.out.splitlines()

    # Check the correct times were printed
    assert capture_lines[1] == "# General stats"
    assert capture_lines[3] == "- Total time: 10.00s"
    assert capture_lines[4] == "  - [1.00s][10.0%]: INPUT Rails"
    assert capture_lines[5] == "  - [2.00s][20.0%]: DIALOG Rails"
    assert capture_lines[6] == "  - [3.00s][30.0%]: GENERATION Rails"
    assert capture_lines[7] == "  - [4.00s][40.0%]: OUTPUT Rails"
    assert (
        capture_lines[8]
        == "- 4 LLM calls, 8.00s total duration, 1000 total prompt tokens, 2000 total completion tokens, 3000 total tokens."
    )
