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

import pytest

from nemoguardrails import RailsConfig
from nemoguardrails.llm.prompts import TaskPrompt


@pytest.fixture(
    params=[
        [
            TaskPrompt(task="self_check_input", output_parser=None, content="..."),
            TaskPrompt(task="self_check_facts", output_parser="parser1", content="..."),
            TaskPrompt(task="self_check_output", output_parser="parser2", content="..."),
        ],
        [
            {"task": "self_check_input", "output_parser": None},
            {"task": "self_check_facts", "output_parser": "parser1"},
            {"task": "self_check_output", "output_parser": "parser2"},
        ],
    ]
)
def prompts(request):
    return request.param


def test_check_output_parser_exists(caplog, prompts):
    caplog.set_level(logging.INFO)
    values = {"prompts": prompts}

    result = RailsConfig.check_output_parser_exists(values)

    assert result == values
    assert "Deprecation Warning: Output parser is not registered for the task." in caplog.text
    assert "self_check_input" in caplog.text
