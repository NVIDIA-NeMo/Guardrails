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
import pytest

from nemoguardrails.utils import safe_eval


@pytest.mark.parametrize(
    "input_value, expected_result",
    [
        ('"It\'s a sunny day"', "It's a sunny day"),  # double quotes with single quote
        (
            "\"He said, 'Hello'\"",
            "He said, 'Hello'",
        ),  # double quotes with nested single quote
        (
            "It's a sunny day",
            "It's a sunny day",
        ),  # unquoted string containing single quote
        (
            "It is a sunny day",
            "It is a sunny day",
        ),  # plain string not wrapped in quotes
        ("", ""),  # empty string
    ],
)
def test_safe_eval(input_value, expected_result):
    """Test safe_eval with various input values."""
    result = safe_eval(input_value)
    assert result == expected_result
