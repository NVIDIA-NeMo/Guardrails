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

import pytest

from nemoguardrails.actions.validation import validate_input, validate_response


@validate_input("name", validators=["length"], max_len=100)
@validate_response(validators=["ip_filter", "is_default_resp"])
def say_name(name: str = ""):
    """return back the name"""
    return name


@validate_input("name", validators=["length"], max_len=100)
@validate_response(validators=["ip_filter", "is_default_resp"])
class SayQuery:
    """function run should have validate decorator"""

    def run(self, name: str = ""):
        """return back the name"""
        return name


def test_func_validation():
    """Test validation on input and resp from functions"""

    # length is smaller than max len validation
    assert say_name(name="Alice") == "Alice"

    # Raise ValueError when input is longer than max len
    with pytest.raises(ValueError, match="Attribute name is too long."):
        say_name(name="Hello Alice" * 10)

    # Response validation: Response should not contain default response
    with pytest.raises(ValueError, match="Default Response received from action"):
        say_name(name="No good Wikipedia Search Result was found")

    # length is smaller than max len validation
    assert say_name(name="IP 10.40.139.92 should be trimmed") == "IP  should be trimmed"


def test_cls_validation():
    """Test validation on input and resp from functions"""

    s_name = SayQuery()

    # length is smaller than max len validation
    assert s_name.run(name="Alice") == "Alice"

    # Raise ValueError when input is longer than max len
    with pytest.raises(ValueError, match="Attribute name is too long."):
        s_name.run(name="Hello Alice" * 10)

    # Response validation: Response should not contain default response
    with pytest.raises(ValueError, match="Default Response received from action"):
        s_name.run(name="No good Wikipedia Search Result was found")

    # length is smaller than max len validation
    assert (
        s_name.run(name="IP 10.40.139.92 should be trimmed") == "IP  should be trimmed"
    )
