# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy

import pytest

from nemoguardrails.colang.v1_0.runtime.utils import AttributeDict


@pytest.mark.parametrize("copy_value", [copy.copy, copy.deepcopy])
def test_attribute_dict_tool_calls_support_copy_protocol(copy_value):
    """Allow callers and rail actions to copy tool payloads on every supported Python version."""
    call = AttributeDict({"id": "call_weather", "function": {"name": "weather", "arguments": {"cities": ["Paris"]}}})
    copied = copy_value(call)
    assert isinstance(copied, AttributeDict)
    assert copied == call
    assert copied is not call
    if copy_value is copy.deepcopy:
        copied["function"]["arguments"]["cities"].append("London")
        assert call["function"]["arguments"]["cities"] == ["Paris"]
    else:
        assert copied["function"] is call["function"]


def test_attribute_dict_deepcopy_preserves_shared_and_recursive_values():
    """Copy nested runtime values without losing shared references or recursing indefinitely."""
    call = AttributeDict({"id": "call_weather"})
    history = AttributeDict({"first": call, "second": call})
    history["self"] = history
    copied = copy.deepcopy(history)
    assert copied is not history
    assert copied["self"] is copied
    assert copied["first"] is copied["second"]
    assert copied["first"] is not call


@pytest.mark.parametrize("attribute", ["__deepcopy__", "__setstate__", "__missing_protocol__"])
def test_attribute_dict_missing_protocol_attributes_raise(attribute):
    """Report missing Python protocol hooks as absent instead of fabricating a None-valued hook."""
    value = AttributeDict()
    with pytest.raises(AttributeError):
        getattr(value, attribute)


def test_attribute_dict_keeps_colang_attribute_semantics():
    """Retain missing-key defaults, nested attribute access, and assignment used by Colang expressions."""
    value = AttributeDict({"nested": {"answer": 42}, "results": [{"answer": 42}], "_private": "retained"})
    assert value.missing is None
    assert value._private == "retained"
    assert value.nested.answer == 42
    assert value.results[0].answer == 42
    value.new_field = "assigned"
    assert value["new_field"] == "assigned"
