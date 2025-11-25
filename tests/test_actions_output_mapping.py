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


from nemoguardrails.actions import action
from nemoguardrails.actions.output_mapping import (
    default_output_mapping,
    is_output_blocked,
)

# Tests for default_output_mapping


def test_default_output_mapping_boolean_true():
    # For booleans, the mapping returns the negation (block if result is False).
    # If result is True, not True == False, so output is not blocked.
    assert default_output_mapping(True) is False


def test_default_output_mapping_boolean_false():
    # If result is False, then not False == True, so it is blocked.
    assert default_output_mapping(False) is True


def test_default_output_mapping_numeric_below_threshold():
    # For numeric values, block if the value is less than 0.5.
    assert default_output_mapping(0.4) is True


def test_default_output_mapping_numeric_above_threshold():
    # For numeric values greater than or equal to 0.5, do not block.
    assert default_output_mapping(0.5) is False
    assert default_output_mapping(0.6) is False


def test_default_output_mapping_non_numeric_non_boolean():
    # For other types (e.g., strings), default mapping returns False (allowed).
    assert default_output_mapping("anything") is False


# Tests for should_block_output


# Create a dummy action function with an attached mapping in its metadata.
def dummy_action_output_mapping(val):
    # For testing, block if the value equals "block", otherwise do not block.
    return val == "block"


@action(output_mapping=dummy_action_output_mapping)
def dummy_action(result):
    return result


def test_should_block_output_with_tuple_result_and_mapping():
    # Test should_block_output when the result is a tuple and the dummy mapping is used.
    # When the first element equals "block", we expect True.
    result = ("block",)
    assert is_output_blocked(result, dummy_action) is True

    # When the result is not "block", we expect False.
    result = ("allow",)
    assert is_output_blocked(result, dummy_action) is False


def test_should_block_output_with_non_tuple_result_and_mapping():
    # Test should_block_output when the result is not a tuple.
    # The function should wrap it into a tuple.
    result = "block"
    assert is_output_blocked(result, dummy_action) is True

    result = "allow"
    assert is_output_blocked(result, dummy_action) is False


def test_should_block_output_without_action_meta():
    # Test should_block_output when the action function does not have an "action_meta" attribute.
    # In this case, default_output_mapping should be used.
    def action_without_meta(res):
        return res

    # Ensure there is no action_meta attribute.
    if hasattr(action_without_meta, "action_meta"):
        del action_without_meta.action_meta

    # Test with a boolean: default_output_mapping for True is False and for False is True.
    assert is_output_blocked(True, action_without_meta) is False
    assert is_output_blocked(False, action_without_meta) is True

    # Test with a numeric value: block if < 0.5.
    assert is_output_blocked(0.4, action_without_meta) is True
    assert is_output_blocked(0.6, action_without_meta) is False
