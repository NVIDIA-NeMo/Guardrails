# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.colang.v1_0.lang.utils import get_numbered_lines, split_args


def test_1():
    assert split_args("1") == ["1"]
    assert split_args('1, "a"') == ["1", '"a"']
    assert split_args("1, [1,2,3]") == ["1", "[1,2,3]"]
    assert split_args("1, numbers = [1,2,3]") == ["1", "numbers = [1,2,3]"]
    assert split_args("1, data = {'name': 'John'}") == ["1", "data = {'name': 'John'}"]
    assert split_args("'a,b, c'") == ["'a,b, c'"]

    assert split_args("1, 'a,b, c', x=[1,2,3], data = {'name': 'John'}") == [
        "1",
        "'a,b, c'",
        "x=[1,2,3]",
        "data = {'name': 'John'}",
    ]


def test_get_numbered_lines_or_continuation_at_eof():
    # A line ending in " or" as the final line must not raise IndexError.
    # The bounds guard previously only protected the "\\" continuation, so a
    # dangling " or" at end-of-file indexed past the last line.
    lines = get_numbered_lines("define flow test\n  user express greeting or")
    assert lines[-1]["text"] == "user express greeting or"


def test_get_numbered_lines_or_continuation_merges():
    # A valid " or" continuation still merges onto the next line.
    lines = get_numbered_lines("define flow test\n  a or\n  b")
    assert [line["text"] for line in lines] == ["define flow test", "a or b"]


@pytest.mark.parametrize("args_str", [")", "a)b", "foo]", "x=1, y)", "}"])
def test_split_args_unbalanced_closing_bracket_raises_value_error(args_str):
    # A stray closing bracket with no matching opener previously indexed an empty
    # `stack` (`closing_char[stack[-1]]`), raising an uncaught IndexError instead of
    # the ValueError the function uses to report invalid syntax.
    with pytest.raises(ValueError):
        split_args(args_str)


@pytest.mark.parametrize(
    "args_str, expected",
    [
        # A closing bracket inside a quoted string is literal text, not structural.
        ('x="]"', ['x="]"']),
        ('x=")"', ['x=")"']),
        ('x="}"', ['x="}"']),
        # Real-world strings: an emoticon or a citation marker.
        ('msg="Hello :)"', ['msg="Hello :)"']),
        ('note="cite [1] and (2)"', ['note="cite [1] and (2)"']),
        ('reason="closed ]"', ['reason="closed ]"']),
        # A comma inside a string does not split the arguments.
        ('x="a, b"', ['x="a, b"']),
        # The other quote character inside a string is literal too.
        ('x="it\'s ok)"', ['x="it\'s ok)"']),
        # A bracket-in-string argument alongside a normal argument still splits.
        ('n=1, x="]"', ["n=1", 'x="]"']),
    ],
)
def test_split_args_bracket_inside_string(args_str, expected):
    # Brackets, commas, and the other quote character inside a quoted string
    # argument are literal text and must not be treated as structural syntax.
    assert split_args(args_str) == expected
