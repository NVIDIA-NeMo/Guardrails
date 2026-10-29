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

import pytest

from nemoguardrails.llm.completion_parsing import get_multiline_response, strip_quotes


@pytest.mark.parametrize(
    "text,expected",
    [
        ('"Hello!"', "Hello!"),
        ("“Hello!”", "Hello!"),
        ('"She said “hello”', "She said “hello”"),
        ("“Hello!", "Hello!"),
        ('"Hello!', "Hello!"),
        ("“", ""),
        ('"', ""),
        ("“”", ""),
        ("", ""),
        ("Hello!”", "Hello!”"),
        ("'Hello!'", "'Hello!'"),
        ("“Say “hello” to me.”", "Say “hello” to me."),
        ("She said “hello”.", "She said “hello”."),
    ],
)
def test_strip_quotes(text, expected):
    assert strip_quotes(text) == expected


@pytest.mark.parametrize("opening,closing", [('"', '"'), ("“", "”")])
def test_multiline_response_stops_at_closing_quote(opening, closing):
    text = f'\n  {opening}Hello!\nHow can I help?{closing}\nbot another intent\n  "Unwanted reply"'
    assert strip_quotes(get_multiline_response(text)) == "Hello!\nHow can I help?"


@pytest.mark.parametrize(
    "message",
    [
        "She said “hello”\nand then waved.",
        "The phrase is “helpful bot”\nbot software can assist.",
    ],
)
def test_straight_outer_quotes_preserve_multiline_inner_curly_quotes(message):
    assert strip_quotes(get_multiline_response(f'"{message}"')) == message
