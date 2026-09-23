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

"""Text helpers for interpreting raw LLM completions.

Pure string transformations with no Colang, event, or configuration dependency, so
they can be used from a rail action that runs without a Colang runtime.

The completion parsers that do depend on Colang intent/action vocabulary stay in
``nemoguardrails.actions.llm.utils``.
"""

from typing import List


def strip_quotes(s: str) -> str:
    """Helper that removes quotes from a string if the entire string is between quotes"""
    if s and s[0] == '"':
        if s[-1] == '"':
            s = s[1:-1]
        else:
            s = s[1:]
    return s


def get_multiline_response(s: str) -> str:
    """Helper that extracts multi-line responses from the LLM.
    Stopping conditions: when a non-empty line ends with a quote or when the token "user" appears after a newline.
    Empty lines at the begging of the string are skipped."""

    # Check if the token "user" appears after a newline, as this would mark a new dialogue turn.
    # Remove everything after this marker.
    if "\nuser" in s:
        # Remove everything after the interrupt signal
        s = s.split("\nuser")[0]

    lines = [line.strip() for line in s.split("\n")]
    result = ""
    for line in lines:
        # Keep getting additional non-empty lines until the message ends
        if len(line) > 0:
            if len(result) == 0:
                result = line
            else:
                result += "\n" + line
            if line.endswith('"'):
                break

    return result


def remove_action_intent_identifiers(lines: List[str]) -> List[str]:
    """Removes the action/intent identifiers."""
    return [
        s.replace("bot intent: ", "")
        .replace("bot action: ", "")
        .replace("user intent: ", "")
        .replace("user action: ", "")
        for s in lines
    ]
