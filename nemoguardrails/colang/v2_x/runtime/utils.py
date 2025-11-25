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

import re


class AttributeDict(dict):
    """Simple utility to allow accessing dict members as attributes."""

    def __getattr__(self, attr):
        val = self.get(attr, None)
        if isinstance(val, dict):
            return AttributeDict(val)
        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
            return [AttributeDict(x) for x in val]
        else:
            return val

    def __setattr__(self, attr, value):
        self[attr] = value


def escape_special_string_characters(string: str) -> str:
    """Escapes all occurrences of special characters."""
    # Replace " or ' with \\" or \\' if not already escaped
    string = re.sub(r"(^|[^\\])('|\")", r"\1\\\2", string)
    # Replace other special characters
    escaped_characters_map = {
        "\n": "\\n",
        "\t": "\\t",
        "\r": "\\r",
        "\b": "\\b",
        "\f": "\\f",
        "\v": "\\v",
    }

    for c, s in escaped_characters_map.items():
        string = str(string).replace(c, s)

    return string
