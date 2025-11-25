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
import logging
import re

from .v1_0.lang import parser as parser_v1_0
from .v2_x.lang import parser as parser_v2_x

log = logging.getLogger(__name__)


def parse_colang_file(
    filename: str,
    content: str,
    include_source_mapping: bool = True,
    version: str = "1.0",
):
    """Parse the content of a .co file into the CoYML format."""

    parsers = {
        "1.0": parser_v1_0.parse_colang_file,
        "2.x": parser_v2_x.parse_colang_file,
    }

    if version not in parsers:
        raise ValueError(f"Unsupported colang version {version}")

    if version == "2.x" and not _is_colang_v2(content):
        log.debug(f"Skipping parsing of {filename} because it is not a v2.x file.")
        return {}

    if version == "1.0" and _is_colang_v2(content):
        log.debug(f"Skipping parsing of {filename} because it is not a v1.0 file.")
        return {}

    return parsers[version](filename, content, include_source_mapping)


def parse_flow_elements(items, version: str = "1.0"):
    """Parse the flow elements from CoYML format to CIL."""

    parsers = {
        "1.0": parser_v1_0.parse_flow_elements,
        "2.x": None,
    }

    if version not in parsers:
        raise ValueError(f"Unsupported colang version {version}")

    if parsers[version] is None:
        raise NotImplementedError(f"Parsing flow elements not supported for colang version {version}")

    return parsers[version](items)


def _is_colang_v2(content):
    """Checks if the content of a file is in Colang 2.x format.

    This function uses a simple heuristic to determine if the content is a Colang 2.x file.
    Initially, it removes comments and content within triple quotes, as these could potentially
    contain misleading keywords. Then, it checks for the presence of certain keywords in the content:

    - If the keyword `import` is present, the content is likely a Colang 2.x file.
    - If the keyword `define` is present at the beginning of a line, the content is likely a Colang 1.0 file.

    Args:
        content (str): The content of the file to check.

    Returns:
        bool: True if the content is likely a Colang 2.x file, False otherwise.
    """

    # Remove content within triple quotes
    content = re.sub(r'""".*?"""', "", content, flags=re.DOTALL)
    # Remove content after #
    content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
    # Check for v1 keyword at the beginning of a line
    lines = content.split("\n")
    if any(re.match(r"^\s*define", line) for line in lines):
        return False
    # Check for v2 keyword
    if "import" in content:
        return True
    # If none of the above conditions are met, return True
    return True
