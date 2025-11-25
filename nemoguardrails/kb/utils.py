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

from typing import List

import yaml


def split_markdown_in_topic_chunks(content: str, max_chunk_size: int = 400) -> List[dict]:
    """
    Splits a markdown content into topic chunks.

    This function takes a markdown content as input and divides it into topic chunks based on
    headings and subsections. Each chunk includes a title and body, with an optional maximum size.

    Parameters:
    - content (str): The markdown content to be split.
    - max_chunk_size (int): The maximum size of a chunk. Default is 400.

    Returns:
    List[dict]: A list of dictionaries, each representing a topic chunk with 'title' and 'body' keys.

    Example:
    ```python
    content = "# Introduction\n\nThis is an introduction.\n## Section 1\n\nContent of section 1."
    chunks = split_markdown_in_topic_chunks(content, max_chunk_size=500)
    ```

    Note:
    - The function considers '#' as heading markers.
    - Meta information can be included at the beginning of the markdown using triple backticks.
    """
    chunks = []
    lines = content.strip().split("\n")

    # Meta information for the whole document
    meta = {}

    # If there's a code block at the beginning, with meta data, we parse that first.
    if lines[0].startswith("```"):
        meta_yaml = ""
        lines = lines[1:]
        while not lines[0].startswith("```"):
            meta_yaml += lines[0] + "\n"
            lines = lines[1:]
        lines = lines[1:]

        meta.update(yaml.safe_load(meta_yaml))

    # Every section and subsection title will be part of the title of the chunk.
    chunk_title_parts = []

    # The data for the current chunk.
    chunk_body_lines = []
    chunk_size = 0

    def _record_chunk():
        nonlocal chunk_body_lines, chunk_size

        body = "\n".join(chunk_body_lines).strip()

        # Skip saving if body is empty
        if body:
            chunks.append(
                {
                    "title": " - ".join(chunk_title_parts),
                    "body": body,
                    # We also include the document level meta information
                    **meta,
                }
            )

        chunk_body_lines = []
        chunk_size = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("#"):
            # If we have a chunk up to this point, we need to record it
            if chunk_body_lines:
                _record_chunk()

            # Update the title parts with the new section/subsection
            level = 0
            while len(line) > 0 and line[0] == "#":
                level += 1
                line = line[1:]

            # Remove all title parts greater than the current level
            chunk_title_parts[level - 1 :] = []
            chunk_title_parts.append(line.strip())

        elif line.strip() == "":
            chunk_body_lines.append("")

            # If the chunk is over the desired size, we reset it
            if chunk_size > max_chunk_size:
                _record_chunk()
        else:
            chunk_body_lines.append(line)
            chunk_size += len(line)

        i += 1

    if chunk_body_lines:
        _record_chunk()

    return chunks
