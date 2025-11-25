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
import textwrap
from typing import List, Optional

from nemoguardrails.colang.v1_0.lang.colang_parser import (
    parse_coflows_to_yml_flows,
    parse_snippets_and_imports,
)
from nemoguardrails.colang.v1_0.lang.comd_parser import parse_md_file
from nemoguardrails.colang.v1_0.lang.coyml_parser import parse_flow_elements

log = logging.getLogger(__name__)


def _extract_flow_code(file_content: str, flow_elements: List[dict]) -> Optional[str]:
    """Helper to extract the source code for a flow.

    Currently, it uses a simple heuristic that copies all the lines between the minimum
    and the maximum lines
    """

    content_lines = file_content.split("\n")
    min_line = -1
    max_line = -1

    for element in flow_elements:
        if "_source_mapping" not in element:
            continue
        line_number = element["_source_mapping"]["line_number"] - 1
        if min_line == -1 or line_number < min_line:
            min_line = line_number
        if max_line == -1 or line_number > max_line:
            max_line = line_number

    # If we have a range, we extract it
    if min_line >= 0:
        # Exclude all non-blank lines
        flow_lines = [_line for _line in content_lines[min_line : max_line + 1] if _line.strip() != ""]

        return textwrap.dedent("\n".join(flow_lines))

    return None


def parse_colang_file(filename: str, content: str, include_source_mapping: bool = True):
    """Parse the content of a .co file into the CoYML format."""
    snippets, imports = parse_snippets_and_imports(filename, content)
    result = parse_coflows_to_yml_flows(
        filename,
        content,
        snippets=snippets,
        include_source_mapping=include_source_mapping,
    )

    flows = []
    for flow_id, items in result["flows"].items():
        elements = parse_flow_elements(items)
        source_code = _extract_flow_code(content, elements)
        flows.append({"id": flow_id, "elements": elements, "source_code": source_code})

    user_messages = {}
    bot_messages = {}

    if result.get("markdown"):
        log.debug(f"Found markdown content in {filename}")
        md_result = parse_md_file(filename, content=result["markdown"])

        # Record the user messages
        # The `patterns` result from Markdown parsing contains patterns of the form
        # {'lang': 'en', 'type': 'PATTERN', 'sym': 'intent:express|greeting', 'body': 'hi', 'params': {}}
        # We need to convert these to the CoYML format.
        for pattern in md_result["patterns"]:
            sym = pattern["sym"]

            # Ignore non-intent symbols
            if not sym.startswith("intent:"):
                continue

            # The "|" is an old convention made by the parser, we roll back.
            intent = sym[7:].replace("|", " ")

            if intent not in user_messages:
                user_messages[intent] = []

            user_messages[intent].append(pattern["body"])

        # For the bot messages, we just copy them from the `utterances` dict.
        # The elements have the structure {"text": ..., "_context": ...}
        for intent, utterances in md_result["utterances"].items():
            if intent not in bot_messages:
                bot_messages[intent] = []

            if not isinstance(utterances, list):
                utterances = [utterances]

            for utterance in utterances:
                bot_messages[intent].append(utterance["text"])

    data = {
        "user_messages": user_messages,
        "bot_messages": bot_messages,
        "flows": flows,
    }

    return data
