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

import json
import logging
from typing import List, Optional, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome, TransformTarget
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult, scope_arguments, tool_output_validation
from nemoguardrails.library.regex.rail_config import RegexDetectionOptions
from nemoguardrails.types import ToolCall

log = logging.getLogger(__name__)


class RegexDetectionResult(TypedDict):
    is_match: bool
    text: str
    detections: List[str]


def _regex_outcome(source: str, result: RegexDetectionResult) -> RailOutcome:
    metadata = dict(result)
    metadata["source"] = source
    if result["is_match"] and source == "retrieval":
        return RailOutcome.transform([(TransformTarget.RELEVANT_CHUNKS, "")], metadata=metadata)
    if result["is_match"]:
        return RailOutcome.block(metadata=metadata)
    return RailOutcome.allow(metadata=metadata)


def _match_patterns(source: str, text: str, options: RegexDetectionOptions) -> RegexDetectionResult:
    """Match *text* against a pattern group's pre-compiled patterns, logging as each miss/hit occurs.

    Extracted from detect_regex_pattern's own matching loop so detect_tool_output_regex_pattern
    and detect_tool_input_regex_pattern don't duplicate it. detect_regex_pattern's inline loop
    below is left as-is for now. TODO: refactor it to call this helper too.
    """
    compiled_patterns = options.compiled_patterns
    if not compiled_patterns:
        log.debug("No regex patterns specified for source: %s", source)
        return RegexDetectionResult(is_match=False, text=text, detections=[])

    if not text:
        log.debug("Empty text provided, skipping regex check.")
        return RegexDetectionResult(is_match=False, text=text, detections=[])

    matched: List[str] = []
    for compiled, raw_pattern in zip(compiled_patterns, options.patterns):
        if compiled.search(text):
            log.info("Regex pattern matched: %s", raw_pattern)
            matched.append(raw_pattern)
    return RegexDetectionResult(is_match=bool(matched), text=text, detections=matched)


@action(is_system_action=True)
async def detect_regex_pattern(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> RailOutcome:
    """Checks whether the provided text matches any forbidden regex pattern.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        RailOutcome with RegexDetectionResult fields in metadata:
            - is_match (bool): Whether any pattern matched.
            - text (str): The original text that was checked.
            - detections (List[str]): List of pattern strings that matched.
    """
    if source not in ("input", "output", "retrieval"):
        raise ValueError("source must be one of 'input', 'output', or 'retrieval'")

    regex_config = config.rails.config.regex_detection
    if regex_config is None:
        log.warning("No regex_detection configuration found.")
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    options = getattr(regex_config, source, None)

    if options is None:
        log.warning("No regex rails configuration found for source: %s", source)
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    compiled_patterns = options.compiled_patterns
    if not compiled_patterns:
        log.debug("No regex patterns specified for source: %s", source)
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    if not text:
        log.debug("Empty text provided, skipping regex check.")
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    # Match against pre-compiled patterns and collect all matches.
    matched: List[str] = []
    for compiled, raw_pattern in zip(compiled_patterns, options.patterns):
        if compiled.search(text):
            log.info("Regex pattern matched: %s", raw_pattern)
            matched.append(raw_pattern)

    if matched:
        return _regex_outcome(source, RegexDetectionResult(is_match=True, text=text, detections=matched))

    return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))


@action(is_system_action=True)
@tool_output_validation
async def detect_tool_output_regex_pattern(
    source: str,
    tool_call: ToolCall,
    tool_definition: Optional[Tool],
    config: RailsConfig,
    argument_name: Optional[str] = None,
    **kwargs,
) -> RailOutcome:
    """Checks a tool call's arguments against that tool's regex patterns.

    Args:
        source: Fixed per surface, always "tool_output".
        tool_call: The tool call to check.
        tool_definition: The declared tool, used only by @tool_output_validation to check
            the call's arguments against its schema before this function runs.
        config: The rails configuration object.
        argument_name: Narrows the checked arguments to one named argument, from
            `$argument=` on the flow. Absent, the full arguments dict is checked.

    Returns:
        RailOutcome with RegexDetectionResult fields in metadata (see detect_regex_pattern).
    """
    if source != "tool_output":
        raise ValueError("source must be 'tool_output'")

    tool_name = tool_call.function.name or tool_call.type
    text = json.dumps(scope_arguments(tool_call.function.arguments, argument_name))

    regex_config = config.rails.config.regex_detection
    options = getattr(regex_config, source).get(tool_name)
    if options is None:
        log.debug("No regex patterns configured for tool %r under source: %s", tool_name, source)
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    return _regex_outcome(source, _match_patterns(source, text, options))


@action(is_system_action=True)
async def detect_tool_input_regex_pattern(
    source: str,
    tool_call: ToolCall,
    tool_result: ToolResult,
    config: RailsConfig,
    **kwargs,
) -> RailOutcome:
    """Checks a tool result's content against that tool's regex patterns.

    Args:
        source: Fixed per surface, always "tool_input".
        tool_call: The prior call this result answers, used to resolve the tool name.
        tool_result: The tool result to check.
        config: The rails configuration object.

    Returns:
        RailOutcome with RegexDetectionResult fields in metadata (see detect_regex_pattern).
    """
    if source != "tool_input":
        raise ValueError("source must be 'tool_input'")

    tool_name = tool_call.function.name or tool_call.type
    content = tool_result.content
    text = content if isinstance(content, str) else json.dumps(content)

    regex_config = config.rails.config.regex_detection
    options = getattr(regex_config, source).get(tool_name)
    if options is None:
        log.debug("No regex patterns configured for tool %r under source: %s", tool_name, source)
        return _regex_outcome(source, RegexDetectionResult(is_match=False, text=text, detections=[]))

    return _regex_outcome(source, _match_patterns(source, text, options))
