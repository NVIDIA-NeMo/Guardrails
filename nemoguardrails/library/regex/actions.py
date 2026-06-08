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

import logging
from typing import List, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)


class RegexDetectionResult(TypedDict):
    is_match: bool
    text: str
    detections: List[str]


def _regex_blocked_mapping(result: RegexDetectionResult) -> bool:
    """Return True (blocked) when a regex match was found."""
    return result.get("is_match", False)


def _get_regex_options(source: str, config: RailsConfig):
    """Return the RegexDetectionOptions for *source*, or None with a warning."""
    if source not in ("input", "output", "retrieval"):
        raise ValueError("source must be one of 'input', 'output', or 'retrieval'")

    regex_config = config.rails.config.regex_detection
    if regex_config is None:
        log.warning("No regex_detection configuration found.")
        return None

    options = getattr(regex_config, source, None)
    if options is None:
        log.warning("No regex rails configuration found for source: %s", source)
        return None

    if not options.compiled_patterns:
        log.debug("No regex patterns specified for source: %s", source)
        return None

    return options


@action(is_system_action=True, output_mapping=_regex_blocked_mapping)
async def detect_regex_pattern(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> RegexDetectionResult:
    """Checks whether the provided text matches any forbidden regex pattern.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to check.
        config: The rails configuration object.

    Returns:
        RegexDetectionResult: A TypedDict containing:
            - is_match (bool): Whether any pattern matched.
            - text (str): The original text that was checked.
            - detections (List[str]): List of pattern strings that matched.
    """
    options = _get_regex_options(source, config)
    if options is None:
        return RegexDetectionResult(is_match=False, text=text, detections=[])

    if not text:
        log.debug("Empty text provided, skipping regex check.")
        return RegexDetectionResult(is_match=False, text=text, detections=[])

    # Match against pre-compiled patterns and collect all matches.
    matched: List[str] = []
    for compiled, pcfg in zip(options.compiled_patterns, options.normalized_patterns):
        if compiled.search(text):
            log.info("Regex pattern matched: %s", pcfg.pattern)
            matched.append(pcfg.pattern)

    if matched:
        return RegexDetectionResult(is_match=True, text=text, detections=matched)

    return RegexDetectionResult(is_match=False, text=text, detections=[])


@action(is_system_action=True)
async def redact_regex_pattern(
    source: str,
    text: str,
    config: RailsConfig,
    **kwargs,
) -> str:
    """Replace all regex-matched spans with the configured mask token.

    Args:
        source: The source for the text, i.e. "input", "output", "retrieval".
        text: The text to redact.
        config: The rails configuration object.

    Returns:
        The text with every match of every configured pattern replaced by
        the mask_token (default ``<REDACTED>``).
    """
    options = _get_regex_options(source, config)
    if options is None:
        return text

    if not text:
        log.debug("Empty text provided, skipping regex redaction.")
        return text

    redacted = text
    for compiled, pcfg in zip(options.compiled_patterns, options.normalized_patterns):
        if compiled.search(redacted):
            log.info("Regex pattern redacted: %s", pcfg.pattern)
            mask = pcfg.mask_token
            redacted = compiled.sub(lambda m, _mask=mask: _mask, redacted)

    return redacted
