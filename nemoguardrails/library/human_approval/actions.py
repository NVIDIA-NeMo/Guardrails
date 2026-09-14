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

import logging
from typing import List, TypedDict

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action

log = logging.getLogger(__name__)


class HumanApprovalResult(TypedDict):
    needs_approval: bool
    text: str
    matched_patterns: List[str]


def _get_approval_config(config: RailsConfig):
    return getattr(config.rails.config, "human_approval", None)


@action()
async def human_approval_check(text: str, config: RailsConfig) -> HumanApprovalResult:
    """Check whether the provided text matches any patterns that require human approval."""
    approval_config = _get_approval_config(config)

    if approval_config is None or not approval_config.compiled_patterns:
        return HumanApprovalResult(needs_approval=False, text=text, matched_patterns=[])

    if not text:
        return HumanApprovalResult(needs_approval=False, text=text, matched_patterns=[])

    matched: List[str] = []
    for compiled, raw_pattern in zip(approval_config.compiled_patterns, approval_config.patterns):
        if compiled.search(text):
            log.info("Human approval pattern matched: %s", raw_pattern)
            matched.append(raw_pattern)

    if matched:
        return HumanApprovalResult(needs_approval=True, text=text, matched_patterns=matched)

    return HumanApprovalResult(needs_approval=False, text=text, matched_patterns=[])


@action(is_system_action=True)
async def human_approval_matches_keywords(text: str, config: RailsConfig) -> bool:
    """Return True when the text matches a configured approval keyword (case-insensitive)."""
    approval_config = _get_approval_config(config)
    if approval_config is None or not text:
        return False

    normalized = text.strip().lower()
    return any(keyword.lower() == normalized for keyword in approval_config.approval_keywords)
