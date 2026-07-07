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

import re
from typing import List

# Persona injection attack patterns based on the C2 (Gradual Identity Substitution)
# behavior class documented in Dwivedi (2026, preprint, under review).
# Covers: direct persona assignment, mode activation, override commands,
# and academic/hypothetical framing used to bypass identity constraints.
_PERSONA_PATTERNS = [
    # Direct persona assignment
    r"\byou are now\b",
    r"\bact as\b",
    r"\bfrom now on you\b",
    r"\bpretend (to be|you are|you're)\b",
    r"\broleplay as\b",
    r"\byou'?re now\b",
    # Special mode and persona names
    r"\bdan\b",
    r"\bdevmode\b",
    r"\bdeveloper mode\b",
    r"\bunrestricted mode\b",
    r"\bjailbreak mode\b",
    r"\bno (restrictions|limits|rules|guidelines|filters)\b",
    # Override / reset commands
    r"\bignore (your|all|previous|prior) (instructions|guidelines|rules|training|constraints)\b",
    r"\bforget (your|all) (instructions|guidelines|training|previous)\b",
    r"\bdisregard (your|all) (instructions|guidelines|rules)\b",
    r"\boverride (your|all)? ?(previous|prior|existing)? ?(instructions|guidelines|rules|training)\b",
    # Hypothetical / academic framing for persona injection
    r"\bimagine (you|an ai).{0,60}(no (restrictions|guidelines)|unrestricted|without (filters|constraints))\b",
    r"\bfor (my|our|this) (research|study|experiment|test).{0,60}(pretend|imagine|act as|roleplay)\b",
    r"\bhypothetically.{0,40}(no restrictions|unrestricted|without (constraints|guidelines))\b",
]

# Assistant refusal markers — used to detect the Refusal Counterfactual pattern:
# a user rephrases a refused request using persona injection framing.
_REFUSAL_PATTERNS = [
    r"\bi (can'?t|won'?t|am unable to|cannot|refuse to)\b",
    r"\bi'?m (sorry|afraid)[,.]? (but )?(i|that)\b",
    r"\bi'?m not able to\b",
    r"\bi must decline\b",
    r"\bagainst my (guidelines|training|values|principles|policy)\b",
    r"\bI'?m designed to\b.{0,40}\bnot\b",
]

_COMPILED_PERSONA = [re.compile(p, re.IGNORECASE) for p in _PERSONA_PATTERNS]
_COMPILED_REFUSAL = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]


def _has_persona_marker(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED_PERSONA)


def _has_refusal_marker(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED_REFUSAL)


def check_persona_injection_arc(
    messages: List[dict],
    min_turns_flagged: int = 2,
    window: int = 8,
) -> dict:
    """
    Detect persona injection patterns accumulated across a conversation window.

    A single-turn persona injection attempt is already handled by the existing
    ``jailbreak_detection`` heuristics. This check targets the multi-turn case
    where an adversary spreads persona injection language across several turns
    to gradually normalise an alternative identity (C2 escalation pattern).

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts in
            chronological order, as produced by ``to_chat_messages``.
        min_turns_flagged: Number of user turns that must contain at least one
            persona injection marker before the arc is flagged. Default: 2.
        window: Only the most recent *window* turns are examined. Default: 8.

    Returns:
        ``{"jailbreak": bool}``
    """
    user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
    user_messages = user_messages[-window:]

    if len(user_messages) < min_turns_flagged:
        return {"jailbreak": False}

    flagged = sum(1 for msg in user_messages if _has_persona_marker(msg))
    return {"jailbreak": flagged >= min_turns_flagged}


def check_post_refusal_rephrase(
    messages: List[dict],
    window: int = 8,
) -> dict:
    """
    Detect the Refusal Counterfactual pattern.

    If the model refused a request in one turn and the immediately following
    user message contains persona injection markers, the user is attempting to
    bypass the refusal through rephrasing — a high-confidence signal of
    adversarial intent regardless of topic.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        window: Number of most-recent messages to examine. Default: 8.

    Returns:
        ``{"jailbreak": bool}``
    """
    windowed = messages[-window:]

    for i, msg in enumerate(windowed):
        if msg.get("role") != "assistant":
            continue

        if not _has_refusal_marker(msg.get("content", "")):
            continue

        # Look at the very next user turn after the refusal.
        for j in range(i + 1, len(windowed)):
            if windowed[j].get("role") == "user":
                if _has_persona_marker(windowed[j].get("content", "")):
                    return {"jailbreak": True}
                break

    return {"jailbreak": False}
