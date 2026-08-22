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

"""Static safety analysis for configured regex patterns (#2203).

Python's ``re`` engine backtracks, so a pattern that applies an unbounded
quantifier inside the body of another unbounded quantifier (``(a+)+``,
``(?:\\w+\\s*)*``, ...) can exhibit catastrophic backtracking on crafted
input and pin a worker thread even though the pattern is syntactically
valid. Configuration-load-time syntax checking does not catch this.

``ensure_pattern_is_safe`` implements a conservative, documented safe
subset: reject any pattern where an unbounded repetition (``*``, ``+``,
``{n,}``, lazy variants included) transitively contains another unbounded
repetition through grouping, alternation, lookarounds, or bounded
repetitions. Patterns in this subset run in linear-ish time on the
standard engine.

Known limitation (documented, accepted): ambiguous alternations without
nested quantifiers (``(a|aa)*b``) are not detected by this analysis; the
input-length cap applied by :mod:`nemoguardrails.library.regex.actions`
is the backstop for residual cases.
"""

import logging

try:
    from re import _constants as _sre_constants
    from re import _parser as _sre_parser
except ImportError:  # pragma: no cover - Python < 3.11
    import sre_constants as _sre_constants
    import sre_parse as _sre_parser


log = logging.getLogger(__name__)


MAXREPEAT = _sre_constants.MAXREPEAT
_REPEAT_OPS = (_sre_constants.MAX_REPEAT, _sre_constants.MIN_REPEAT)
_MAX_ANALYSIS_DEPTH = 64


class UnsafeRegexPatternError(ValueError):
    """Raised when a configured pattern is rejected as unsafe."""


def _child_sequences(op, av):
    """Yield the nested sequences of a parsed node, if any."""
    if op == _sre_constants.SUBPATTERN:
        return (av[3],)
    if op == _sre_constants.BRANCH:
        return tuple(av[1])
    if op in (_sre_constants.ASSERT, _sre_constants.ASSERT_NOT):
        return (av[1],)
    if getattr(_sre_constants, "ATOMIC_GROUP", None) is not None and op == _sre_constants.ATOMIC_GROUP:
        return (av,)
    return ()


def _has_unbounded_descendant(seq, depth):
    """True if any descendant of ``seq`` applies an unbounded repetition."""
    if depth > _MAX_ANALYSIS_DEPTH:
        # re itself bounds pattern size/complexity; treat pathological
        # parse trees conservatively rather than recursing forever.
        return True
    for op, av in seq:
        if op in _REPEAT_OPS:
            maximum = av[1]
            if maximum == MAXREPEAT:
                return True
            if _has_unbounded_descendant(av[2], depth + 1):
                return True
            continue
        found = False
        for child in _child_sequences(op, av):
            if _has_unbounded_descendant(child, depth + 1):
                found = True
                break
        if found:
            return True
    return False


def _contains_unsafe_nesting(seq, depth):
    """True if an unbounded repetition contains another one below it."""
    if depth > _MAX_ANALYSIS_DEPTH:
        return True
    for op, av in seq:
        if op in _REPEAT_OPS:
            _, maximum, body = av
            if maximum == MAXREPEAT and _has_unbounded_descendant(body, depth + 1):
                return True
            if _contains_unsafe_nesting(body, depth + 1):
                return True
            continue
        found = False
        for child in _child_sequences(op, av):
            if _contains_unsafe_nesting(child, depth + 1):
                found = True
                break
        if found:
            return True
    return False


def ensure_pattern_is_safe(pattern: str, flags: int = 0) -> None:
    """Reject patterns that can exhibit catastrophic backtracking.

    Called after a successful ``re.compile`` so syntax errors have already
    been reported by the compiler; parse failures here are ignored because
    they can only be syntax problems.
    """
    try:
        parsed = _sre_parser.parse(pattern, flags)
    except Exception:  # noqa: BLE001 - unreachable after re.compile succeeded
        log.debug("Pattern %r could not be re-parsed for safety analysis.", pattern)
        return
    if _contains_unsafe_nesting(parsed, 0):
        raise UnsafeRegexPatternError(
            "pattern nests an unbounded quantifier inside another "
            "unbounded quantifier (catastrophic backtracking risk); "
            "rewrite it so repetitions do not enclose other "
            "variable-length repetitions"
        )
