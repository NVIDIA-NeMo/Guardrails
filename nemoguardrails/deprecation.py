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

"""Removal warnings for what leaves with the LLMRails engine: direct LLMRails use and Colang 2.x."""

import os
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

DEPRECATED_SINCE = "0.25.0"
# The removal criteria are not agreed yet, so the release is a lower bound, not a date.
REMOVAL_RELEASE = "0.27.0"

COLANG_2_ALTERNATIVE = "IORails will not support Colang 2.x, so there is no replacement."

_removal_warnings_silenced: ContextVar[bool] = ContextVar("llmrails_removal_warnings_silenced", default=False)

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep


class LLMRailsRemovalWarning(DeprecationWarning):
    """A feature that is removed together with the LLMRails engine."""


def llmrails_removal_message(subject: str, alternative: str) -> str:
    """Return the removal notice for *subject*, naming *alternative* as the way forward."""
    return (
        f"{subject} is deprecated since NeMo Guardrails {DEPRECATED_SINCE} and will be removed no earlier than "
        f"{REMOVAL_RELEASE}. {alternative}"
    )


def warn_llmrails_removal(subject: str, alternative: str) -> None:
    """Emit an LLMRailsRemovalWarning attributed to the nearest caller outside the nemoguardrails package."""
    if _removal_warnings_silenced.get():
        return
    warn_at_caller_outside_package(llmrails_removal_message(subject, alternative), LLMRailsRemovalWarning)


def warn_at_caller_outside_package(message: str, category: type[Warning]) -> None:
    """Emit *message* as *category*, attributed to the nearest caller outside the nemoguardrails package."""
    warnings.warn(message, category, stacklevel=_stacklevel_outside_package())


@contextmanager
def removal_warnings_silenced() -> Iterator[None]:
    """Silence LLMRailsRemovalWarnings in the block, so that one construction emits at most one."""
    token = _removal_warnings_silenced.set(True)
    try:
        yield
    finally:
        _removal_warnings_silenced.reset(token)


def called_from_package() -> bool:
    """Return True when the function calling this one was itself called from inside the nemoguardrails package."""
    return _is_package_file(sys._getframe(2).f_code.co_filename)


def _is_package_file(filename: str) -> bool:
    return os.path.realpath(filename).startswith(_PACKAGE_DIR)


def _stacklevel_outside_package() -> int:
    """Return the stacklevel, counted from warn_at_caller_outside_package, of the nearest frame outside the package."""
    # Python 3.11 lacks warnings.warn(skip_file_prefixes=...), so walk the stack instead.
    frame = sys._getframe(2)
    stacklevel = 2
    while frame.f_back is not None and _is_package_file(frame.f_code.co_filename):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel
