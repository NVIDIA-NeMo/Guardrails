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

"""Removal warnings for LLMRails, the Colang runtimes, and the features that depend on them."""

import functools
import os
import sys
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from email.utils import format_datetime
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import RailsConfig

DEPRECATED_SINCE = "0.25.0"
REMOVAL_RELEASE = "0.27.0"
REMOVAL_MONTH = "January 2027"
MIGRATION_GUIDE_URL = "https://docs.nvidia.com/nemo/guardrails/latest/reference/0-25"

# The schedule is announced to the month; HTTP headers need a timestamp, so each takes the first of its month.
DEPRECATION_DATE = datetime(2026, 11, 1, tzinfo=timezone.utc)
SUNSET_DATE = datetime(2027, 1, 1, tzinfo=timezone.utc)

LLMRAILS_FALLBACK_ALTERNATIVE = "Use only rails that IORails supports; see the Rail Engine Support reference."
DIALOG_FLOWS_SUBJECT = "Using Colang dialog flows (define user, define flow)"
DIALOG_FLOWS_ALTERNATIVE = "IORails does not run them; move this logic into input and output rails."
CONFIG_PY_SUBJECT = "Using config.py"
CONFIG_PY_ALTERNATIVE = (
    "IORails does not import it, so its init() and registrations have no effect; "
    "configure models and rails in config.yml instead."
)
LANGCHAIN_FRAMEWORK_SUBJECT = "The LangChain model framework"
LANGCHAIN_BACKEND_ALTERNATIVE = "Serve the model through an OpenAI-compatible endpoint with the default framework."

_removal_warnings_silenced: ContextVar[bool] = ContextVar("llmrails_removal_warnings_silenced", default=False)

_DEPRECATION_HEADERS = {
    "Deprecation": f"@{int(DEPRECATION_DATE.timestamp())}",
    "Sunset": format_datetime(SUNSET_DATE, usegmt=True),
    "Link": f'<{MIGRATION_GUIDE_URL}>; rel="deprecation"; type="text/html"',
}

_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__)) + os.sep


class LLMRailsRemovalWarning(FutureWarning):
    """A feature that is removed together with the LLMRails engine."""


def llmrails_removal_message(subject: str, alternative: str) -> str:
    """Return the removal notice for *subject*, naming *alternative* as the way forward."""
    return (
        f"{subject} is deprecated since NeMo Guardrails {DEPRECATED_SINCE} and will be removed in "
        f"{REMOVAL_RELEASE} ({REMOVAL_MONTH}). {alternative} Migration guide: {MIGRATION_GUIDE_URL}"
    )


def warn_llmrails_removal(subject: str, alternative: str) -> None:
    """Emit an LLMRailsRemovalWarning attributed to the nearest caller outside the nemoguardrails package."""
    if _removal_warnings_silenced.get():
        return
    warnings.warn(
        llmrails_removal_message(subject, alternative),
        LLMRailsRemovalWarning,
        stacklevel=_stacklevel_outside_package(),
    )


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


def has_dialog_flows(config: "RailsConfig") -> bool:
    """Whether a Colang 1.0 config defines its own flows or user messages, neither of which IORails runs."""
    # A flow needs no `define user` block to match a user turn, so user flows count on their own.
    return bool(config.user_messages) or any(not _is_flow_llmrails_adds(flow) for flow in config.flows)


def _is_flow_llmrails_adds(flow: dict) -> bool:
    """Whether *flow* is one LLMRails appends to the config it starts on, unchanged from its own definition."""
    return (flow.get("id"), flow.get("source_code")) in _flows_llmrails_adds()


@functools.cache
def _flows_llmrails_adds() -> frozenset[tuple[Optional[str], Optional[str]]]:
    """The (id, source_code) of each flow LLMRails appends to a Colang 1.0 config: its default flows and the library's."""
    from nemoguardrails.colang import parse_colang_file

    # Mirrors the files LLMRails.__init__ parses; matching the source keeps a user's redefinition of a rail counted.
    paths = [os.path.join(_PACKAGE_DIR, "rails", "llm", "llm_flows.co")]
    for root, _, files in os.walk(os.path.join(_PACKAGE_DIR, "library")):
        paths.extend(os.path.join(root, name) for name in files if name.endswith(".co"))
    flows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            content = parse_colang_file(os.path.basename(path), content=f.read(), version="1.0")
        flows.extend(content.get("flows", []) if content else [])
    return frozenset((flow.get("id"), flow.get("source_code")) for flow in flows)


def has_config_py(config: "RailsConfig") -> bool:
    """Whether the config, or a config it imports, has the config.py module that only LLMRails loads."""
    # Mirrors the directories LLMRails searches for config.py when it starts.
    config_paths = list(config.imported_paths.values()) if config.imported_paths else []
    if config.config_path:
        config_paths.extend(path.strip() for path in config.config_path.split(",") if path.strip())
    return any(os.path.isfile(os.path.join(path, "config.py")) for path in config_paths)


def llmrails_deprecation_headers() -> dict[str, str]:
    """Return a copy of the RFC 9745 and RFC 8594 headers for a response served by the LLMRails engine."""
    return dict(_DEPRECATION_HEADERS)


def _is_package_file(filename: str) -> bool:
    return os.path.realpath(filename).startswith(_PACKAGE_DIR)


def _stacklevel_outside_package() -> int:
    """Return the stacklevel, counted from warn_llmrails_removal, of the nearest frame outside the package."""
    # Python 3.11 lacks warnings.warn(skip_file_prefixes=...), so walk the stack instead.
    frame = sys._getframe(2)
    stacklevel = 2
    while frame.f_back is not None and _is_package_file(frame.f_code.co_filename):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel
