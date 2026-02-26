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


import secrets
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

LLMMessage: TypeAlias = dict[str, str]  # e.g. {"role": "user", "content": "What can you do?"}
LLMMessages: TypeAlias = list[LLMMessage]


class RailDirection(Enum):
    """Direction of a rail check, used for logging."""

    INPUT = "Input"
    OUTPUT = "Output"


@dataclass(frozen=True, slots=True)
class RailResult:
    """Result of a rail safety check."""

    is_safe: bool
    reason: str | None = None


# Default max character length for truncate(). Used to keep DEBUG log lines short.
LOG_CONTENT_TRUNCATE_LENGTH = 200

_request_id_var: ContextVar[str] = ContextVar("request_id", default="no-req-id")


def set_new_request_id() -> Token[str]:
    """Generate an 8-char hex request ID, set it in the current context, and return the reset token."""
    rid = secrets.token_hex(4)  # 4 bytes -> 8 hex chars
    return _request_id_var.set(rid)


def get_request_id() -> str:
    """Return the current per-request correlation ID."""
    return _request_id_var.get()


def reset_request_id(token: Token[str]) -> None:
    """Restore the request ID ContextVar to its previous value."""
    _request_id_var.reset(token)


def truncate(text: object, max_len: int | None = None) -> str:
    """Return ``str(text)`` truncated to *max_len* characters (default: LOG_CONTENT_TRUNCATE_LENGTH)."""
    s = str(text)
    limit = max_len if max_len is not None else LOG_CONTENT_TRUNCATE_LENGTH
    if len(s) <= limit:
        return s
    return s[:limit] + "..."
