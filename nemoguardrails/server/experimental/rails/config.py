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

"""Define trusted Guardrails configuration resolution for the proxy server."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from fastapi import Request


@dataclass(frozen=True, slots=True)
class ConfigEntry:
    """Associate a deployment-owned identifier with a Rails config directory."""

    id: str
    config_path: Path


@dataclass(frozen=True, slots=True)
class ConfigContext:
    """Describe trusted request context available to a config resolver."""

    request: Request
    provider_type: str
    provider_instance_id: str
    operation_name: str


ConfigResolver: TypeAlias = Callable[[ConfigContext], str | Awaitable[str]]


def _contains_config(config_path: Path) -> bool:
    candidates = (
        config_path / "config.yml",
        config_path / "config.yaml",
        config_path / "config" / "config.yml",
        config_path / "config" / "config.yaml",
    )
    return any(candidate.is_file() for candidate in candidates)


def validate_config_entries(entries: Sequence[ConfigEntry]) -> tuple[ConfigEntry, ...]:
    """Normalize and validate the config entries declared by one application."""

    resolved = []
    ids = set()
    for entry in entries:
        if not isinstance(entry, ConfigEntry):
            raise TypeError("Rails config entries must be ConfigEntry instances.")
        if not entry.id or entry.id != entry.id.strip():
            raise ValueError("A Rails config id must be non-empty and trimmed.")
        if entry.id in ids:
            raise ValueError(f"Duplicate Rails config id: {entry.id}")
        config_path = entry.config_path.expanduser().resolve()
        if not config_path.is_dir() or not _contains_config(config_path):
            raise ValueError(f"Rails config '{entry.id}' does not reference a configuration directory.")
        ids.add(entry.id)
        resolved.append(ConfigEntry(id=entry.id, config_path=config_path))
    return tuple(resolved)


__all__ = [
    "ConfigContext",
    "ConfigEntry",
    "ConfigResolver",
]
