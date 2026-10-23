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
from typing import ClassVar, TypeAlias

from fastapi import Request
from pydantic import ConfigDict

from nemoguardrails.server.experimental._configuration_error import DeploymentValidationError


@dataclass(frozen=True, slots=True)
class ConfigEntry:
    """Associate a deployment-owned identifier with a Rails config directory."""

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid", revalidate_instances="always")

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


def validate_config_entries(entries: Sequence[ConfigEntry]) -> tuple[ConfigEntry, ...]:
    """Normalize and validate the config entries declared by one application."""

    resolved = []
    ids = set()
    for entry in entries:
        if not isinstance(entry, ConfigEntry):
            raise TypeError("Rails config entries must be ConfigEntry instances.")
        if not entry.id or entry.id != entry.id.strip():
            raise DeploymentValidationError("A Rails config id must be non-empty and trimmed.")
        if entry.id in ids:
            raise DeploymentValidationError("Duplicate Rails config id.")
        config_path = entry.config_path.expanduser().resolve()
        if not config_path.is_dir():
            raise DeploymentValidationError("A Rails config entry does not reference a configuration directory.")
        ids.add(entry.id)
        resolved.append(ConfigEntry(id=entry.id, config_path=config_path))
    return tuple(resolved)


__all__ = [
    "ConfigContext",
    "ConfigEntry",
    "ConfigResolver",
]
