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

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from textwrap import dedent

from nemoguardrails import RailsConfig


@dataclass(frozen=True)
class RailsConfigSource:
    """A RailsConfig descriptor: either a filesystem path or inline yaml/colang content.

    Construct via the factory methods rather than the raw constructor: use
    ``from_path(base_dir, name)`` for on-disk configs and ``from_content(name, ...)``
    for inline ones. ``name`` is used for parametrize IDs and error messages.
    """

    name: str
    path: Path | None = None
    yaml_content: str = ""
    colang_content: str = ""

    @classmethod
    def from_path(cls, base_dir: Path, name: str) -> RailsConfigSource:
        return cls(name=name, path=base_dir / name)

    @classmethod
    def from_content(cls, name: str, *, yaml_content: str = "", colang_content: str = "") -> RailsConfigSource:
        return cls(name=name, yaml_content=yaml_content, colang_content=colang_content)


@lru_cache(maxsize=None)
def _cached_config(source: RailsConfigSource) -> RailsConfig:
    if source.path is not None:
        return RailsConfig.from_path(str(source.path))
    return RailsConfig.from_content(
        colang_content=dedent(source.colang_content).strip(),
        yaml_content=dedent(source.yaml_content).strip(),
    )


def load_config(source: RailsConfigSource) -> RailsConfig:
    """Load a ``RailsConfig`` from a ``RailsConfigSource``."""
    return _cached_config(source).model_copy(deep=True)


def enable_streaming(
    config: RailsConfig,
    *,
    chunk_size: int | None = None,
    context_size: int | None = None,
    stream_first: bool | None = None,
) -> RailsConfig:
    """Return a copy of ``config`` with output-rail streaming enabled.

    Source config is not mutated. Optional overrides apply only when set; otherwise the
    framework defaults from ``OutputRailsStreamingConfig`` apply.
    """
    config = config.model_copy(deep=True)
    config.rails.output.streaming.enabled = True
    if chunk_size is not None:
        config.rails.output.streaming.chunk_size = chunk_size
    if context_size is not None:
        config.rails.output.streaming.context_size = context_size
    if stream_first is not None:
        config.rails.output.streaming.stream_first = stream_first
    return config
