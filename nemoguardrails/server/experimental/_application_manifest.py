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

"""Load bounded deployment YAML into the existing application configuration.

The manifest is data, not a Python import or policy-authoring mechanism. Rails
directories remain trusted executable configuration, loaded only by the serving
application's lifespan. Validation errors deliberately omit input values.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, StrictStr, ValidationError, field_validator
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from nemoguardrails.server.experimental._configuration import ProviderConfig, ProxyConfig
from nemoguardrails.server.experimental.rails import ConfigEntry

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MANIFEST_DEPTH = 32
MAX_MANIFEST_NODES = 10_000


class ManifestError(ValueError):
    """Report a safe diagnostic without source excerpts or input values."""


class ManifestProviderConfig(ProviderConfig):
    """Restrict the Python provider declaration to a required fixed Rails ID."""

    rails_config_id: StrictStr = Field(...)


class ApplicationManifest(ProxyConfig):
    """Add an explicit file-format revision to the Python declaration.

    Revision 1 does not promise API stability. The standalone loader requires
    fixed Rails selections because it cannot supply trusted Python callbacks.
    """

    version: Literal[1]
    providers: tuple[ManifestProviderConfig, ...] = Field(min_length=1)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        """Reject boolean and floating-point spellings of the version."""
        if type(value) is not int:
            raise ValueError("The manifest version must be an integer.")
        return value


def _location(node: Node) -> str:
    """Return source coordinates without exposing a source excerpt."""
    return f"line {node.start_mark.line + 1}, column {node.start_mark.column + 1}"


class _ManifestLoader(yaml.SafeLoader):
    """Bound the YAML tree before construction and prohibit indirection."""

    def __init__(self, source: str):
        super().__init__(source)
        self._depth = 0
        self._nodes = 0

    def compose_node(self, parent, index):
        """Reject aliases, explicit tags, anchors and oversized nesting."""
        event = self.peek_event()
        location = f"line {event.start_mark.line + 1}, column {event.start_mark.column + 1}"
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise ManifestError(f"{location}: YAML anchors and aliases are not supported.")
        if getattr(event, "tag", None) is not None:
            raise ManifestError(f"{location}: explicit YAML tags are not supported.")
        self._nodes += 1
        self._depth += 1
        try:
            if self._depth > MAX_MANIFEST_DEPTH or self._nodes > MAX_MANIFEST_NODES:
                raise ManifestError(f"{location}: manifest structure exceeds the supported limits.")
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1


def _construct(loader: _ManifestLoader, node: Node, locations: dict, path: tuple = ()) -> object:
    """Construct only ordinary scalar/list/object data and record coordinates."""
    locations[path] = _location(node)
    if isinstance(node, MappingNode):
        result = {}
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ManifestError(f"{_location(key)}: mapping keys must be strings; merges are not supported.")
            if key.value in result:
                raise ManifestError(f"{_location(key)}: duplicate mapping key.")
            result[key.value] = _construct(loader, value, locations, (*path, key.value))
        return result
    if isinstance(node, SequenceNode):
        return [_construct(loader, value, locations, (*path, index)) for index, value in enumerate(node.value)]
    if not isinstance(node, ScalarNode) or node.tag not in {
        "tag:yaml.org,2002:str",
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:bool",
        "tag:yaml.org,2002:null",
    }:
        raise ManifestError(f"{_location(node)}: unsupported YAML value.")
    value = loader.construct_object(node)
    if isinstance(value, str) and "${" in value:
        raise ManifestError(f"{_location(node)}: environment interpolation is not supported.")
    return value


def load_manifest(path: Path) -> ProxyConfig:
    """Read one UTF-8 manifest without executing Rails or constructing owners.

    Relative Rails paths are anchored to the resolved manifest's directory,
    including when the caller supplies a symlink. Application assembly remains
    responsible for checking directory availability, routes and upstream URLs.
    """
    try:
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ManifestError("The manifest must be a regular file.")
        with path.open("rb") as source:
            raw = source.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError("The manifest exceeds the 1 MiB size limit.")
        source_text = raw.decode("utf-8")
    except (OSError, UnicodeError, RuntimeError):
        raise ManifestError("Cannot read the manifest as a UTF-8 file.") from None

    locations: dict[tuple, str] = {}
    loader = None
    try:
        loader = _ManifestLoader(source_text)
        node = loader.get_single_node()
        if not isinstance(node, MappingNode):
            raise ManifestError("The manifest root must be an object.")
        data = _construct(loader, node, locations)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f"line {mark.line + 1}, column {mark.column + 1}: " if mark is not None else ""
        raise ManifestError(f"{location}invalid YAML.") from None
    finally:
        if loader is not None:
            loader.dispose()
    try:
        manifest = ApplicationManifest.model_validate(data)
    except ValidationError as error:
        details = []
        for failure in error.errors(include_url=False, include_context=False, include_input=False)[:8]:
            location = tuple(failure["loc"])
            while location not in locations and location:
                location = location[:-1]
            details.append(f"{locations.get(location, 'manifest')}: {failure['type']}")
        raise ManifestError("Invalid manifest: " + "; ".join(details) + ". Check the manifest schema.") from None

    entries = tuple(
        ConfigEntry(entry.id, (path.parent / entry.config_path).resolve()) for entry in manifest.rails_configs
    )
    return ProxyConfig.model_validate(
        {**manifest.model_dump(exclude={"version", "rails_configs"}), "rails_configs": entries}
    )
