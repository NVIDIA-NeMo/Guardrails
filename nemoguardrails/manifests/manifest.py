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

"""Versioned, declarative contract describing a rail and how it runs.

A `RailManifest` combines descriptive `RailMetadata` with an executable
`RailSpec` of configuration, flows, actions, and surfaces. Import references
remain declarative until `resolve_import_ref` explicitly imports their targets.
Descriptive fields are lenient so manifests stay forward-compatible, while the
executable spec is strict so misconfiguration fails loudly at load time.
"""

import importlib
from enum import Enum
from typing import Any, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemoguardrails.actions.rail_outcome import TransformTarget

RailCategory = Literal[
    "input",
    "output",
    "retrieval",
    "dialog",
    "execution",
    "tool_call",
    "tool_result",
    "config",
]
RailCapability = Literal[
    "allow",
    "block",
    "classify",
    "content_safety",
    "detect_jailbreak",
    "detect_pii",
    "fact_check",
    "mask",
    "moderate",
    "topic_control",
    "transform",
]
RailLifecycle = Literal["stable", "experimental", "deprecated"]
BindingKind = Literal["surface_param", "context", "literal"]


class RailDirection(str, Enum):
    """Pipeline direction in which a rail surface runs."""

    INPUT = "input"
    OUTPUT = "output"
    RETRIEVAL = "retrieval"


class RailMetadata(BaseModel):
    """Descriptive, non-executable facets of a rail used by the catalog.

    None of these fields change runtime behavior; they drive display, discovery,
    and filtering. `categories` and `capabilities` are closed taxonomies (the
    pipeline stage a rail runs in and the functional behavior it advertises,
    respectively); use the free-form `tags` for labels that belong to neither.

    Unknown keys are preserved rather than rejected (`extra="allow"`) so a
    manifest authored against a newer schema still loads on an older install and
    authors can attach custom annotations without a schema change.
    """

    display_name: Optional[str] = None
    description: Optional[str] = None
    long_description: Optional[str] = None
    categories: Tuple[RailCategory, ...] = ()
    capabilities: Tuple[RailCapability, ...] = ()
    tags: Tuple[str, ...] = ()
    docs_url: Optional[str] = None
    lifecycle: RailLifecycle = "stable"
    owner: Optional[str] = None
    version: Optional[str] = None

    model_config = ConfigDict(extra="allow", frozen=True)


def _validate_import_target(target: str) -> str:
    module_name, separator, attribute_path = target.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"Invalid import reference {target!r}; expected 'module:attribute'.")
    return target


class ImportTargetRef(BaseModel):
    target: str

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("target")
    @classmethod
    def _target_must_be_import_ref(cls, value: str) -> str:
        return _validate_import_target(value)


class ConfigSpecRef(ImportTargetRef):
    """Import reference to a rail configuration specification."""


class ActionRef(ImportTargetRef):
    """Named import reference to a rail action."""

    name: str

    @field_validator("name")
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("ActionRef name must not be empty.")
        return value


ImportRef = Union[ConfigSpecRef, ActionRef]


class RailConfigSchema(BaseModel):
    """Manifest reference to a rail's typed configuration schema."""

    key: str
    spec: ConfigSpecRef
    export_names: Tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailFlows(BaseModel):
    """Colang flow files and flow names declared by a rail."""

    files: Tuple[str, ...] = ("flows.co",)
    v1_files: Tuple[str, ...] = ("flows.v1.co",)
    flow_names: Tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailActions(BaseModel):
    """Import references for the actions declared by a rail."""

    refs: Tuple[ActionRef, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class Binding(BaseModel):
    """Maps a single surface action parameter to its value source.

    Each binding tells the runtime where one argument of a surface's action comes
    from. Prefer the constructor classmethods over building instances directly, as
    they set `kind` and the relevant fields correctly.
    """

    kind: BindingKind
    action_param: str
    key: Optional[str] = None
    value: Any = None
    required: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _source_must_match_kind(self) -> "Binding":
        if self.kind == "literal" and self.key is not None:
            raise ValueError("Literal bindings cannot declare a source key.")
        if self.kind != "literal" and self.key is None:
            raise ValueError(f"{self.kind} bindings must declare a source key.")
        return self

    @classmethod
    def surface_param(cls, action_param: str, name: str, *, required: bool = True) -> "Binding":
        """Bind `action_param` to a caller-supplied surface parameter.

        Args:
            action_param: Name of the action parameter to populate.
            name: Name of the surface parameter that supplies the value.
            required: Whether the surface parameter must be provided.

        Returns:
            A `surface_param` binding.
        """
        return cls(kind="surface_param", action_param=action_param, key=name, required=required)

    @classmethod
    def context(cls, action_param: str, key: str, *, required: bool = True) -> "Binding":
        """Bind `action_param` to a context variable.

        Args:
            action_param: Name of the action parameter to populate.
            key: Name of the context variable that supplies the value.
            required: Whether the context variable must be present.

        Returns:
            A `context` binding.
        """
        return cls(kind="context", action_param=action_param, key=key, required=required)

    @classmethod
    def literal(cls, action_param: str, value: Any) -> "Binding":
        """Bind `action_param` to a fixed value baked into the manifest.

        Args:
            action_param: Name of the action parameter to populate.
            value: Constant value passed to the action.

        Returns:
            A `literal` binding.
        """
        return cls(kind="literal", action_param=action_param, value=value)


class RailSurface(BaseModel):
    """Configured flow surface mapped to a declared rail action."""

    name: str
    direction: RailDirection
    action: ActionRef
    bindings: Tuple[Binding, ...] = ()
    transform_target: Optional[TransformTarget] = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_execution_contract(self) -> "RailSurface":
        seen = set()
        duplicates = set()
        for binding in self.bindings:
            if binding.action_param in seen:
                duplicates.add(binding.action_param)
            seen.add(binding.action_param)
        if duplicates:
            raise ValueError(f"Rail surfaces cannot bind action parameters more than once: {sorted(duplicates)}.")
        return self


class EnvVar(BaseModel):
    """Environment variable declared by a rail requirement."""

    name: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceRequirement(BaseModel):
    """External service declared by a rail requirement."""

    name: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRequirement(BaseModel):
    """Model resource declared by a rail requirement."""

    type: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailRequirements(BaseModel):
    """Installation and runtime resources declared by a rail."""

    extras: Tuple[str, ...] = ()
    env_vars: Tuple[EnvVar, ...] = ()
    services: Tuple[ServiceRequirement, ...] = ()
    models: Tuple[ModelRequirement, ...] = ()
    optional_dependencies: Tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailPrivacy(BaseModel):
    """Data handling and remote-service behavior declared by a rail."""

    sends_user_text: bool = False
    sends_bot_text: bool = False
    sends_retrieved_chunks: bool = False
    remote_services: Tuple[str, ...] = ()
    data_retention: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailSpec(BaseModel):
    """Executable configuration, flows, actions, and requirements for a rail."""

    config_schema: Optional[RailConfigSchema] = None
    flows: Optional[RailFlows] = None
    actions: Optional[RailActions] = None
    surfaces: Tuple[RailSurface, ...] = ()
    requirements: RailRequirements = Field(default_factory=RailRequirements)
    privacy: RailPrivacy = Field(default_factory=RailPrivacy)
    model_config = ConfigDict(extra="forbid", frozen=True)


class RailManifest(BaseModel):
    """Top-level, versioned manifest for a single rail.

    The nested `spec` holds executable declarations such as the config schema,
    flows, actions, surfaces, requirements, and privacy metadata. Those fields
    are also exposed as read-only properties for convenient access.
    """

    manifest_version: Literal[1] = 1
    name: str
    metadata: RailMetadata = Field(default_factory=RailMetadata)
    spec: RailSpec = Field(default_factory=RailSpec)
    origin: str = Field(default="", exclude=True)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def config_schema(self) -> Optional[RailConfigSchema]:
        return self.spec.config_schema

    @property
    def flows(self) -> Optional[RailFlows]:
        return self.spec.flows

    @property
    def actions(self) -> Optional[RailActions]:
        return self.spec.actions

    @property
    def surfaces(self) -> Tuple[RailSurface, ...]:
        return self.spec.surfaces

    @property
    def requirements(self) -> RailRequirements:
        return self.spec.requirements

    @property
    def privacy(self) -> RailPrivacy:
        return self.spec.privacy


def import_ref_target(ref: ImportRef) -> str:
    """Return the import target encoded by a supported manifest reference."""
    if isinstance(ref, (ActionRef, ConfigSpecRef)):
        return ref.target
    raise TypeError("Import reference must be an ActionRef or ConfigSpecRef.")


def resolve_import_ref(ref: ImportRef) -> Any:
    """Import and return the Python object referenced by a manifest entry."""
    target = import_ref_target(ref)
    module_name, _, attribute_path = target.partition(":")
    obj = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        obj = getattr(obj, attribute)
    return obj


def iter_manifest_import_refs(manifest: RailManifest) -> Tuple[ImportRef, ...]:
    """Return every configuration and action import reference in a manifest."""
    refs = []
    if manifest.config_schema is not None:
        refs.append(manifest.config_schema.spec)
    if manifest.actions is not None:
        refs.extend(manifest.actions.refs)
    refs.extend(surface.action for surface in manifest.surfaces)
    return tuple(refs)
