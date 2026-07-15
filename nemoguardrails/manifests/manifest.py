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

"""Versioned, declarative contract describing a rail and how to run it.

A `RailManifest` records everything the runtime and catalog need to know
about a rail without importing its implementation: descriptive
`RailMetadata` plus an executable `RailSpec` of config schema, flows,
actions, and surfaces. Descriptive fields are lenient so manifests stay
forward-compatible, while the executable spec is strict so misconfiguration fails
loudly at load time.
"""

import importlib
import re
import shlex
from enum import Enum
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from nemoguardrails.actions.rail_outcome import TransformTarget

RailCategory = Literal[
    "input",
    "output",
    "retrieval",
    "dialog",
    "execution",
    "tool_input",
    "tool_output",
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
RailStatus = RailLifecycle
BindingKind = Literal["surface_param", "context", "literal"]


class RailDirection(str, Enum):
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
    lifecycle: RailLifecycle = Field(default="stable", validation_alias=AliasChoices("lifecycle", "status"))
    owner: Optional[str] = None
    version: Optional[str] = None

    model_config = ConfigDict(extra="allow", frozen=True)

    @property
    def status(self) -> RailLifecycle:
        return self.lifecycle


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
    pass


class ActionRef(ImportTargetRef):
    name: str

    @field_validator("name")
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("ActionRef name must not be empty.")
        return value


ImportRef = Union[ConfigSpecRef, ActionRef]


class RailConfigSchema(BaseModel):
    key: str
    spec: ConfigSpecRef
    export_names: Tuple[str, ...] = Field(default=(), exclude=True)

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailFlows(BaseModel):
    files: Tuple[str, ...] = ("flows.co",)
    v1_files: Tuple[str, ...] = ("flows.v1.co",)
    flow_names: Tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailActions(BaseModel):
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
    name: str
    direction: RailDirection
    action: ActionRef
    bindings: Tuple[Binding, ...] = ()
    transform_target: Optional[TransformTarget] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class EnvVar(BaseModel):
    name: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ServiceRequirement(BaseModel):
    name: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRequirement(BaseModel):
    type: str
    required: bool = False
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailRequirements(BaseModel):
    extras: Tuple[str, ...] = ()
    env_vars: Tuple[EnvVar, ...] = ()
    services: Tuple[ServiceRequirement, ...] = ()
    models: Tuple[ModelRequirement, ...] = ()
    optional_dependencies: Tuple[str, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailPrivacy(BaseModel):
    sends_user_text: bool = False
    sends_bot_text: bool = False
    sends_retrieved_chunks: bool = False
    remote_services: Tuple[str, ...] = ()
    data_retention: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExampleRef(BaseModel):
    title: str
    path: str
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailSpec(BaseModel):
    config_schema: Optional[RailConfigSchema] = None
    flows: Optional[RailFlows] = None
    actions: Optional[RailActions] = None
    surfaces: Tuple[RailSurface, ...] = ()
    requirements: RailRequirements = Field(default_factory=RailRequirements)
    privacy: RailPrivacy = Field(default_factory=RailPrivacy)
    examples: Tuple[ExampleRef, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True)


class RailManifest(BaseModel):
    """Top-level, versioned manifest for a single rail.

    Accepts either the nested shape, where `spec` holds the config schema, flows,
    actions, and surfaces, or a flat mapping where those spec fields sit at the top
    level; the flat form is folded into `spec` during validation. The spec fields
    are also exposed as read-only properties (`manifest.surfaces`,
    `manifest.actions`, and so on) for convenient access.
    """

    manifest_version: Literal[1] = 1
    name: str
    metadata: RailMetadata = Field(default_factory=RailMetadata)
    spec: RailSpec = Field(default_factory=RailSpec)
    origin: str = Field(default="", exclude=True)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_manifest(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "spec" in value:
            return value
        data = dict(value)
        spec_fields = {
            "config_schema",
            "flows",
            "actions",
            "surfaces",
            "requirements",
            "privacy",
            "examples",
        }
        spec = {name: data.pop(name) for name in spec_fields if name in data}
        data["spec"] = spec
        return data

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

    @property
    def examples(self) -> Tuple[ExampleRef, ...]:
        return self.spec.examples


def import_ref_target(ref: ImportRef) -> str:
    if isinstance(ref, (ActionRef, ConfigSpecRef)):
        return ref.target
    raise TypeError("Import reference must be an ActionRef or ConfigSpecRef.")


def resolve_import_ref(ref: ImportRef) -> Any:
    target = import_ref_target(ref)
    module_name, _, attribute_path = target.partition(":")
    obj = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        obj = getattr(obj, attribute)
    return obj


def iter_manifest_import_refs(manifest: RailManifest) -> Tuple[ImportRef, ...]:
    refs = []
    if manifest.config_schema is not None:
        refs.append(manifest.config_schema.spec)
    if manifest.actions is not None:
        refs.extend(manifest.actions.refs)
    refs.extend(surface.action for surface in manifest.surfaces)
    return tuple(refs)


def iter_manifest_import_targets(manifest: RailManifest) -> Tuple[str, ...]:
    return tuple(import_ref_target(ref) for ref in iter_manifest_import_refs(manifest))


_PAREN_SURFACE = re.compile(r"^\s*([^($]+?)\s*\((.*)\)\s*$")
_SURFACE_PARAM_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_surface_parameter_items(items: Iterable[str]) -> Dict[str, str]:
    parameters = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Invalid surface parameter {item!r}; expected name=value.")
        key = key.strip().removeprefix("$")
        value = value.strip()
        if not _SURFACE_PARAM_NAME.fullmatch(key):
            raise ValueError(f"Invalid surface parameter name {key!r}.")
        if not value:
            raise ValueError(f"Surface parameter {key!r} must have a value.")
        if key in parameters:
            raise ValueError(f"Duplicate surface parameter {key!r}.")
        parameters[key] = value
    return parameters


def _validate_dollar_parameter_boundaries(arguments: str) -> None:
    quote: str | None = None
    escaped = False
    separated = False
    for character in arguments:
        if escaped:
            escaped = False
            separated = False
        elif character == "\\" and quote != "'":
            escaped = True
            separated = False
        elif quote is not None:
            if quote == character:
                quote = None
            separated = False
        elif character in {"'", '"'}:
            quote = character
            separated = False
        elif character == "$":
            if not separated:
                raise ValueError("Dollar-form surface parameters must be separated by whitespace.")
            separated = False
        else:
            separated = character.isspace()


def parse_configured_surface(flow_text: str) -> Tuple[str, Dict[str, str]]:
    flow_text = flow_text.strip()
    if not flow_text:
        raise ValueError("Configured surface must not be empty.")
    parenthesized = _PAREN_SURFACE.match(flow_text)
    if parenthesized is not None:
        name = parenthesized.group(1).strip()
        arguments = parenthesized.group(2).strip()
        if not arguments:
            return name, {}
        lexer = shlex.shlex(arguments, posix=True)
        lexer.whitespace += ","
        lexer.whitespace_split = True
        lexer.commenters = ""
        return name, _parse_surface_parameter_items(lexer)
    if "(" in flow_text or ")" in flow_text:
        raise ValueError(f"Invalid parenthesized surface reference {flow_text!r}.")
    name, separator, arguments = flow_text.partition("$")
    name = name.strip()
    if not name:
        raise ValueError("Configured surface name must not be empty.")
    if not separator:
        return name, {}
    _validate_dollar_parameter_boundaries(arguments)
    try:
        items = shlex.split(f"${arguments}")
    except ValueError as error:
        raise ValueError(f"Invalid surface parameters in {flow_text!r}: {error}") from error
    return name, _parse_surface_parameter_items(items)


def normalize_configured_surface_name(flow_text: str) -> str:
    flow_text = flow_text.strip()
    parenthesized = _PAREN_SURFACE.match(flow_text)
    if parenthesized is not None:
        return parenthesized.group(1).strip()
    return flow_text.split("$", 1)[0].strip()


def configured_rail_surfaces(
    direction: RailDirection, flows: Iterable[str], surfaces: Mapping[Tuple[RailDirection, str], RailSurface]
) -> Dict[str, RailSurface]:
    """Return unique declared surfaces enabled by the configured flows.

    Parameters and repeated configured instances are intentionally collapsed;
    iterate the flows with `parse_configured_surface` when instances are needed.
    """
    selected = {}
    for flow in flows:
        name, _ = parse_configured_surface(flow)
        surface = surfaces.get((direction, name))
        if surface is not None:
            selected[name] = surface
    return selected
