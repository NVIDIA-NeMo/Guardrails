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

"""Validate deployment declarations and resolve them to existing proxy bindings.

These models select reviewed Python policy; they neither load Rails directories
nor construct runtime owners. Provider URLs never select a dialect implicitly.
"""

from collections.abc import Awaitable
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator
from typing_extensions import Self

from nemoguardrails.server.experimental._http_kernel import (
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_MAX_RESPONSE_BODY_BYTES,
)
from nemoguardrails.server.experimental._runtime import DEFAULT_CONFIG_RESOLUTION_TIMEOUT
from nemoguardrails.server.experimental.provider.endpoint import GuardedJsonEndpoint
from nemoguardrails.server.experimental.provider.integration import ProviderIntegration
from nemoguardrails.server.experimental.rails import ConfigContext, ConfigEntry, ConfigResolver


class ProviderConfig(BaseModel):
    """Select one built-in provider instance and its guarded operations.

    An omitted dialect selects the provider's reviewed native bindings.
    Operations install guard bindings, not an HTTP route allowlist. Other
    routes retain passthrough behavior. A fixed Rails ID bypasses the user
    resolver; otherwise the application must supply a trusted resolver.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, revalidate_instances="always")

    provider: Literal["openai"]
    instance_id: StrictStr
    base_url: StrictStr = Field(repr=False)
    operations: tuple[Literal["chat_completions", "responses"], ...] = Field(min_length=1)
    mount_path: StrictStr = ""
    dialect: Literal["litellm"] | None = None
    rails_config_id: StrictStr | None = None

    @field_validator("operations", mode="before")
    @classmethod
    def ordered_operations(cls, value: object) -> object:
        """Require an ordered concrete selection and reject coercible containers."""
        if not isinstance(value, (list, tuple)):
            raise ValueError("Operations must be a list or tuple of names.")
        return value

    @field_validator("operations")
    @classmethod
    def unique_operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicates instead of silently collapsing the declared policy."""
        if len(value) != len(set(value)):
            raise ValueError("Operations must not contain duplicate names.")
        return value

    @field_validator("rails_config_id")
    @classmethod
    def fixed_config_id(cls, value: str | None) -> str | None:
        """Accept an exact catalogue ID or an explicit dynamic selection."""
        if value is not None and (not value or value != value.strip()):
            raise ValueError("The fixed Rails config ID must be non-empty and trimmed.")
        return value


class ProxyConfig(BaseModel):
    """Declare built-in instances and one shared Rails configuration catalogue.

    Construction validates data without reading configuration files. Paths are
    resolved by application assembly using the existing Python path semantics.
    Callbacks are supplied to the application factory, not stored in this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, revalidate_instances="always")

    providers: tuple[ProviderConfig, ...] = Field(min_length=1)
    rails_configs: tuple[ConfigEntry, ...] = Field(min_length=1)
    path_prefix: StrictStr = ""
    max_request_body_bytes: StrictInt = Field(default=DEFAULT_MAX_REQUEST_BODY_BYTES, gt=0)
    max_response_body_bytes: StrictInt = Field(default=DEFAULT_MAX_RESPONSE_BODY_BYTES, gt=0)
    timeout: float = Field(default=30.0, strict=True, gt=0, allow_inf_nan=False)
    config_resolution_timeout: float = Field(
        default=DEFAULT_CONFIG_RESOLUTION_TIMEOUT, strict=True, gt=0, allow_inf_nan=False
    )

    @field_validator("providers", "rails_configs", mode="before")
    @classmethod
    def ordered_entries(cls, value: object) -> object:
        """Snapshot lists or tuples rather than consuming arbitrary iterables."""
        if not isinstance(value, (list, tuple)):
            raise ValueError("Configuration entries must be supplied as a list or tuple.")
        return value

    @model_validator(mode="after")
    def config_references(self) -> Self:
        """Check catalogue identity and fixed references without filesystem I/O."""
        ids = [entry.id for entry in self.rails_configs]
        if any(not value or value != value.strip() for value in ids):
            raise ValueError("Rails config IDs must be non-empty and trimmed.")
        if len(ids) != len(set(ids)):
            raise ValueError("Rails config IDs must be unique.")
        for provider in self.providers:
            if provider.rails_config_id is not None and provider.rails_config_id not in ids:
                raise ValueError("A provider references an unknown Rails config ID.")
        return self


def resolve_provider(config: ProviderConfig) -> ProviderIntegration:
    """Resolve the entire selection to reviewed bindings, without fallback."""
    from nemoguardrails.server.experimental.providers.openai.chat_completions.endpoint import CHAT_COMPLETIONS_ENDPOINT
    from nemoguardrails.server.experimental.providers.openai.dialects.litellm.chat_completions import (
        LITELLM_CHAT_COMPLETIONS_ENDPOINT,
    )
    from nemoguardrails.server.experimental.providers.openai.dialects.litellm.responses import (
        LITELLM_RESPONSES_ENDPOINT,
    )
    from nemoguardrails.server.experimental.providers.openai.integration import create_openai_integration
    from nemoguardrails.server.experimental.providers.openai.responses.endpoint import RESPONSES_ENDPOINT

    bindings: dict[tuple[str, str | None], dict[str, GuardedJsonEndpoint]] = {
        ("openai", None): {
            "chat_completions": CHAT_COMPLETIONS_ENDPOINT,
            "responses": RESPONSES_ENDPOINT,
        },
        ("openai", "litellm"): {
            "chat_completions": LITELLM_CHAT_COMPLETIONS_ENDPOINT,
            "responses": LITELLM_RESPONSES_ENDPOINT,
        },
    }
    try:
        selected = bindings[(config.provider, config.dialect)]
        endpoints = tuple(selected[name] for name in config.operations)
    except KeyError:
        raise ValueError("Unsupported provider, operation, and dialect combination.") from None
    integration = create_openai_integration(
        instance_id=config.instance_id, base_url=config.base_url, mount_path=config.mount_path
    )
    return replace(integration, endpoints=endpoints)


def resolve_config_selection(config: ProxyConfig, resolver: ConfigResolver | None) -> ConfigResolver:
    """Adapt explicit fixed/dynamic selection to the existing runtime resolver.

    Fixed providers never call user code. Dynamic providers receive the original
    context and may select any registered ID; authorization belongs to that
    trusted callback. The existing runtime validates its result before dispatch.
    """
    if resolver is not None and not callable(resolver):
        raise TypeError("The config resolver must be callable.")
    fixed = {provider.instance_id: provider.rails_config_id for provider in config.providers}
    if resolver is None and any(value is None for value in fixed.values()):
        raise ValueError("Providers without a fixed Rails config ID require a resolver.")

    def select(context: ConfigContext) -> str | Awaitable[str]:
        config_id = fixed[context.provider_instance_id]
        if config_id is not None:
            return config_id
        if resolver is None:
            raise ValueError("No resolver is configured for this provider.")
        return resolver(context)

    return select
