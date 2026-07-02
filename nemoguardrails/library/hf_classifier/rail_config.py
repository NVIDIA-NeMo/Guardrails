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

import logging
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from nemoguardrails.manifests.config_schema import (
    Discriminator,
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    model_validator,
    rail_field,
)

log = logging.getLogger(__name__)


class _HFClassifierBase(RailConfigBaseModel):
    """Shared fields for all HuggingFace classifier engines."""

    model: str = Field(
        min_length=1,
        description="HF model ID, local path, or server-side model identifier.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score for a detection to trigger blocking.",
    )
    blocked_labels: List[str] = Field(
        default_factory=list,
        description="Labels that should trigger blocking when detected above threshold.",
    )

    @model_validator(mode="after")
    def _validate_common(self) -> "_HFClassifierBase":
        if not self.blocked_labels:
            log.warning(
                "HFClassifierConfig '%s': blocked_labels is empty — this classifier will never block anything.",
                self.model,
            )
        return self


class LocalHFClassifierConfig(_HFClassifierBase):
    """Configuration for a local HuggingFace Transformers pipeline classifier."""

    engine: Literal["local"] = "local"
    task: Literal["text-classification", "token-classification"] = Field(
        default="text-classification",
        description="HuggingFace pipeline task type.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Forwarded as kwargs to transformers.pipeline() "
        "(e.g. device, dtype, trust_remote_code, token, revision, "
        "aggregation_strategy).",
    )

    @model_validator(mode="after")
    def _validate_local(self) -> "LocalHFClassifierConfig":
        agg = self.parameters.get("aggregation_strategy")
        if agg and self.task != "token-classification":
            raise ValueError("aggregation_strategy is only valid when task is 'token-classification'.")
        return self


class RemoteHFClassifierConfig(_HFClassifierBase):
    """Configuration for a remote HuggingFace classifier (vLLM, KServe, FMS)."""

    engine: Literal["vllm", "kserve", "fms"]
    base_url: str = Field(
        description="Base URL for the inference server (e.g. 'http://host:8000').",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Environment variable name holding the API key. "
        "Resolved at runtime to an Authorization: Bearer header.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Remote backend parameters: "
        "'timeout' (float, seconds), 'verify_ssl' (bool), "
        "'ca_cert'/'client_cert'/'client_key' (str, paths). "
        "Note: 'ca_cert' replaces (not extends) system CAs; use a "
        "concatenated bundle to include both custom and system CAs.",
    )

    _KNOWN_PARAMS: frozenset = frozenset({"timeout", "verify_ssl", "ca_cert", "client_cert", "client_key"})

    @model_validator(mode="after")
    def _validate_remote(self) -> "RemoteHFClassifierConfig":
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must start with 'http://' or 'https://', got '{self.base_url}'")
        self.base_url = self.base_url.rstrip("/")
        unknown = set(self.parameters) - self._KNOWN_PARAMS
        if unknown:
            log.warning(
                "HFClassifierConfig '%s': unknown parameters ignored: %s. Supported: %s",
                self.model,
                sorted(unknown),
                sorted(self._KNOWN_PARAMS),
            )
        if self.parameters.get("verify_ssl") is False:
            log.warning(
                "HFClassifierConfig '%s': TLS verification is disabled.",
                self.model,
            )
        return self


HFClassifierConfig = Annotated[
    Union[LocalHFClassifierConfig, RemoteHFClassifierConfig],
    Discriminator("engine"),
]


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[Dict[str, HFClassifierConfig]],
        field_info=rail_field(
            default=None,
            description="Named HF classifier configurations. Keys are classifier names referenced by flows.",
        ),
        exports={
            "HFClassifierConfig": HFClassifierConfig,
            "LocalHFClassifierConfig": LocalHFClassifierConfig,
            "RemoteHFClassifierConfig": RemoteHFClassifierConfig,
        },
    )
