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

"""Configuration for the decision model rail."""

import logging
import warnings
from typing import Dict, Optional

from pydantic import ConfigDict, field_validator, model_validator

from nemoguardrails.manifests.config_schema import Field, RailConfigBaseModel, RailConfigSpec, rail_field

log = logging.getLogger(__name__)


class DecisionModelQuestion(RailConfigBaseModel):
    """A single Noul (yes/no) question asked of the decision model about the text."""

    model_config = ConfigDict(extra="forbid")

    instructions: str = Field(description="The yes/no question to ask the decision model about the text.")
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-question block threshold in [0, 1]. Falls back to the rail-level threshold when unset.",
    )


class DecisionModelOptions(RailConfigBaseModel):
    """Per-direction (input / output) options for the decision model."""

    model_config = ConfigDict(extra="forbid")

    questions: Dict[str, DecisionModelQuestion] = Field(
        default_factory=dict,
        description=(
            "Noul questions to ask about the text, keyed by an identifier. "
            "The text is blocked if any answer probability reaches its threshold."
        ),
    )


class DecisionModelConfig(RailConfigBaseModel):
    """Configuration for the decision model rail."""

    model_config = ConfigDict(extra="forbid")

    server_endpoint: str = Field(min_length=1, description="The decision model assessment endpoint URL.")
    model: str = Field(min_length=1, description="The decision model identifier. Pin a version for reproducibility.")
    api_key_env_var: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "Environment variable containing the API key, sent as a bearer token. "
            "Optional: when unset, requests are sent without an Authorization header. "
            "Requires an https:// server_endpoint when set."
        ),
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default probability at or above which a question blocks the text.",
    )
    timeout: float = Field(default=10.0, gt=0, description="Per-request timeout in seconds.")
    verify_ssl: bool = Field(
        default=True,
        description=(
            "Verify the endpoint's TLS certificate. Set to false only for self-signed or "
            "internal certificates you trust."
        ),
    )
    fail_open: bool = Field(
        default=False,
        description="If true, allow the text when the decision model cannot be reached. Default is to fail closed and block.",
    )
    input: DecisionModelOptions = Field(
        default_factory=DecisionModelOptions,
        description="Questions asked about the user input.",
    )
    output: DecisionModelOptions = Field(
        default_factory=DecisionModelOptions,
        description="Questions asked about the bot output.",
    )

    @field_validator("server_endpoint", "model", "api_key_env_var")
    @classmethod
    def _check_nonblank(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def _check_server_endpoint(self) -> "DecisionModelConfig":
        """Require an http(s) URL, and https when an API key is configured.

        Plain HTTP is allowed without a key so in-cluster and local endpoints work; a bearer
        token is never sent over it.
        """
        self.server_endpoint = self.server_endpoint.strip()
        if not self.server_endpoint.lower().startswith(("http://", "https://")):
            raise ValueError(f"server_endpoint must start with 'http://' or 'https://', got '{self.server_endpoint}'")
        if self.server_endpoint.split("://", 1)[1][:1] in ("", "/", "?", "#"):
            raise ValueError(f"server_endpoint must include a host, got '{self.server_endpoint}'")
        if self.api_key_env_var and not self.server_endpoint.lower().startswith("https://"):
            raise ValueError("server_endpoint must use HTTPS when api_key_env_var is configured")
        if not self.verify_ssl:
            log.warning("Decision model: TLS verification is disabled (verify_ssl is false).")
        return self

    @model_validator(mode="after")
    def _warn_disabled_screening(self) -> "DecisionModelConfig":
        for direction in ("input", "output"):
            if not getattr(self, direction).questions:
                warnings.warn(
                    f"Decision model {direction} screening is disabled because rails.config.decision_model.{direction}.questions is empty. "
                    f"All {direction} text will be allowed even when fail_open is false.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[DecisionModelConfig],
        field_info=rail_field(
            default=None,
            description="Configuration for the decision model rail.",
        ),
        exports={
            "DecisionModelConfig": DecisionModelConfig,
            "DecisionModelOptions": DecisionModelOptions,
            "DecisionModelQuestion": DecisionModelQuestion,
        },
    )
