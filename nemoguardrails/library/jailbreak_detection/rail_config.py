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
import os
from typing import Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    SecretStr,
    model_validator,
    rail_field,
)

log = logging.getLogger(__name__)


class JailbreakDetectionConfig(RailConfigBaseModel):
    """Configuration data for jailbreak detection."""

    server_endpoint: Optional[str] = Field(
        default=None,
        description="The endpoint for the jailbreak detection heuristics/model container.",
    )
    length_per_perplexity_threshold: float = Field(default=89.79, gt=0, description="The length/perplexity threshold.")
    prefix_suffix_perplexity_threshold: float = Field(
        default=1845.65, gt=0, description="The prefix/suffix perplexity threshold."
    )
    nim_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for jailbreak detection model. Example: http://localhost:8000/v1",
    )
    nim_server_endpoint: Optional[str] = Field(
        default="classify",
        description="Classification path uri. Defaults to 'classify' for NemoGuard JailbreakDetect.",
    )
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Secret String with API key for use in Jailbreak requests. Takes precedence over api_key_env_var",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Environment variable containing API key for jailbreak detection model",
    )
    nim_url: Optional[str] = Field(
        default=None,
        deprecated="Use 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Use nim_base_url instead",
    )
    nim_port: Optional[int] = Field(
        default=None,
        deprecated="Include port in 'nim_base_url' instead. This field will be removed in a future version.",
        description="DEPRECATED: Include port in nim_base_url instead",
    )
    embedding: Optional[str] = Field(
        default=None,
        deprecated="This field is no longer used.",
    )

    @model_validator(mode="after")
    def migrate_deprecated_fields(self) -> "JailbreakDetectionConfig":
        """Migrate deprecated nim_url/nim_port fields to nim_base_url format."""
        if self.nim_url and not self.nim_base_url:
            port = self.nim_port or 8000
            self.nim_base_url = f"http://{self.nim_url}:{port}/v1"
        return self

    @model_validator(mode="after")
    def validate_urls(self) -> "JailbreakDetectionConfig":
        """Validate URL formats for endpoints."""
        if self.nim_base_url and not self.nim_base_url.startswith(("http://", "https://")):
            raise ValueError(f"nim_base_url must start with 'http://' or 'https://', got '{self.nim_base_url}'")
        if self.server_endpoint and not self.server_endpoint.startswith(("http://", "https://")):
            raise ValueError(f"server_endpoint must start with 'http://' or 'https://', got '{self.server_endpoint}'")
        return self

    def get_api_key(self) -> Optional[str]:
        """Helper to return an API key (if it exists) from a Jailbreak configuration.
          This can come from (in descending order of priority):

        1. The `api_key` field, a Pydantic SecretStr from which we extract the full string.
        2. The `api_key_env_var` field, a string stored in this environment variable.

        If neither is found, None is returned.
        """

        if self.api_key:
            return self.api_key.get_secret_value()

        if self.api_key_env_var:
            nim_auth_token = os.getenv(self.api_key_env_var)
            if nim_auth_token:
                return nim_auth_token

            log.warning(
                "Specified a value for jailbreak config api_key_env var at %s but the environment variable was not set!"
                % self.api_key_env_var
            )

        return None


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[JailbreakDetectionConfig],
        field_info=rail_field(
            default_factory=JailbreakDetectionConfig,
            description="Configuration for jailbreak detection.",
        ),
        exports={"JailbreakDetectionConfig": JailbreakDetectionConfig},
    )
