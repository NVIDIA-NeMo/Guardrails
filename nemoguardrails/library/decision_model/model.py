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

"""System One adapter kept local until the shared decisions package is introduced."""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemoguardrails.http import HTTPClient
from nemoguardrails.library.decision_model.rail_config import NoulQuestion
from nemoguardrails.library.decision_model.request import decision_model_request

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import Model

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoulAnswer:
    probability: float
    model: str | None = None


class DecisionModelConfigurationError(ValueError):
    """Invalid model selection or missing credentials; never covered by fail_open."""


class SystemOneParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    base_url: str = Field(min_length=1)
    timeout: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    verify_ssl: bool = True

    @field_validator("base_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        try:
            parsed = urlsplit(value)
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
                and not any(char.isspace() for char in value)
            )
            parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("base_url must be an absolute http(s) URL without credentials, query, or fragment")
        return value


class SystemOneModel:
    """Answer one named Noul question per request, without choosing the rail verdict."""

    def __init__(self, config: "Model") -> None:
        if config.engine != "system_one" or config.type in {"main", "embeddings", "jailbreak_detection"}:
            raise DecisionModelConfigurationError("System One requires a dedicated decision model type.")
        if config.mode != "chat" or (config.cache and config.cache.enabled):
            raise DecisionModelConfigurationError("System One does not support text-completion mode or model caching.")
        self.parameters = SystemOneParameters.model_validate(config.parameters)
        self.model = config.model
        self.api_key_env_var = config.api_key_env_var
        if self.api_key_env_var is not None and not self.api_key_env_var.strip():
            raise DecisionModelConfigurationError("api_key_env_var must not be blank.")
        if self.api_key_env_var and urlsplit(self.parameters.base_url).scheme != "https":
            raise DecisionModelConfigurationError("base_url must use HTTPS when api_key_env_var is configured.")
        if not self.parameters.verify_ssl:
            log.warning("Decision model: TLS verification is disabled (verify_ssl is false).")

    async def decide(
        self, name: str, question: NoulQuestion, state: dict[str, str], *, http_client: HTTPClient | None = None
    ) -> NoulAnswer:
        api_key = None
        if self.api_key_env_var:
            api_key = os.environ.get(self.api_key_env_var)
            if not api_key:
                raise DecisionModelConfigurationError(
                    "The configured decision model API-key environment variable is unset."
                )
        assert self.model is not None
        assessment = await decision_model_request(
            state,
            {name: question.instructions},
            f"{self.parameters.base_url}/v1/systemone",
            self.model,
            api_key,
            http_client=http_client,
            timeout=self.parameters.timeout,
            verify_ssl=self.parameters.verify_ssl,
        )
        return NoulAnswer(probability=assessment.scores[name], model=assessment.model)
