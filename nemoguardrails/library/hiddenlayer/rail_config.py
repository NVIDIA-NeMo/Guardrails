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

import re
from typing import Optional

from pydantic import ConfigDict, field_validator

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)

DEFAULT_API_URL = "https://api.us.hiddenlayer.ai"
DEFAULT_REQUESTER_ID = "hiddenlayer-nemo"

_API_URL_PATTERN = re.compile(
    r"(?:https://[^/?#\s]+|http://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?)(?:[/?#]\S*)?",
    re.IGNORECASE,
)


def _check_api_url(api_url: str) -> str:
    """Return *api_url* if it is https, or http on a loopback host.

    The token request carries the client secret and the evaluation carries the bearer token, so
    a mistyped ``http://`` URL must fail rather than send them in cleartext.
    """
    if _API_URL_PATTERN.fullmatch(api_url):
        return api_url
    # The URL itself is left out of the message: it can carry userinfo.
    raise ValueError(
        "HiddenLayer api_url must be an https:// URL, or an http:// URL on a loopback host "
        "(127.0.0.1, ::1 or localhost)."
    )


class HiddenLayerRailConfig(RailConfigBaseModel):
    """Configuration for the HiddenLayer AI Runtime Security integration.

    The client ID and secret are intentionally not part of this config. They are secrets and
    must be provided via the HIDDENLAYER_CLIENT_ID and HIDDENLAYER_CLIENT_SECRET environment
    variables.
    """

    model_config = ConfigDict(extra="forbid")

    api_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the HiddenLayer API serving the NeMo Guardrails request-evaluation and "
            "response-evaluation routes. Falls back to the HIDDENLAYER_BASE_URL environment "
            f"variable, then to {DEFAULT_API_URL}. Must use https://, except on a loopback host."
        ),
    )
    project_id: Optional[str] = Field(
        default=None,
        description=(
            "HiddenLayer project ID or alias whose policy evaluates the interactions. Falls back to "
            "the HIDDENLAYER_PROJECT_ID environment variable. When unset, or when the ID is unknown, "
            "HiddenLayer uses the tenant's default project."
        ),
    )
    requester_id: Optional[str] = Field(
        default=None,
        description=(
            "Requester ID recorded with each evaluation. Falls back to the HIDDENLAYER_REQUESTER_ID "
            f"environment variable, then to {DEFAULT_REQUESTER_ID!r}."
        ),
    )
    fail_open: bool = Field(
        default=False,
        description=(
            "If True, allow content through when the HiddenLayer API cannot be reached or does not "
            "answer with a verdict. It covers transport failures only: an evaluation HiddenLayer "
            "itself could not complete comes back as a verdict and is always applied. This changes "
            "the security posture of the rail and should be reviewed as part of the guardrails config."
        ),
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Number of additional attempts after the HiddenLayer API returns HTTP 429, 502, 503 or "
            "504. Total attempts per request equal max_retries + 1."
        ),
    )
    max_retry_after_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description=(
            "Upper bound (in seconds) on how long to honor a Retry-After header, and on the exponential backoff delay."
        ),
    )
    retry_backoff_seconds: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Base delay (in seconds) for exponential backoff with jitter when a retryable response "
            "has no usable Retry-After header."
        ),
    )

    send_session_id: Optional[bool] = Field(
        default=None,
        description=(
            "If True, send the conversation ID of each request, when the caller set one with "
            "nemoguardrails.context.conversation_id(), as the HiddenLayer session ID. It is the only "
            "source of conversation history: without it HiddenLayer evaluates each check on its own. "
            "The ID is recorded with the interactions, so do not use one that carries personal data. "
            "Falls back to the HIDDENLAYER_SEND_SESSION_ID environment variable (0, false, no or off "
            "disables it), then to True."
        ),
    )

    @field_validator("api_url")
    @classmethod
    def _validate_api_url(cls, value: Optional[str]) -> Optional[str]:
        if value:
            _check_api_url(value)
        return value


def build_config_spec() -> RailConfigSpec:
    return RailConfigSpec(
        annotation=Optional[HiddenLayerRailConfig],
        field_info=rail_field(
            default_factory=HiddenLayerRailConfig,
            description="Configuration for HiddenLayer AI Runtime Security.",
        ),
        exports={"HiddenLayerRailConfig": HiddenLayerRailConfig},
    )
