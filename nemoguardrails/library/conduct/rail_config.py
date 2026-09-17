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

"""Config schema for the Conduct guard rail."""

from typing import Literal, Optional

from nemoguardrails.manifests.config_schema import (
    Field,
    RailConfigBaseModel,
    RailConfigSpec,
    rail_field,
)


class ConductRailConfig(RailConfigBaseModel):
    """Configuration for the Conduct guard rail.

    Every field falls through to an environment variable if unset, so
    the config block can be as short as::

        rails:
          conduct: {}

    when ``CONDUCT_AGENT_TOKEN`` is exported.
    """

    api_url: str = Field(
        default="https://api.conductai.ai",
        description="Base URL for the Conduct API. Override for self-hosted deployments.",
    )

    workspace_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional workspace pin. When the agent_token is provisioned "
            "for more than one workspace, set this to disambiguate. Usually "
            "the token already owns its workspace."
        ),
    )

    fail_mode: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "What happens if Conduct is unreachable. fail_closed blocks the "
            "request; fail_open allows it. Matches the plugin default."
        ),
    )

    timeout: float = Field(
        default=8.0,
        description="Seconds to wait for a Conduct decision before applying fail_mode.",
    )


def build_config_spec() -> RailConfigSpec:
    """Referenced by :class:`~nemoguardrails.library.conduct.rail.RAIL`."""
    return RailConfigSpec(model=ConductRailConfig)
