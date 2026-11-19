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

"""The hooks server's agent-start policy: whether an agent may start, from plain facts about its parts and models."""

from dataclasses import dataclass
from typing import Optional

from nemoguardrails.guardrails.guardrails_types import quoted_identity
from nemoguardrails.rails.llm.config import AgentStartConfig
from nemoguardrails.server.hooks.schemas import AgentPart, AgentStartRequest

_RULES = "hooks_server.agent_start"


@dataclass(frozen=True)
class AgentStartPolicy:
    """`hooks_server.agent_start`, read once when the server starts. It calls no model."""

    allowed_sources: Optional[tuple[str, ...]] = None
    allow_local_sources: bool = True
    require_digests: bool = False
    allowed_models: Optional[frozenset[str]] = None
    require_policy_enforcers: bool = False

    @classmethod
    def from_config(cls, agent_start: AgentStartConfig) -> "AgentStartPolicy":
        """The policy a config's hooks_server.agent_start section sets."""
        return cls(
            allowed_sources=None if agent_start.allowed_sources is None else tuple(agent_start.allowed_sources),
            allow_local_sources=agent_start.allow_local_sources,
            require_digests=agent_start.require_digests,
            allowed_models=None if agent_start.allowed_models is None else frozenset(agent_start.allowed_models),
            require_policy_enforcers=agent_start.require_policy_enforcers,
        )

    def decide(self, request: AgentStartRequest) -> Optional[str]:
        """Why the policy denies the start, naming the first part, model or policy a rule denies; None to allow."""
        for part in request.parts:
            reason = self._part_reason(part)
            if reason is not None:
                return reason
        if self.allowed_models is not None:
            for model in request.models:
                if model not in self.allowed_models:
                    return f"model '{quoted_identity(model)}' is not in {_RULES}.allowed_models"
        if self.require_policy_enforcers:
            present = set(request.enforcers_present)
            for policy in request.policies:
                if not policy.required:
                    continue
                if not policy.enforcers:
                    return (
                        f"policy '{quoted_identity(policy.id)}' names no enforcer, and "
                        f"{_RULES}.require_policy_enforcers is true"
                    )
                missing = next((enforcer for enforcer in policy.enforcers if enforcer not in present), None)
                if missing is not None:
                    return (
                        f"policy '{quoted_identity(policy.id)}' needs enforcer '{quoted_identity(missing)}', which "
                        f"is not present, and {_RULES}.require_policy_enforcers is true"
                    )
        return None

    def _part_reason(self, part: AgentPart) -> Optional[str]:
        """Why a rule denies *part*, or None."""
        name = f"{part.type} '{quoted_identity(part.id)}'"
        if part.local:
            if not self.allow_local_sources:
                return f"{name} comes from a local folder, and {_RULES}.allow_local_sources is false"
        elif self.allowed_sources is not None and not part.source.startswith(self.allowed_sources):
            # The source is the caller's text, so the reason names the part, not the source.
            return f"{name} comes from a source {_RULES}.allowed_sources doesn't list"
        if self.require_digests and not part.digest:
            return f"{name} has no digest, and {_RULES}.require_digests is true"
        return None
