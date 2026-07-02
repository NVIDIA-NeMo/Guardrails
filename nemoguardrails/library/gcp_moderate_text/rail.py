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

from nemoguardrails.manifests import (
    ActionRef,
    EnvVar,
    RailActions,
    RailDirection,
    RailFlows,
    RailManifest,
    RailMetadata,
    RailPrivacy,
    RailRequirements,
    RailSpec,
    RailSurface,
    ServiceRequirement,
)

CALL_GCP_TEXT_MODERATION_API = ActionRef(
    name="call gcpnlp api",
    target="nemoguardrails.library.gcp_moderate_text.actions:call_gcp_text_moderation_api",
)

RAIL = RailManifest(
    name="gcp_moderate_text",
    metadata=RailMetadata(
        display_name="GCP Text Moderation",
        description="Moderates user input with Google Cloud Natural Language.",
        categories=("input",),
        capabilities=("allow", "block", "classify", "content_safety", "moderate"),
        tags=("third-party", "gcp", "moderation"),
        docs_url="docs/configure-rails/guardrail-catalog/community/gcp-text-moderations.md",
    ),
    spec=RailSpec(
        flows=RailFlows(flow_names=("gcpnlp moderation", "gcpnlp moderation detailed")),
        actions=RailActions(refs=(CALL_GCP_TEXT_MODERATION_API,)),
        surfaces=(
            RailSurface(name="gcpnlp moderation", direction=RailDirection.INPUT, action=CALL_GCP_TEXT_MODERATION_API),
            RailSurface(
                name="gcpnlp moderation detailed", direction=RailDirection.INPUT, action=CALL_GCP_TEXT_MODERATION_API
            ),
        ),
        requirements=RailRequirements(
            env_vars=(EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS", required=False),),
            services=(ServiceRequirement(name="Google Cloud Natural Language API", required=True),),
            optional_dependencies=("google-cloud-language",),
        ),
        privacy=RailPrivacy(sends_user_text=True, remote_services=("Google Cloud Natural Language API",)),
    ),
)
