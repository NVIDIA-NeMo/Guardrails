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
    ConfigSpecRef,
    RailConfigSchema,
    RailManifest,
    RailMetadata,
    RailSpec,
)

RAIL = RailManifest(
    name="factchecking",
    metadata=RailMetadata(
        display_name="Fact Checking",
        description="Provides shared configuration for fact-checking rails.",
        categories=("config", "output"),
        capabilities=("fact_check",),
        tags=("fact-checking", "rag", "grounding"),
        docs_url="docs/configure-rails/guardrail-catalog/fact-checking.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="fact_checking",
            spec=ConfigSpecRef(target="nemoguardrails.library.factchecking.rail_config:build_config_spec"),
        ),
    ),
)
