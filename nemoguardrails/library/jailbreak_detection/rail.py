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
    Binding,
    ConfigSpecRef,
    EnvVar,
    RailActions,
    RailConfigSchema,
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

JAILBREAK_DETECTION_HEURISTICS = ActionRef(
    name="jailbreak_detection_heuristics",
    target="nemoguardrails.library.jailbreak_detection.actions:jailbreak_detection_heuristics",
)
JAILBREAK_DETECTION_MODEL = ActionRef(
    name="jailbreak_detection_model",
    target="nemoguardrails.library.jailbreak_detection.actions:jailbreak_detection_model",
)
RAIL = RailManifest(
    name="jailbreak_detection",
    metadata=RailMetadata(
        display_name="Jailbreak Detection",
        description="Detects jailbreak attempts using heuristic, model-based, or remote classifier checks.",
        categories=("input",),
        capabilities=("allow", "block", "classify", "detect_jailbreak"),
        tags=("nemoguard", "security", "jailbreak"),
        docs_url="docs/configure-rails/guardrail-catalog/jailbreak-protection.mdx",
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="jailbreak_detection",
            spec=ConfigSpecRef(target="nemoguardrails.library.jailbreak_detection.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=("jailbreak detection heuristics", "jailbreak detection model"),
        ),
        actions=RailActions(
            refs=(
                JAILBREAK_DETECTION_HEURISTICS,
                JAILBREAK_DETECTION_MODEL,
            ),
        ),
        surfaces=(
            RailSurface(
                name="jailbreak detection heuristics",
                direction=RailDirection.INPUT,
                action=JAILBREAK_DETECTION_HEURISTICS,
                bindings=(Binding.context("user_message", "user_message"),),
            ),
            RailSurface(
                name="jailbreak detection model",
                direction=RailDirection.INPUT,
                action=JAILBREAK_DETECTION_MODEL,
                bindings=(Binding.context("user_message", "user_message"),),
            ),
        ),
        requirements=RailRequirements(
            env_vars=(
                EnvVar(
                    name="NVIDIA_API_KEY",
                    required=False,
                    description="API key used when the shipped hosted NVIDIA NIM configuration selects this variable.",
                ),
                EnvVar(
                    name="HF_TOKEN",
                    required=False,
                    description="Optional Hugging Face Hub token used when downloading local jailbreak models.",
                ),
                EnvVar(
                    name="HF_HOME",
                    required=False,
                    description="Optional Hugging Face cache directory used by local jailbreak model downloads.",
                ),
                EnvVar(
                    name="HF_HUB_OFFLINE",
                    required=False,
                    description="Optional Hugging Face Hub offline-mode setting used when loading local jailbreak models.",
                ),
                EnvVar(
                    name="JAILBREAK_CHECK_DEVICE",
                    required=False,
                    description="Optional device override for local jailbreak detection models, such as cpu or cuda.",
                ),
                EnvVar(
                    name="EMBEDDING_CLASSIFIER_PATH",
                    required=False,
                    description="Directory containing or receiving the local jailbreak classifier model.",
                ),
            ),
            services=(ServiceRequirement(name="NVIDIA NIM", required=False),),
            # `transformers` required by both in-process paths
            optional_dependencies=("torch", "transformers"),
        ),
        privacy=RailPrivacy(sends_user_text=True, remote_services=("NVIDIA NIM",)),
    ),
)
