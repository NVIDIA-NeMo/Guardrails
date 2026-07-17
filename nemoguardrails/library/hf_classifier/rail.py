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
    TransformTarget,
)

HF_CLASSIFIER_CHECK_INPUT = ActionRef(
    name="hf_classifier_check_input",
    target="nemoguardrails.library.hf_classifier.actions:hf_classifier_check_input",
)
HF_CLASSIFIER_CHECK_OUTPUT = ActionRef(
    name="hf_classifier_check_output",
    target="nemoguardrails.library.hf_classifier.actions:hf_classifier_check_output",
)
HF_CLASSIFIER_CHECK_RETRIEVAL = ActionRef(
    name="hf_classifier_check_retrieval",
    target="nemoguardrails.library.hf_classifier.actions:hf_classifier_check_retrieval",
)

RAIL = RailManifest(
    name="hf_classifier",
    metadata=RailMetadata(
        display_name="Hugging Face Classifier",
        description="Checks input, output, or retrieved chunks with named Hugging Face classifier configurations.",
        categories=("input", "output", "retrieval"),
        capabilities=("allow", "block", "classify", "transform"),
        tags=("huggingface", "classifier", "local", "remote"),
    ),
    spec=RailSpec(
        config_schema=RailConfigSchema(
            key="hf_classifier",
            spec=ConfigSpecRef(target="nemoguardrails.library.hf_classifier.rail_config:build_config_spec"),
        ),
        flows=RailFlows(
            flow_names=(
                "hf classifier check input",
                "hf classifier check output",
                "hf classifier check retrieval",
            ),
        ),
        actions=RailActions(
            refs=(
                HF_CLASSIFIER_CHECK_INPUT,
                HF_CLASSIFIER_CHECK_OUTPUT,
                HF_CLASSIFIER_CHECK_RETRIEVAL,
            ),
        ),
        surfaces=(
            RailSurface(
                name="hf classifier check input",
                direction=RailDirection.INPUT,
                action=HF_CLASSIFIER_CHECK_INPUT,
                bindings=(Binding.surface_param("classifier", "classifier"),),
            ),
            RailSurface(
                name="hf classifier check output",
                direction=RailDirection.OUTPUT,
                action=HF_CLASSIFIER_CHECK_OUTPUT,
                bindings=(Binding.surface_param("classifier", "classifier"),),
            ),
            RailSurface(
                name="hf classifier check retrieval",
                direction=RailDirection.RETRIEVAL,
                action=HF_CLASSIFIER_CHECK_RETRIEVAL,
                bindings=(Binding.surface_param("classifier", "classifier"),),
                transform_target=TransformTarget.RELEVANT_CHUNKS,
            ),
        ),
        requirements=RailRequirements(
            services=(
                ServiceRequirement(name="vLLM classifier endpoint", required=False),
                ServiceRequirement(name="KServe classifier endpoint", required=False),
                ServiceRequirement(name="FMS guardrails-detectors endpoint", required=False),
            ),
            optional_dependencies=("transformers",),
        ),
        privacy=RailPrivacy(
            sends_user_text=True,
            sends_bot_text=True,
            sends_retrieved_chunks=True,
            remote_services=(
                "vLLM classifier endpoint",
                "KServe classifier endpoint",
                "FMS guardrails-detectors endpoint",
            ),
        ),
    ),
)
