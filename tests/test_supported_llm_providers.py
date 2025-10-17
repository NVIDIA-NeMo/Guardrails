# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemoguardrails.llm.providers import get_llm_provider_names


def test_get_llm_provider_names():
    supported_providers = [
        "ai21",
        "aleph_alpha",
        "anthropic",
        "anyscale",
        "azure",
        "bananadev",
        "cerebriumai",
        "cohere",
        "deepinfra",
        "forefrontai",
        "google_palm",
        "gooseai",
        "gpt4all",
        "huggingface_endpoint",
        "huggingface_hub",
        "huggingface_pipeline",
        "huggingface_textgen_inference",
        "human-input",
        "llamacpp",
        "modal",
        "nlpcloud",
        "openai",
        "petals",
        "pipelineai",
        "replicate",
        "rwkv",
        "sagemaker_endpoint",
        "self_hosted",
        "self_hosted_hugging_face",
        "stochasticai",
        "writer",
    ]

    provider_names = get_llm_provider_names()

    for provider_name in supported_providers:
        assert provider_name in provider_names
