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

import json

from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.rails.llm.config import Model


def initialize_llm(model_config: Model):
    """Initializes the model from LLM provider."""

    return init_llm_model(
        model_name=model_config.model,
        provider_name=model_config.engine,
        kwargs=model_config.parameters,
    )


def load_dataset(dataset_path: str):
    """Loads a dataset from a file."""

    with open(dataset_path, "r") as f:
        if dataset_path.endswith(".json"):
            dataset = json.load(f)
        else:
            dataset = f.readlines()

    return dataset
