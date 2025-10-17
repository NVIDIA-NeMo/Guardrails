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
import os
import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.providers.huggingface import HuggingFacePipelineCompatible


def _get_model_config(config: RailsConfig, type: str):
    """Quick helper to return the config for a specific model type."""
    for model_config in config.models:
        if model_config.type == type:
            return model_config


def _load_model(model_name_or_path, device, num_gpus, hf_auth_token=None, debug=False):
    """Load an HF locally saved checkpoint."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {i: "13GiB" for i in range(num_gpus)},
                    }
                )
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    if hf_auth_token is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_auth_token=hf_auth_token, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            use_auth_token=hf_auth_token,
            **kwargs,
        )

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def init_main_llm(config: RailsConfig):
    """Initialize the main model from a locally saved path.

    The path is taken from the main model config.

    models:
      - type: main
        engine: hf_pipeline_bloke
        parameters:
          path: "<PATH TO THE LOCALLY SAVED CHECKPOINT>"
    """
    # loading custom llm  from disk with multiGPUs support
    # model_name = "< path_to_the_saved_custom_llm_checkpoints >"  # loading model ckpt from disk
    model_config = _get_model_config(config, "main")
    model_path = model_config.parameters.get("path")
    device = model_config.parameters.get("device", "cuda")
    num_gpus = model_config.parameters.get("num_gpus", 1)
    hf_token = os.environ[
        "HF_TOKEN"
    ]  # [TODO] to register this into the config.yaml as well
    model, tokenizer = _load_model(
        model_path, device, num_gpus, hf_auth_token=hf_token, debug=False
    )

    # repo_id="TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    # pipe = pipeline("text-generation", model=repo_id, device_map={"":"cuda:0"}, max_new_tokens=256, temperature=0.1, do_sample=True,use_cache=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        do_sample=True,
    )

    hf_llm = HuggingFacePipelineCompatible(pipeline=pipe)
    provider = get_llm_instance_wrapper(
        llm_instance=hf_llm, llm_type="hf_pipeline_llama2_13b"
    )
    register_llm_provider("hf_pipeline_llama2_13b", provider)


def init(llm_rails: LLMRails):
    config = llm_rails.config

    # Initialize the various models
    init_main_llm(config)
