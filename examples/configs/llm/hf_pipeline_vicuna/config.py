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
from functools import lru_cache

from torch import float16
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.providers.huggingface import HuggingFacePipelineCompatible


@lru_cache
def get_vicuna_7b_llm():
    """Loads the Vicuna 7B LLM."""
    repo_id = "lmsys/vicuna-7b-v1.3"
    params = {"temperature": 0, "max_length": 1024}

    # Using the first GPU
    device = 0

    llm = HuggingFacePipelineCompatible.from_model_id(
        model_id=repo_id,
        device=device,
        task="text-generation",
        model_kwargs=params,
    )

    return llm


def get_vicuna_13b_llm():
    """Loads the Vicuna 13B LLM."""
    repo_id = "lmsys/vicuna-13b-v1.3"
    # If you want Bloke Wizard Vicuna, comment one of the next lines
    # repo_id = "TheBloke/wizard-vicuna-13B-HF"
    # repo_id = "TheBloke/Wizard-Vicuna-13B-Uncensored-HF"
    params = {"temperature": 0, "max_length": 1024}

    # Using the first GPU
    device = 0

    llm = HuggingFacePipelineCompatible.from_model_id(
        model_id=repo_id,
        device=device,
        task="text-generation",
        model_kwargs=params,
    )

    return llm


def _load_model(model_name, device, num_gpus, debug=False):
    """Helper function to load the model."""
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": float16}
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
        kwargs = {"torch_dtype": float16}
        # Avoid bugs in mps backend by not using in-place operations.
        print("mps not supported")
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)

    if device == "cuda" and num_gpus == 1:
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


def get_vicuna_13b_llm_from_path(model_path: str = "/workspace/ckpt/"):
    """Loads the Vicuna 13B LLM from a local path."""
    device = "cuda"
    num_gpus = 2  # making sure GPU-GPU are NVlinked, GPUs-GPUS with NVSwitch
    model, tokenizer = _load_model(model_path, device, num_gpus, debug=False)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0,
        do_sample=True,
    )

    llm = HuggingFacePipelineCompatible(pipeline=pipe)
    return llm


# On the next line, change the Vicuna LLM instance depending on your needs
HFPipelineVicuna = get_llm_instance_wrapper(llm_instance=get_vicuna_7b_llm(), llm_type="hf_pipeline_vicuna")

register_llm_provider("hf_pipeline_vicuna", HFPipelineVicuna)
