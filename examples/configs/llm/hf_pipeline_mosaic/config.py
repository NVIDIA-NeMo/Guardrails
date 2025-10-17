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

from torch import bfloat16
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.providers.huggingface import HuggingFacePipelineCompatible


@lru_cache
def get_mpt_7b_instruct_llm():
    # For Mosaic MBT LLM, need to use from_pretrained instead of HuggingFacePipelineCompatible.from_model_id
    # in order to use the GPU. Default config uses CPU and cannot be modified.
    # Bug submitted here: https://github.com/huggingface/transformers/issues/24471#issuecomment-1606549042
    name = "mosaicml/mpt-7b-instruct"
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    # Use GPU (with id 0 in this case) for fast initialization
    device = "cuda:0"
    config.init_device = device
    config.max_seq_len = 450

    params = {"temperature": 0.01, "max_new_tokens": 100, "max_length": 450}

    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    pipe = pipeline(
        model=model,
        task="text-generation",
        tokenizer=tokenizer,
        device=device,
        do_sample=True,
        use_cache=True,
        **params,
    )

    llm = HuggingFacePipelineCompatible(pipeline=pipe, model_kwargs=params)

    return llm


HFPipelineMosaic = get_llm_instance_wrapper(
    llm_instance=get_mpt_7b_instruct_llm(), llm_type="hf_pipeline_mosaic"
)

register_llm_provider("hf_pipeline_mosaic", HFPipelineMosaic)
