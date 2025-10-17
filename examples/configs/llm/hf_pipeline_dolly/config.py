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

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from nemoguardrails.llm.helpers import get_llm_instance_wrapper
from nemoguardrails.llm.providers import register_llm_provider
from nemoguardrails.llm.providers.huggingface import HuggingFacePipelineCompatible


@lru_cache
def get_dolly_v2_3b_llm(streaming: bool = True):
    name = "databricks/dolly-v2-3b"

    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    device = "cpu"
    config.init_device = device
    config.max_seq_len = 45

    model = AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    params = {"temperature": 0.01, "max_new_tokens": 100}

    # If we want streaming, we create a streamer.
    if streaming:
        from nemoguardrails.llm.providers.huggingface import AsyncTextIteratorStreamer

        streamer = AsyncTextIteratorStreamer(tokenizer, skip_prompt=True)
        params["streamer"] = streamer

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


HFPipelineDolly = get_llm_instance_wrapper(
    llm_instance=get_dolly_v2_3b_llm(), llm_type="hf_pipeline_dolly"
)

register_llm_provider("hf_pipeline_dolly", HFPipelineDolly)
