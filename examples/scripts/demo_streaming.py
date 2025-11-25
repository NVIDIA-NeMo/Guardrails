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

"""Demo script."""

import asyncio
import logging
from typing import Optional

from langchain_core.language_models import BaseLLM
from langchain_core.runnables import RunnableConfig

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.context import streaming_handler_var
from nemoguardrails.streaming import StreamingHandler

logging.basicConfig(level=logging.INFO)

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4

streaming: True
"""


async def demo_1():
    """Demo using the streaming of response chunks directly."""
    config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
    app = LLMRails(config)

    history = [{"role": "user", "content": "What is the capital of France?"}]

    async for chunk in app.stream_async(messages=history):
        print(f"CHUNK: {chunk}")
        # Or do something else with the token


async def demo_2():
    """Demo of using the streaming of chunks with the final response as well."""
    config = RailsConfig.from_content(yaml_content=YAML_CONFIG)
    app = LLMRails(config)

    history = [{"role": "user", "content": "What is the capital of France?"}]

    streaming_handler = StreamingHandler()

    async def process_tokens():
        async for chunk in streaming_handler:
            print(f"CHUNK: {chunk}")
            # Or do something else with the token

    asyncio.create_task(process_tokens())

    result = await app.generate_async(messages=history, streaming_handler=streaming_handler)
    print(result)


async def demo_hf_pipeline():
    """Demo for streaming of response chunks directly with HuggingFacePipline deployed LLMs."""
    config = RailsConfig.from_path(config_path="./../configs/llm/hf_pipeline_dolly")
    app = LLMRails(config)

    history = [{"role": "user", "content": "What is the capital of France?"}]

    async for chunk in app.stream_async(messages=history):
        print(f"CHUNK: {chunk}")
        # Or do something else with the token


async def demo_streaming_from_custom_action():
    """Demo of using the streaming of chunks from custom actions."""
    config = RailsConfig.from_content(
        yaml_content="""
            models:
              - type: main
                engine: openai
                model: gpt-4

            # We're not interested in the user message canonical forms, since we
            # are only using a generic flow with `user ...`. So, we compute it purely
            # based on the embedding, without any additional LLM call.
            rails:
                dialog:
                    user_messages:
                        embeddings_only: True

            streaming: True
        """,
        colang_content="""
            # We need to have at least on canonical form to enable dialog rails.
            define user ask question
                "..."

            define flow
                user ...
                # Here we call the custom action which will
                $result = execute call_llm(user_query=$user_message)

                # In this case, we also return the result as the final message.
                # This is optional.
                bot $result
        """,
    )
    app = LLMRails(config, verbose=True)

    @action(is_system_action=True)
    async def call_llm(user_query: str, llm: Optional[BaseLLM]) -> str:
        call_config = RunnableConfig(callbacks=[streaming_handler_var.get()])
        response = await llm.ainvoke(user_query, config=call_config)
        return response.content

    app.register_action(call_llm)

    history = [{"role": "user", "content": "Write a short paragraph about France."}]

    streaming_handler = StreamingHandler()
    streaming_handler_var.set(streaming_handler)

    async def process_tokens():
        async for chunk in streaming_handler:
            print(f"CHUNK: {chunk}")
            # Or do something else with the token

    asyncio.create_task(process_tokens())

    result = await app.generate_async(messages=history, streaming_handler=streaming_handler)
    print(result)


if __name__ == "__main__":
    asyncio.run(demo_1())
    asyncio.run(demo_2())
    # asyncio.run(demo_hf_pipeline())
    asyncio.run(demo_streaming_from_custom_action())
