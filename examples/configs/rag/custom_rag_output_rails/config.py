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

from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase

TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""


async def rag(context: dict, llm: BaseLLM, kb: KnowledgeBase) -> ActionResult:
    user_message = context.get("last_user_message")
    context_updates = {}

    # For our custom RAG, we re-use the built-in retrieval
    chunks = await kb.search_relevant_chunks(user_message)
    relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
    # 💡 Store the chunks for fact-checking
    context_updates["relevant_chunks"] = relevant_chunks

    # Use a custom prompt template
    prompt_template = PromptTemplate.from_template(TEMPLATE)
    input_variables = {"question": user_message, "context": relevant_chunks}
    # 💡 Store the template for hallucination-checking
    context_updates["_last_bot_prompt"] = prompt_template.format(**input_variables)

    print(f"💬 RAG :: prompt_template: {context_updates['_last_bot_prompt']}")

    # Put together a simple LangChain chain
    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    answer = await chain.ainvoke(input_variables)

    return ActionResult(return_value=answer, context_updates=context_updates)


def init(app: LLMRails):
    app.register_action(rag, "rag")
