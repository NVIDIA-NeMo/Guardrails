# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from typing import Optional

from nemoguardrails.actions.actions import ActionResult, action
from nemoguardrails.kb.kb import KnowledgeBase

log = logging.getLogger(__name__)


@action(is_system_action=True)
async def retrieve_relevant_chunks(
    context: Optional[dict] = None,
    kb: Optional[KnowledgeBase] = None,
    is_colang_2: Optional[bool] = False,
):
    """Retrieve relevant knowledge chunks and update the context.

    Args:
        context (Optional[dict]): The context for the execution of the action. Defaults to None.
        kb (Optional[KnowledgeBase]): The KnowledgeBase to search for relevant chunks. Defaults to None.

    Returns:
        ActionResult: An action result containing the retrieved relevant chunks with context updates:
            - "relevant_chunks" -- the relevant chunks as a single string,
            - "relevant_chunks_sep" -- the relevant chunks as a list of strings before concatenation,
            - "retrieved_for" -- the user message that the chunks were retrieved for.

    Note:
        This action retrieves relevant chunks from the KnowledgeBase based on the user's last message
        and updates the context with the information.

    Example:
        ```
        result = await retrieve_relevant_chunks(context=my_context, kb=my_knowledge_base)
        print(result.return_value)  # Relevant chunks as a string
        print(result.context_updates)  # Updated context with relevant chunks
        ```
    """

    user_message: Optional[str] = context.get("last_user_message") if context else None
    context_updates = {}

    if user_message and kb:
        # Are these needed two needed?
        context_updates["relevant_chunks"] = ""
        context_updates["relevant_chunks_sep"] = []

        context_updates["retrieved_for"] = user_message

        chunks = [
            chunk["body"] for chunk in await kb.search_relevant_chunks(user_message)
        ]

        context_updates["relevant_chunks"] = "\n".join(chunks)
        context_updates["relevant_chunks_sep"] = chunks

    else:
        # No KB is set up, we keep the existing relevant_chunks if we have them.
        if is_colang_2:
            context_updates["relevant_chunks"] = (
                context.get("relevant_chunks", "") if context else None
            )
            if context_updates["relevant_chunks"]:
                context_updates["relevant_chunks"] += "\n"
        else:
            context_updates["relevant_chunks"] = (
                (context.get("relevant_chunks", "") + "\n") if context else None
            )
        context_updates["relevant_chunks_sep"] = (
            context.get("relevant_chunks_sep", []) if context else None
        )
        context_updates["retrieved_for"] = None

    return ActionResult(
        return_value=context_updates["relevant_chunks"],
        context_updates=context_updates,
    )
