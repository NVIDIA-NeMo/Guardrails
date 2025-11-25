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
import logging
import uuid
from time import time
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
    BaseCallbackManager,
)
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from nemoguardrails.context import explain_info_var, llm_call_info_var, llm_stats_var
from nemoguardrails.logging.explain import LLMCallInfo
from nemoguardrails.logging.processing_log import processing_log_var
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.utils import new_uuid

log = logging.getLogger(__name__)


class LoggingCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running."""
        # We initialize a new LLM call if we don't have one already. This can happen
        # when a chain is used directly.
        llm_call_info = llm_call_info_var.get()
        if llm_call_info is None:
            llm_call_info = LLMCallInfo()
            llm_call_info_var.set(llm_call_info)

        llm_call_info.id = new_uuid()

        # We also add it to the explain object
        explain_info = explain_info_var.get()
        if explain_info:
            explain_info.llm_calls.append(llm_call_info)

        log.info("Invocation Params :: %s", kwargs.get("invocation_params", {}))
        log.info(
            "Prompt :: %s",
            prompts[0],
            extra={"id": llm_call_info.id, "task": llm_call_info.task},
        )
        llm_call_info.prompt = prompts[0]

        llm_call_info.started_at = time()

        llm_stats = llm_stats_var.get()
        if llm_stats is None:
            llm_stats = LLMStats()
            llm_stats_var.set(llm_stats)

        llm_stats.inc("total_calls")

    async def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running."""
        # We initialize a new LLM call if we don't have one already. This can happen
        # when a chain is used directly.
        llm_call_info = llm_call_info_var.get()
        if llm_call_info is None:
            llm_call_info = LLMCallInfo()
            llm_call_info_var.set(llm_call_info)

        llm_call_info.id = new_uuid()

        # We also add it to the explain object
        explain_info = explain_info_var.get()
        if explain_info:
            explain_info.llm_calls.append(llm_call_info)

        type_map = {
            "human": "User",
            "ai": "Bot",
            "tool": "Tool",
            "system": "System",
            "developer": "Developer",
        }
        prompt = "\n" + "\n".join(
            [
                "[cyan]"
                + type_map.get(msg.type, msg.type.title())
                + "[/]"
                + "\n"
                + (msg.content if isinstance(msg.content, str) else "")
                for msg in messages[0]
            ]
        )

        log.info("Invocation Params :: %s", kwargs.get("invocation_params", {}))
        log.info(
            "Prompt Messages :: %s",
            prompt,
            extra={"id": llm_call_info.id, "task": llm_call_info.task},
        )
        llm_call_info.prompt = prompt
        llm_call_info.started_at = time()

        llm_stats = llm_stats_var.get()
        if llm_stats is None:
            llm_stats = LLMStats()
            llm_stats_var.set(llm_stats)

        llm_stats.inc("total_calls")

    async def on_llm(self, *args, **kwargs) -> Any:
        """NOTE: this needs to be implemented to avoid a warning by LangChain."""
        pass

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running."""
        llm_call_info = llm_call_info_var.get()
        if llm_call_info is None:
            llm_call_info = LLMCallInfo()
        llm_call_info.completion = response.generations[0][0].text
        llm_call_info.finished_at = time()

        completion_text = response.generations[0][0].text
        reasoning_content = None

        if isinstance(response.generations[0][0], ChatGeneration):
            chat_gen = response.generations[0][0]
            if hasattr(chat_gen, "message") and hasattr(chat_gen.message, "additional_kwargs"):
                reasoning_content = chat_gen.message.additional_kwargs.get("reasoning_content")

        if reasoning_content:
            full_completion = f"{reasoning_content}\n---\n{completion_text}"
        else:
            full_completion = completion_text

        log.info(
            "Completion :: %s",
            full_completion,
            extra={"id": llm_call_info.id, "task": llm_call_info.task},
        )

        llm_stats = llm_stats_var.get()
        if llm_stats is None:
            llm_stats = LLMStats()
            llm_stats_var.set(llm_stats)

        # If there are additional completions, we show them as well
        if len(response.generations[0]) > 1:
            for i, generation in enumerate(response.generations[0][1:]):
                log.info("--- :: Completion %d", i + 2)
                log.info(
                    "Completion :: %s",
                    generation.text,
                    extra={"id": llm_call_info.id, "task": llm_call_info.task},
                )

        log.info("Output Stats :: %s", response.llm_output)
        if llm_call_info.finished_at is not None and llm_call_info.started_at is not None:
            took = llm_call_info.finished_at - llm_call_info.started_at
            log.info("--- :: LLM call took %.2f seconds", took)
            llm_stats.inc("total_time", took)
            llm_call_info.duration = took
        else:
            log.warning("LLM call timing information incomplete")
            llm_call_info.duration = 0.0

        # Update the token usage stats as well
        token_stats_found = False
        if response.generations:
            # For chat models completions (most models) token stats should be accessed from
            # the standardized fields present in the AIMessage messages from response.generations.

            # Initialize LLM call info stats
            if not llm_call_info.total_tokens:
                llm_call_info.total_tokens = 0
            if not llm_call_info.prompt_tokens:
                llm_call_info.prompt_tokens = 0
            if not llm_call_info.completion_tokens:
                llm_call_info.completion_tokens = 0

            # Compute stats over all LLM generations in the response object
            for gen_list in response.generations:
                for gen in gen_list:
                    if (
                        isinstance(gen, ChatGeneration)
                        and isinstance(gen.message, AIMessage)
                        and gen.message.usage_metadata
                    ):
                        token_stats_found = True
                        token_usage = gen.message.usage_metadata
                        llm_stats.inc("total_tokens", token_usage.get("total_tokens", 0))
                        llm_call_info.total_tokens += token_usage.get("total_tokens", 0)
                        llm_stats.inc("total_prompt_tokens", token_usage.get("input_tokens", 0))
                        llm_call_info.prompt_tokens += token_usage.get("input_tokens", 0)
                        llm_stats.inc(
                            "total_completion_tokens",
                            token_usage.get("output_tokens", 0),
                        )
                        llm_call_info.completion_tokens += token_usage.get("output_tokens", 0)
        if not token_stats_found and response.llm_output:
            # Fail-back mechanism for non-chat models. This works for OpenAI models,
            # but it may not work for others as response.llm_output is not standardized.
            token_usage = response.llm_output.get("token_usage", {})
            if len(token_usage.items()) > 0:
                token_stats_found = True
            llm_stats.inc("total_tokens", token_usage.get("total_tokens", 0))
            llm_call_info.total_tokens = token_usage.get("total_tokens", 0)
            llm_stats.inc("total_prompt_tokens", token_usage.get("prompt_tokens", 0))
            llm_call_info.prompt_tokens = token_usage.get("prompt_tokens", 0)
            llm_stats.inc("total_completion_tokens", token_usage.get("completion_tokens", 0))
            llm_call_info.completion_tokens = token_usage.get("completion_tokens", 0)

        if not token_stats_found:
            log.info("Token stats in LLM call info cannot be computed for current model!")

        # Finally, we append the LLM call log to the processing log
        processing_log = processing_log_var.get()
        if processing_log:
            processing_log.append({"type": "llm_call_info", "timestamp": time(), "data": llm_call_info})

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors."""

    async def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain starts running."""

    async def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain ends running."""

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors."""

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""

    async def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool ends running."""

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on arbitrary text."""

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent end."""


handlers = [LoggingCallbackHandler()]
logging_callbacks = BaseCallbackManager(
    handlers=cast(List[BaseCallbackHandler], handlers),
    inheritable_handlers=cast(List[BaseCallbackHandler], handlers),
)

logging_callback_manager_for_chain = AsyncCallbackManagerForChainRun(
    run_id=uuid.uuid4(),
    parent_run_id=None,
    handlers=cast(List[BaseCallbackHandler], handlers),
    inheritable_handlers=cast(List[BaseCallbackHandler], handlers),
    tags=[],
    inheritable_tags=[],
)
