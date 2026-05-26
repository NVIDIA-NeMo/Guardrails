# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Top-level Guardrails interface module.

This module provides a simplified, user-friendly interface for interacting with
NeMo Guardrails. The Guardrails class wraps either IORails or LLMRails (chosen
automatically based on config) and provides a streamlined API for generating
LLM responses with programmable guardrails.
"""

import logging
import warnings
from typing import Any, AsyncIterator, Callable, List, Optional, Tuple, Type, Union, cast, overload

from typing_extensions import Self

from nemoguardrails.base_guardrails import BaseGuardrails
from nemoguardrails.colang.runtime import Runtime
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.guardrails import configure_logging
from nemoguardrails.guardrails.guardrails_types import LLMMessages
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import GenerationResponse, RailsResult, RailType
from nemoguardrails.types import LLMModel

log = logging.getLogger(__name__)


class Guardrails(BaseGuardrails):
    """Top-level interface for NeMo Guardrails functionality."""

    config: RailsConfig
    verbose: bool
    use_iorails_engine: bool

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[LLMModel] = None,
        verbose: bool = False,
        *,
        use_iorails: bool = True,  # False -> fall back to LLMRails instead
        require_iorails: bool = False,
    ):
        """Initialize a Guardrails instance.

        When ``use_iorails`` is True, the wrapper attempts to use the IORails engine.
        If the config or arguments are incompatible (an ``llm`` is provided, or the
        config contains flows IORails does not support), the wrapper falls back to
        LLMRails and logs a warning. Set ``require_iorails=True`` to raise a
        ``ValueError`` instead — use this when IORails-only features such as
        OpenTelemetry metrics are required.
        """

        self.config = config
        self.verbose = verbose

        if verbose:
            configure_logging(logging.DEBUG)
        else:
            configure_logging(logging.INFO)

        if use_iorails:
            fallback_reason = IORails.unsupported_reason(config, llm)
            if fallback_reason is None:
                self._rails_engine = IORails(config)
                self.use_iorails_engine = True
            else:
                message = (
                    f"use_iorails=True was requested but IORails cannot be used: {fallback_reason}. "
                    "Falling back to LLMRails; IORails-only features (such as OpenTelemetry "
                    "metrics) will not be available."
                )
                if require_iorails:
                    raise ValueError(message)
                log.warning(message)
                self._rails_engine = LLMRails(config, llm, verbose)
                self.use_iorails_engine = False
        else:
            self._rails_engine = LLMRails(config, llm, verbose)
            self.use_iorails_engine = False

        # Track whether startup() has been called (supports lazy initialization)
        self._started = False

    @property
    def rails_engine(self) -> IORails | LLMRails:
        """Get immutable LLMRails object"""
        return self._rails_engine

    @property
    def llm(self) -> Optional[LLMModel]:
        """The main LLM in use. Only supported for LLMRails.
        Read-only; use ``update_llm()`` to replace it.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support llm attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.llm

    @property
    def runtime(self) -> Runtime:
        """The Colang runtime backing the rails engine. Only supported for LLMRails."""
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support runtime attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.runtime

    @property
    def explain_info(self) -> Optional[ExplainInfo]:
        """Deprecated. Use ``explain()`` instead.

        Direct access can return ``None`` for an uninitialized accumulator;
        ``explain()`` guarantees a non-None ExplainInfo. Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support explain_info attribute access")

        warnings.warn(
            "Guardrails.explain_info is deprecated and will be removed in the next release. "
            "Use Guardrails.explain() instead, which guarantees a valid ExplainInfo.",
            DeprecationWarning,
            stacklevel=2,
        )
        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails._explain_info

    @explain_info.setter
    def explain_info(self, value: Optional[ExplainInfo]) -> None:
        """Deprecated. Setting ``explain_info`` is no longer supported; use ``explain()`` to read it."""
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support explain_info attribute access")

        warnings.warn(
            "Setting Guardrails.explain_info is deprecated and will be removed in the next release. "
            "explain_info is an internal accumulator; use Guardrails.explain() to read it.",
            DeprecationWarning,
            stacklevel=2,
        )
        llmrails = cast(LLMRails, self.rails_engine)
        llmrails._explain_info = value

    @property
    def passthrough_fn(self) -> Optional[Callable]:
        """The optional passthrough function that bypasses LLM generation.

        Only supported for LLMRails. When set, the rails pipeline calls this
        function instead of the main LLM for generating responses.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support passthrough_fn attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.passthrough_fn

    @passthrough_fn.setter
    def passthrough_fn(self, fn: Optional[Callable]) -> None:
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support passthrough_fn attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.passthrough_fn = fn

    @staticmethod
    def _convert_to_messages(prompt: str | None = None, messages: LLMMessages | None = None) -> LLMMessages:
        """Return messages in standard format, converting a prompt string if needed.

        If messages is provided, returns it as-is.
        If prompt is provided, wraps it as [{"role": "user", "content": prompt}].
        """

        # Priority: messages first, then prompt
        if messages:
            return messages

        if prompt:
            # Convert string prompt to standard format
            return [{"role": "user", "content": prompt}]

        raise ValueError("Neither prompt nor messages provided for generation")

    async def _ensure_started(self) -> None:
        """Lazy initialization: call startup() on first use if not already started."""
        if not self._started:
            await self.startup()

    def generate(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate an LLM response synchronously with guardrails applied.
        Supported in both IORails and LLMRails
        """

        generate_messages = self._convert_to_messages(prompt, messages)
        return self.rails_engine.generate(messages=generate_messages, **kwargs)

    @overload
    async def generate_async(self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs) -> str: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> dict: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> GenerationResponse: ...

    @overload
    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> tuple[dict, dict]: ...

    async def generate_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> str | dict | GenerationResponse | tuple[dict, dict]:
        """Generate an LLM response asynchronously with guardrails applied.
        Supported by both LLMRails and IORails
        """
        await self._ensure_started()

        generate_messages = self._convert_to_messages(prompt, messages)
        return await self.rails_engine.generate_async(messages=generate_messages, **kwargs)

    def stream_async(
        self, prompt: str | None = None, messages: LLMMessages | None = None, **kwargs
    ) -> AsyncIterator[str | dict]:
        """Generate an LLM response asynchronously with streaming support."""

        stream_messages = self._convert_to_messages(prompt, messages)

        async def _with_startup(iterator: AsyncIterator[str | dict]) -> AsyncIterator[str | dict]:
            await self._ensure_started()
            async for chunk in iterator:
                yield chunk

        if isinstance(self.rails_engine, IORails):
            # IORails.stream_async() only accepts messages, options, include_metadata
            unsupported = set(kwargs) - {"options", "include_metadata"}
            if unsupported:
                log.warning("IORails stream_async: ignoring unsupported kwargs: %s", unsupported)
            return _with_startup(
                self.rails_engine.stream_async(
                    messages=stream_messages,
                    options=kwargs.get("options"),
                    include_metadata=kwargs.get("include_metadata", False),
                )
            )

        llmrails = cast(LLMRails, self.rails_engine)
        return _with_startup(llmrails.stream_async(messages=stream_messages, **kwargs))

    def explain(self) -> ExplainInfo:
        """Get the latest ExplainInfo object for debugging.
        Only supported for LLMRails
        """

        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support explain()")

        # self.rails_engine must be LLMRails since we raise above if we're using IORails
        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.explain()

    def update_llm(self, llm: LLMModel) -> None:
        """Replace the main LLM with a new one.
        Only supported for LLMRails, since IORails doesn't take LLM as argument
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support update_llm()")

        # self.rails_engine must be LLMRails since we raise above if we're using IORails
        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.update_llm(llm)

    @property
    def events_history_cache(self) -> dict:
        """Per-session events history cache. Only supported for LLMRails.

        Used by the server to persist conversation state across requests.
        Stored by reference; assigning replaces the dict object, not its
        contents.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support events_history_cache attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.events_history_cache

    @events_history_cache.setter
    def events_history_cache(self, value: dict) -> None:
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support events_history_cache attribute access")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.events_history_cache = value

    async def generate_events_async(self, events: List[dict]) -> List[dict]:
        """Generate the next events based on the provided history.
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support generate_events_async()")

        llmrails = cast(LLMRails, self.rails_engine)
        return await llmrails.generate_events_async(events)

    def generate_events(self, events: List[dict]) -> List[dict]:
        """Synchronous version of generate_events_async.
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support generate_events()")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.generate_events(events)

    async def process_events_async(
        self,
        events: List[dict],
        state: Union[Optional[dict], State] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], Union[dict, State]]:
        """Process a sequence of events in a given state.
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support process_events_async()")

        llmrails = cast(LLMRails, self.rails_engine)
        return await llmrails.process_events_async(events, state, blocking)

    def process_events(
        self,
        events: List[dict],
        state: Union[Optional[dict], State] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], Union[dict, State]]:
        """Synchronous version of process_events_async.
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support process_events()")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.process_events(events, state, blocking)

    async def check_async(
        self,
        messages: List[dict],
        rail_types: Optional[List[RailType]] = None,
    ) -> RailsResult:
        """Run rails on messages based on their content (asynchronous).
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support check_async()")

        llmrails = cast(LLMRails, self.rails_engine)
        return await llmrails.check_async(messages, rail_types=rail_types)

    def check(
        self,
        messages: List[dict],
        rail_types: Optional[List[RailType]] = None,
    ) -> RailsResult:
        """Synchronous version of check_async.
        Only supported for LLMRails.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support check()")

        llmrails = cast(LLMRails, self.rails_engine)
        return llmrails.check(messages, rail_types=rail_types)

    def register_action(self, action: Callable, name: Optional[str] = None) -> Self:
        """Register a custom action for the rails configuration.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_action()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_action(action, name)
        return self

    def register_action_param(self, name: str, value: Any) -> Self:
        """Register a custom action parameter.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_action_param()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_action_param(name, value)
        return self

    def register_filter(self, filter_fn: Callable, name: Optional[str] = None) -> Self:
        """Register a custom filter for the rails configuration.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_filter()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_filter(filter_fn, name)
        return self

    def register_output_parser(self, output_parser: Callable, name: str) -> Self:
        """Register a custom output parser for the rails configuration.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_output_parser()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_output_parser(output_parser, name)
        return self

    def register_prompt_context(self, name: str, value_or_fn: Any) -> Self:
        """Register a value to be included in the prompt context.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_prompt_context()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_prompt_context(name, value_or_fn)
        return self

    def register_embedding_search_provider(self, name: str, cls: Type[EmbeddingsIndex]) -> Self:
        """Register a new embedding search provider.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_embedding_search_provider()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_embedding_search_provider(name, cls)
        return self

    def register_embedding_provider(self, cls: Type[EmbeddingModel], name: Optional[str] = None) -> Self:
        """Register a custom embedding provider.
        Only supported for LLMRails. Returns self so calls can be chained.
        """
        if isinstance(self.rails_engine, IORails):
            raise NotImplementedError("IORails doesn't support register_embedding_provider()")

        llmrails = cast(LLMRails, self.rails_engine)
        llmrails.register_embedding_provider(cls, name)
        return self

    def __getstate__(self):
        """Pickle support: preserve config, verbose, and use_iorails so the rebuilt
        instance lands on the same engine. The llm is dropped (matches LLMRails).
        """
        return {"config": self.config, "verbose": self.verbose, "use_iorails": self.use_iorails_engine}

    def __setstate__(self, state):
        """Pickle support: rebuild from config + verbose + use_iorails. Older
        pickles missing these keys default to False/True respectively for
        backwards compatibility.
        """
        if state["config"].config_path:
            config = RailsConfig.from_path(state["config"].config_path)
        else:
            config = state["config"]
        self.__init__(
            config=config,
            verbose=state.get("verbose", False),
            use_iorails=state.get("use_iorails", True),
        )

    async def startup(self) -> None:
        """Lifecycle method to start the rails engine.

        Idempotent: safe to call multiple times.  Also called automatically
        on first ``generate_async()`` if not called explicitly, so callers
        are not required to manage the lifecycle.

        The non-streaming admission queue is owned by ``IORails`` and is
        started/stopped as part of ``IORails.start()`` / ``stop()``.
        """
        if self._started:
            return
        if isinstance(self.rails_engine, IORails):
            await self.rails_engine.start()
        self._started = True

    async def shutdown(self) -> None:
        """Lifecycle method to stop the rails engine.

        Idempotent: safe to call multiple times.
        """
        if not self._started:
            return
        if isinstance(self.rails_engine, IORails):
            await self.rails_engine.stop()
        self._started = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
