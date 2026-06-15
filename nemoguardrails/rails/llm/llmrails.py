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

"""LLM Rails entry point."""

import logging
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    overload,
)

from typing_extensions import Self

from nemoguardrails.base_guardrails import BaseGuardrails
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers import register_embedding_provider
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.checks import rails_check
from nemoguardrails.rails.llm.colang_turns.colang_turns import (
    generate_colang_events,
    process_colang_events,
    process_events_semaphore,
)
from nemoguardrails.rails.llm.config import OutputRailsStreamingConfig, RailsConfig
from nemoguardrails.rails.llm.embedding.embedding_search import EmbeddingSearchState
from nemoguardrails.rails.llm.generation.generation_context import (
    ensure_explain_info,
    explain_info_for_current_context,
    start_generation_request_context,
)
from nemoguardrails.rails.llm.generation.generation_request import (
    prepare_generation_request_for_runtime,
    validate_prompt_or_messages,
    validate_public_state,
)
from nemoguardrails.rails.llm.generation.generation_workflow import generate_standard_async
from nemoguardrails.rails.llm.generation.tracing import create_startup_tracing_adapters
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse, RailsResult, RailType
from nemoguardrails.rails.llm.startup.colang_runtime import runtime_for_colang_version
from nemoguardrails.rails.llm.startup.config_preparation import prepare_llmrails_config
from nemoguardrails.rails.llm.startup.config_py import load_config_py_modules, run_config_py_init_hooks
from nemoguardrails.rails.llm.startup.config_validation import validate_llmrails_config
from nemoguardrails.rails.llm.startup.embedding_config import apply_embedding_model_config
from nemoguardrails.rails.llm.startup.generation_actions import register_llm_generation_actions
from nemoguardrails.rails.llm.startup.knowledge_base import init_knowledge_base
from nemoguardrails.rails.llm.startup.llm_action_caches import initialize_llm_action_caches
from nemoguardrails.rails.llm.startup.llm_action_models import (
    load_llm_action_models,
    model_kwargs_from_config,
    sync_update_llm_bindings,
)
from nemoguardrails.rails.llm.streaming.generation_stream import (
    generation_token_stream,
    validate_streaming_with_output_rails,
)
from nemoguardrails.rails.llm.streaming.streaming_output_rails import run_output_rails_in_streaming
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.types import LLMModel
from nemoguardrails.utils import get_or_create_event_loop

log = logging.getLogger(__name__)


def _wrap_legacy_llm(llm):
    try:
        from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter
    except ImportError:
        raise TypeError(
            "Passing a raw LangChain LLM requires langchain to be installed. "
            "Either install langchain or pass an LLMModel instance."
        )
    warnings.warn(
        "Passing a raw LangChain LLM is deprecated. "
        "Use LangChainLLMAdapter(llm) explicitly or pass an LLMModel instance.",
        DeprecationWarning,
        stacklevel=3,
    )
    return LangChainLLMAdapter(llm)


class LLMRails(BaseGuardrails):
    """Rails based on a given configuration."""

    config: RailsConfig
    embedding_search: EmbeddingSearchState
    _explain_info: Optional[ExplainInfo]
    _kb: Any
    _log_adapters: Any
    _llm_generation_actions: Any
    verbose: bool
    events_history_cache: dict[str, list[dict]]
    llm: Optional[LLMModel]
    runtime: Runtime

    @property
    def kb(self):
        warnings.warn(
            "LLMRails.kb is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._kb

    @property
    def embedding_search_providers(self):
        warnings.warn(
            "LLMRails.embedding_search_providers is deprecated and will be removed in a future release. "
            "It is an internal attribute with no replacement read API; "
            "use register_embedding_search_provider() to add providers.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.embedding_search.providers

    @property
    def default_embedding_model(self):
        warnings.warn(
            "LLMRails.default_embedding_model is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.embedding_search.default_model

    @default_embedding_model.setter
    def default_embedding_model(self, value):
        warnings.warn(
            "Setting LLMRails.default_embedding_model is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_search.default_model = value

    @property
    def default_embedding_engine(self):
        warnings.warn(
            "LLMRails.default_embedding_engine is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.embedding_search.default_engine

    @default_embedding_engine.setter
    def default_embedding_engine(self, value):
        warnings.warn(
            "Setting LLMRails.default_embedding_engine is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_search.default_engine = value

    @property
    def default_embedding_params(self):
        warnings.warn(
            "LLMRails.default_embedding_params is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.embedding_search.default_params

    @default_embedding_params.setter
    def default_embedding_params(self, value):
        warnings.warn(
            "Setting LLMRails.default_embedding_params is deprecated and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_search.default_params = value

    @property
    def explain_info(self):
        warnings.warn(
            "LLMRails.explain_info is deprecated and will be removed in the next release. "
            "Use LLMRails.explain() instead, which guarantees a non-None ExplainInfo.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._explain_info

    @explain_info.setter
    def explain_info(self, value):
        warnings.warn(
            "Setting LLMRails.explain_info is deprecated and will be removed in the next release. "
            "explain_info is an internal accumulator; use LLMRails.explain() to read it.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._explain_info = value

    @property
    def llm_generation_actions(self):
        warnings.warn(
            "LLMRails.llm_generation_actions is deprecated and will be removed in a future release. "
            "It is an internal attribute; use the first-class LLMRails.passthrough_fn API if you "
            "previously set passthrough_fn through it.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._llm_generation_actions

    @property
    def passthrough_fn(self):
        return self._llm_generation_actions._passthrough_fn

    @passthrough_fn.setter
    def passthrough_fn(self, fn):
        self._llm_generation_actions._passthrough_fn = fn

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[LLMModel] = None,
        verbose: bool = False,
    ):
        """Initializes the LLMRails instance.

        Args:
            config: A rails configuration.
            llm: An optional LLM engine to use. If provided, this will be used as the main LLM
                and will take precedence over any main LLM specified in the config.
            verbose: Whether the logging should be verbose or not.
        """
        self.config = config
        if llm is not None and not isinstance(llm, LLMModel):
            self.llm = _wrap_legacy_llm(llm)
        else:
            self.llm = llm
        self.verbose = verbose

        if self.verbose:
            set_verbose(True, llm_calls=True)

        self.embedding_search = EmbeddingSearchState.default()

        # We keep a cache of the events history associated with a sequence of user messages.
        # TODO: when we update the interface to allow to return a "state object", this
        #   should be removed
        self.events_history_cache = {}

        self.config = prepare_llmrails_config(config=self.config)

        # We check if the configuration or any of the imported ones have config.py modules.
        config_modules = load_config_py_modules(self.config)

        # First, we initialize the runtime.
        self.runtime = runtime_for_colang_version(config=self.config, verbose=verbose)

        # If we have a config_modules with an `init` function, we call it.
        # We need to call this here because the `init` might register additional
        # LLM providers.
        run_config_py_init_hooks(self, config_modules)

        default_embedding_model, default_embedding_engine, default_embedding_params = apply_embedding_model_config(
            config=self.config,
            default_embedding_model=self.embedding_search.default_model,
            default_embedding_engine=self.embedding_search.default_engine,
            default_embedding_params=self.embedding_search.default_params,
        )
        self.embedding_search.update_defaults(
            default_model=default_embedding_model,
            default_engine=default_embedding_engine,
            default_params=default_embedding_params,
        )

        self._log_adapters = create_startup_tracing_adapters(self.config)

        # We run some additional checks on the config
        validate_llmrails_config(self.config)

        # Next, we initialize the LLM engines (main engine and action engines if specified).
        self._init_llms()

        # Next, we initialize the LLM Generate actions and register them.
        register_llm_generation_actions(self, verbose=verbose)

        # Next, we initialize the Knowledge Base.
        init_knowledge_base(self)

        # Reference to the general ExplainInfo object.
        self._explain_info = None

        from nemoguardrails.telemetry import report_usage

        report_usage(self.config, deployment_type="library", rails_engine="LLMRails")

    def update_llm(self, llm: LLMModel):
        """Replace the main LLM with the provided one.

        Arguments:
            llm: The new LLM that should be used.
        """
        if not isinstance(llm, LLMModel):
            llm = _wrap_legacy_llm(llm)
        sync_update_llm_bindings(self, llm)

    def _prepare_model_kwargs(self, model_config):
        """
        Prepare kwargs for model initialization, including API key from environment variable.

        Args:
            model_config: The model configuration object

        Returns:
            dict: The prepared kwargs for model initialization
        """
        return model_kwargs_from_config(model_config)

    def _init_llms(self):
        """
        Initializes the right LLM engines based on the configuration.
        There can be multiple LLM engines and types that can be specified in the config.
        The main LLM engine is the one that will be used for all the core guardrails generations.
        Other LLM engines can be specified for use in specific actions.

        The reason we provide an option for decoupling the main LLM engine from the action LLM
        is to allow for flexibility in using specialized LLM engines for specific actions.

        Raises:
            ModelInitializationError: If any model initialization fails
        """
        load_llm_action_models(self, init_llm=init_llm_model)
        initialize_llm_action_caches(self)

    @staticmethod
    def _ensure_explain_info() -> ExplainInfo:
        """Ensure that the ExplainInfo variable is present in the current context

        Returns:
            A ExplainInfo class containing the llm calls' statistics
        """
        return ensure_explain_info()

    def _get_embeddings_search_provider_instance(self, esp_config=None):
        return self.embedding_search.get_provider_instance(esp_config)

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        streaming_handler: Optional[StreamingHandler] = None,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate a completion or a next message.

        The format for messages is the following::

            [
                {"role": "context", "content": {"user_name": "John"}},
                {"role": "user", "content": "Hello! How are you?"},
                {"role": "assistant", "content": "I am fine, thank you!"},
                {"role": "event", "event": {"type": "UserSilent"}},
                ...
            ]

        Args:
            prompt: The prompt to be used for completion.
            messages: The history of messages to be used to generate the next message.
            options: Options specific for the generation.
            state: The state object that should be used as the starting point.
            streaming_handler: If specified, and the config supports streaming, the
              provided handler will be used for streaming.

        Returns:
            The completion (when a prompt is provided) or the next message.

        System messages are not yet supported."""
        validate_prompt_or_messages(prompt, messages)
        validate_public_state(self.config, state)
        prepared_request = prepare_generation_request_for_runtime(
            prompt=prompt,
            messages=messages,
            options=options,
            state=state,
        )
        prompt = prepared_request.prompt
        request_messages = prepared_request.request_messages
        messages = prepared_request.runtime_messages
        gen_options = prepared_request.options
        state = prepared_request.state

        request_context = start_generation_request_context(
            gen_options=gen_options,
            messages=request_messages,
            streaming_handler=streaming_handler,
        )
        try:
            if prepared_request.needs_llm and not self.llm:
                log.warning("No main LLM specified in the config and no LLM provided via constructor.")

            return await generate_standard_async(
                self,
                prompt=prompt,
                messages=messages,
                gen_options=gen_options,
                state=state,
                request_context=request_context,
            )
        finally:
            await request_context.close()

    @overload
    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_metadata: Literal[False] = False,
        generator: Optional[AsyncIterator[str]] = None,
        include_generation_metadata: Optional[bool] = None,
    ) -> AsyncIterator[str]: ...

    @overload
    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_metadata: Literal[True] = ...,
        generator: Optional[AsyncIterator[str]] = None,
        include_generation_metadata: Optional[bool] = None,
    ) -> AsyncIterator[Union[str, dict]]: ...

    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        include_metadata: Optional[bool] = False,
        generator: Optional[AsyncIterator[str]] = None,
        include_generation_metadata: Optional[bool] = None,
    ) -> AsyncIterator[Union[str, dict]]:
        """Simplified interface for getting directly the streamed tokens from the LLM."""
        validate_streaming_with_output_rails(self.config)
        validate_public_state(self.config, state)
        return generation_token_stream(
            self,
            prompt=prompt,
            messages=messages,
            options=options,
            state=state,
            include_metadata=include_metadata,
            generator=generator,
            include_generation_metadata=include_generation_metadata,
        )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[dict] = None,
    ):
        """Synchronous version of generate_async."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate` inside async code. "
                "You should replace with `await generate_async(...)` or use `nest_asyncio.apply()`."
            )

        loop = get_or_create_event_loop()

        return loop.run_until_complete(
            self.generate_async(
                prompt=prompt,
                messages=messages,
                options=options,
                state=state,
            )
        )

    async def generate_events_async(
        self,
        events: List[dict],
    ) -> List[dict]:
        """Generate the next events based on the provided history.

        The format for events is the following::

            [
                {"type": "...", ...},
                ...
            ]

        Args:
            events: The history of events to be used to generate the next events.
            options: The options to be used for the generation.

        Returns:
            The newly generate event(s).

        """
        return await generate_colang_events(self, events)

    def generate_events(
        self,
        events: List[dict],
    ) -> List[dict]:
        """Synchronous version of `LLMRails.generate_events_async`."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)` or use `nest_asyncio.apply()`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.generate_events_async(events=events))

    async def process_events_async(
        self,
        events: List[dict],
        state: Union[Optional[dict], State] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], Union[dict, State]]:
        """Process a sequence of events in a given state.

        The events will be processed one by one, in the input order.

        Args:
            events: A sequence of events that needs to be processed.
            state: The state that should be used as the starting point. If not provided,
              a clean state will be used.

        Returns:
            (output_events, output_state) Returns a sequence of output events and an output
              state.
        """
        return await process_colang_events(
            self,
            events,
            state,
            blocking,
            semaphore=process_events_semaphore,
        )

    def process_events(
        self,
        events: List[dict],
        state: Union[Optional[dict], State] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], Union[dict, State]]:
        """Synchronous version of `LLMRails.process_events_async`."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.process_events_async(events, state, blocking))

    async def check_async(
        self,
        messages: List[dict],
        rail_types: Optional[List[RailType]] = None,
    ) -> RailsResult:
        """Run rails on messages based on their content (asynchronous).

        When ``rail_types`` is not provided, automatically determines which rails
        to run based on message roles:
        - Only user messages: runs input rails
        - Only assistant messages: runs output rails
        - Both user and assistant messages: runs both input and output rails
        - No user/assistant messages: logs warning and returns passing result

        When ``rail_types`` is provided, runs exactly the specified rail types,
        skipping the auto-detection logic.

        Args:
            messages: List of message dicts with 'role' and 'content' fields.
                     Messages can contain any roles, but only user/assistant roles
                     determine which rails execute when ``rail_types`` is not provided.
            rail_types: Optional list of rail types to run, e.g.
                  ``[RailType.INPUT]`` or ``[RailType.OUTPUT]``.
                  When provided, overrides automatic detection.

        Returns:
            RailsResult containing:
            - status: PASSED, MODIFIED, or BLOCKED
            - content: The final content after rails processing
            - rail: Name of the rail that blocked (if blocked)

        Examples:
            Check user input (auto-detected)::

                result = await rails.check_async([{"role": "user", "content": "Hello!"}])
                if result.status == RailStatus.BLOCKED:
                    print(f"Blocked by: {result.rail}")

            Check bot output with context (auto-detected)::

                result = await rails.check_async([
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"}
                ])

            Run only input rails explicitly::

                result = await rails.check_async(messages, rail_types=[RailType.INPUT])
        """
        return await rails_check.check_messages(self, messages, rail_types=rail_types)

    def check(
        self,
        messages: List[dict],
        rail_types: Optional[List[RailType]] = None,
    ) -> RailsResult:
        """Run rails on messages based on their content (synchronous).

        This is a synchronous wrapper around check_async().

        Args:
            messages: List of message dicts with 'role' and 'content' fields.
            rail_types: Optional list of rail types to run. See check_async() for details.

        Returns:
            RailsResult containing status, content, and optional blocking rail name.
        """
        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `check` inside async code. You should replace with `await check_async(...)`."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(self.check_async(messages, rail_types=rail_types))

    def register_action(self, action: Callable, name: Optional[str] = None) -> Self:
        """Register a custom action for the rails configuration.

        This mutates the runtime action registry and is intended to be called
        during application startup, before concurrent generation requests begin.
        """
        self.runtime.register_action(action, name)
        return self

    def register_action_param(self, name: str, value: Any) -> Self:
        """Register a custom action parameter.

        This mutates runtime action dependencies and is intended to be called
        during application startup, before concurrent generation requests begin.
        """
        self.runtime.register_action_param(name, value)
        return self

    def register_filter(self, filter_fn: Callable, name: Optional[str] = None) -> Self:
        """Register a custom filter for the rails configuration.

        This mutates the runtime task manager and is intended to be called
        during application startup, before concurrent generation requests begin.
        """
        self.runtime.llm_task_manager.register_filter(filter_fn, name)
        return self

    def register_output_parser(self, output_parser: Callable, name: str) -> Self:
        """Register a custom output parser for the rails configuration.

        This mutates the runtime task manager and is intended to be called
        during application startup, before concurrent generation requests begin.
        """
        self.runtime.llm_task_manager.register_output_parser(output_parser, name)
        return self

    def register_prompt_context(self, name: str, value_or_fn: Any) -> Self:
        """Register a value to be included in the prompt context.

        :name: The name of the variable or function that will be used.
        :value_or_fn: The value or function that will be used to generate the value.

        This mutates the runtime task manager and is intended to be called
        during application startup, before concurrent generation requests begin.
        """
        self.runtime.llm_task_manager.register_prompt_context(name, value_or_fn)
        return self

    def register_embedding_search_provider(self, name: str, cls: Type[EmbeddingsIndex]) -> Self:
        """Register a new embedding search provider.

        Args:
            name: The name of the embedding search provider that will be used.
            cls: The class that will be used to generate and search embedding

        This updates the instance provider registry. Register providers during
        startup so knowledge-base and generation-action setup can see them.
        """

        self.embedding_search.register_provider(name, cls)
        return self

    def register_embedding_provider(self, cls: Type[EmbeddingModel], name: Optional[str] = None) -> Self:
        """Register a custom embedding provider.

        Args:
            model (Type[EmbeddingModel]): The embedding model class.
            name (str): The name of the embedding engine. If available in the model, it will be used.

        Raises:
            ValueError: If the engine name is not provided and the model does not have an engine name.
            ValueError: If the model does not have 'encode' or 'encode_async' methods.

        This updates the process-global embedding provider registry and is
        intended to be called during application startup.
        """
        register_embedding_provider(engine_name=name, model=cls)
        return self

    def explain(self) -> ExplainInfo:
        """Helper function to return the latest ExplainInfo object."""
        self._explain_info = explain_info_for_current_context(self._explain_info)
        return self._explain_info

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        if state["config"].config_path:
            config = RailsConfig.from_path(state["config"].config_path)
        else:
            config = state["config"]
        self.__init__(config=config, verbose=False)

    async def _run_output_rails_in_streaming(
        self,
        streaming_handler: AsyncIterator[str],
        output_rails_streaming_config: OutputRailsStreamingConfig,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        stream_first: Optional[bool] = None,
    ) -> AsyncIterator[str]:
        """
        1. Buffers tokens from 'streaming_handler' via BufferStrategy.
        2. Runs sequential (parallel for colang 2.0 in future) flows for each chunk.
        3. Yields the chunk if not blocked, or STOP if blocked.
        """
        async for chunk in run_output_rails_in_streaming(
            self,
            streaming_handler=streaming_handler,
            output_rails_streaming_config=output_rails_streaming_config,
            prompt=prompt,
            messages=messages,
            stream_first=stream_first,
        ):
            yield chunk
