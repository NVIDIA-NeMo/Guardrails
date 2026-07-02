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

import asyncio
import json
import logging
import re
import time
import warnings
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

from typing_extensions import Self

from nemoguardrails.actions.llm.utils import (
    extract_bot_thinking_from_events,
    extract_tool_calls_from_events,
    get_and_clear_response_metadata_contextvar,
    get_colang_history,
)
from nemoguardrails.actions.output_mapping import is_output_blocked
from nemoguardrails.base_guardrails import BaseGuardrails
from nemoguardrails.colang.v1_0.runtime.flows import compute_context
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime
from nemoguardrails.colang.v2_x.runtime.flows import State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.context import (
    explain_info_var,
    generation_options_var,
    llm_stats_var,
    raw_llm_request,
    streaming_handler_var,
)
from nemoguardrails.embeddings.index import EmbeddingsIndex
from nemoguardrails.embeddings.providers import register_embedding_provider
from nemoguardrails.embeddings.providers.base import EmbeddingModel
from nemoguardrails.exceptions import (
    InvalidStateError,
    StreamingNotSupportedError,
)
from nemoguardrails.llm.models.initializer import init_llm_model
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.buffer import get_buffer_strategy
from nemoguardrails.rails.llm.config import (
    OutputRailsStreamingConfig,
    RailsConfig,
)
from nemoguardrails.rails.llm.conversation.conversation_events import events_for_messages
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
    RailsResult,
    RailStatus,
    RailType,
)
from nemoguardrails.rails.llm.runtime.colang_runtime import runtime_for_colang_version
from nemoguardrails.rails.llm.runtime.colang_turns import (
    generate_colang_events,
    process_colang_events,
    process_events_semaphore,
)
from nemoguardrails.rails.llm.startup.config_preparation import prepare_llmrails_config
from nemoguardrails.rails.llm.startup.config_py import load_config_py_modules, run_config_py_init_hooks
from nemoguardrails.rails.llm.startup.config_validation import validate_llmrails_config
from nemoguardrails.rails.llm.startup.embedding_search import EmbeddingSearchState, apply_embedding_model_config
from nemoguardrails.rails.llm.startup.generation_actions import register_llm_generation_actions
from nemoguardrails.rails.llm.startup.knowledge_base import init_knowledge_base
from nemoguardrails.rails.llm.startup.llm_action_caches import initialize_llm_action_caches
from nemoguardrails.rails.llm.startup.llm_action_models import (
    load_llm_action_models,
    model_kwargs_from_config,
    sync_update_llm_bindings,
)
from nemoguardrails.rails.llm.startup.tracing import create_startup_tracing_adapters
from nemoguardrails.rails.llm.utils import (
    get_action_details_from_flow_id,
    get_history_cache_key,
)
from nemoguardrails.streaming import END_OF_STREAM, StreamingHandler
from nemoguardrails.types import LLMModel
from nemoguardrails.utils import (
    extract_error_json,
    get_or_create_event_loop,
)

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
    _llm_generation_actions: Any
    _verbose: bool
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
        """The optional passthrough function that bypasses LLM generation.

        When set, the rails pipeline calls this function instead of the main LLM
        for generating responses. LLMGenerationActions is private, expose only
        `passthrough_fn` as a public API
        """
        return self._llm_generation_actions._passthrough_fn

    @passthrough_fn.setter
    def passthrough_fn(self, fn):
        """LLMGenerationActions is private, set passthrough_fn directly"""
        self._llm_generation_actions._passthrough_fn = fn

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: bool) -> None:
        self._verbose = verbose

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

    def _get_embeddings_search_provider_instance(self, esp_config=None):
        return self.embedding_search.get_provider_instance(esp_config)

    def _get_events_for_messages(self, messages: List[dict], state: Any):
        return events_for_messages(self, messages, state)

    @staticmethod
    def _ensure_explain_info() -> ExplainInfo:
        """Ensure that the ExplainInfo variable is present in the current context

        Returns:
            A ExplainInfo class containing the llm calls' statistics
        """
        explain_info = explain_info_var.get()
        if explain_info is None:
            explain_info = ExplainInfo()
            explain_info_var.set(explain_info)

        return explain_info

    def _validate_public_state(self, state: Optional[Union[dict, State]]) -> None:
        """Validate public dict state passed through generate/generate_async."""
        if not isinstance(state, dict) or not state:
            return

        if self.config.colang_version == "1.0" and state.get("version") != "2.x":
            if "state" in state:
                raise InvalidStateError(
                    "Invalid Colang 1.0 state format: expected transcript state with an 'events' list."
                )
            if "events" not in state:
                raise InvalidStateError(
                    "Invalid Colang 1.0 state format: state must contain an 'events' key. "
                    "Use an empty dict {} to start a new conversation."
                )
            if not isinstance(state["events"], list):
                raise InvalidStateError("Invalid Colang 1.0 state format: 'events' must be a list.")
            return

        raise InvalidStateError(
            "Colang 2.0 dict state is not supported by generate/generate_async. "
            "Use rails.process_events_async(events, state) with a live State object "
            "for trusted in-process multi-turn execution. Public serialized Colang "
            "2.0 runtime state is not accepted."
        )

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
        # convert options to gen_options of type GenerationOptions
        gen_options: Optional[GenerationOptions] = None

        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided.")

        if prompt is not None and messages is not None:
            raise ValueError("Only one of prompt or messages can be provided.")

        if prompt is not None:
            # Currently, we transform the prompt request into a single turn conversation
            messages = [{"role": "user", "content": prompt}]

        # If a state object is specified, then we switch to "generation options" mode.
        # This is because we want the output to be a GenerationResponse which will contain
        # the output state.
        if state is not None:
            self._validate_public_state(state)

            if options is None:
                gen_options = GenerationOptions()
            elif isinstance(options, dict):
                gen_options = GenerationOptions(**options)
            else:
                gen_options = options
        else:
            # We allow options to be specified both as a dict and as an object.
            if options and isinstance(options, dict):
                gen_options = GenerationOptions(**options)
            elif isinstance(options, GenerationOptions):
                gen_options = options
            elif options is None:
                gen_options = None
            else:
                raise TypeError("options must be a dict or GenerationOptions")

        # Save the generation options in the current async context.
        # At this point, gen_options is either None or GenerationOptions
        generation_options_var.set(gen_options)

        needs_llm = gen_options is None or gen_options.rails.dialog is not False
        if needs_llm and not self.llm:
            log.warning("No main LLM specified in the config and no LLM provided via constructor.")

        if streaming_handler:
            streaming_handler_var.set(streaming_handler)

        # Initialize the object with additional explanation information.
        # We allow this to also be set externally. This is useful when multiple parallel
        # requests are made.
        self._explain_info = self._ensure_explain_info()

        raw_llm_request.set(messages)

        # If we have generation options, we also add them to the context
        if gen_options:
            messages = [
                {
                    "role": "context",
                    "content": {"generation_options": gen_options.model_dump()},
                }
            ] + (messages or [])

        # If the last message is from the assistant, rather than the user, then
        # we move that to the `$bot_message` variable. This is to enable a more
        # convenient interface for text output rails. Tool-call assistant messages
        # must remain in the history so they can be converted into BotToolCalls
        # events and evaluated by tool output rails.
        if (
            messages
            and messages[-1]["role"] == "assistant"
            and not messages[-1].get("tool_calls")
            and gen_options
            and gen_options.rails.dialog is False
        ):
            # We already have the first message with a context update, so we use that
            messages[0]["content"]["bot_message"] = messages[-1]["content"]
            messages = messages[0:-1]

        # TODO: Add support to load back history of events, next to history of messages
        #   This is important as without it, the LLM prediction is not as good.

        t0 = time.time()

        # Initialize the LLM stats
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)
        processing_log = []

        # The array of events corresponding to the provided sequence of messages.
        events = self._get_events_for_messages(messages, state)  # type: ignore

        if self.config.colang_version == "1.0":
            # If we had a state object, we also need to prepend the events from the state.
            state_events = []
            if state:
                assert isinstance(state, dict)
                state_events = state["events"]

            new_events = []
            # Compute the new events.
            try:
                new_events = await self.runtime.generate_events(state_events + events, processing_log=processing_log)
                output_state = None

            except Exception as e:
                log.error("Error in generate_async: %s", e, exc_info=True)
                streaming_handler = streaming_handler_var.get()
                if streaming_handler:
                    # Push an error chunk instead of None.
                    error_message = str(e)
                    error_dict = extract_error_json(error_message)
                    error_payload: str = json.dumps(error_dict)
                    await streaming_handler.push_chunk(error_payload)
                    # push a termination signal
                    await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore
                # Re-raise the exact exception
                raise
        else:
            # In generation mode, by default the bot response is an instant action.
            instant_actions = ["UtteranceBotAction"]
            if self.config.rails.actions.instant_actions is not None:
                instant_actions = self.config.rails.actions.instant_actions

            # Cast this explicitly to avoid certain warnings
            runtime: RuntimeV2_x = cast(RuntimeV2_x, self.runtime)

            # Compute the new events.
            # In generation mode, the processing is always blocking, i.e., it waits for
            # all local actions (sync and async).
            new_events, _output_state = await runtime.process_events(
                events, state=state, instant_actions=instant_actions, blocking=True
            )
            # The runtime State for 2.x is not publicly exposed through generate_async.
            # Callers that need stateful 2.x execution use process_events_async, which
            # returns the live State object directly.
            output_state = None

        # Extract and join all the messages from StartUtteranceBotAction events as the response.
        responses = []
        response_tool_calls = []
        response_events = []
        new_extra_events = []
        exception = None

        # The processing is different for Colang 1.0 and 2.0
        if self.config.colang_version == "1.0":
            for event in new_events:
                if event["type"] == "StartUtteranceBotAction":
                    # Check if we need to remove a message
                    if event["script"] == "(remove last message)":
                        responses = responses[0:-1]
                    else:
                        responses.append(event["script"])
                elif event["type"].endswith("Exception"):
                    exception = event

        else:
            for event in new_events:
                start_action_match = re.match(r"Start(.*Action)", event["type"])

                if start_action_match:
                    action_name = start_action_match[1]
                    # TODO: is there an elegant way to extract just the arguments?
                    arguments = {
                        k: v
                        for k, v in event.items()
                        if k != "type"
                        and k != "uid"
                        and k != "event_created_at"
                        and k != "source_uid"
                        and k != "action_uid"
                    }
                    response_tool_calls.append(
                        {
                            "id": event["action_uid"],
                            "type": "function",
                            "function": {"name": action_name, "arguments": arguments},
                        }
                    )

                elif event["type"] == "UtteranceBotActionFinished":
                    responses.append(event["final_script"])
                else:
                    # We just append the event
                    response_events.append(event)

        if exception:
            new_message: dict = {"role": "exception", "content": exception}

        else:
            # Ensure all items in responses are strings
            responses = [str(response) if not isinstance(response, str) else response for response in responses]
            new_message: dict = {"role": "assistant", "content": "\n".join(responses)}
        if response_tool_calls:
            new_message["tool_calls"] = response_tool_calls
        if response_events:
            new_message["events"] = response_events

        if self.config.colang_version == "1.0":
            events.extend(new_events)
            events.extend(new_extra_events)

            # If a state object is not used, then we use the implicit caching
            if state is None:
                # Save the new events in the history and update the cache
                cache_key = get_history_cache_key((messages) + [new_message])  # type: ignore
                self.events_history_cache[cache_key] = events
            else:
                output_state = {"events": events}

        # If logging is enabled, we log the conversation
        # TODO: add support for logging flag
        self._explain_info.colang_history = get_colang_history(events)
        if self.verbose:
            log.info(f"Conversation history so far: \n{self._explain_info.colang_history}")

        total_time = time.time() - t0
        log.info("--- :: Total processing took %.2f seconds. LLM Stats: %s" % (total_time, llm_stats))

        # If there is a streaming handler, we make sure we close it now
        streaming_handler = streaming_handler_var.get()
        if streaming_handler:
            # print("Closing the stream handler explicitly")
            await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore

        # IF tracing is enabled we need to set GenerationLog attrs
        original_log_options = None
        if self.config.tracing.enabled:
            if gen_options is None:
                gen_options = GenerationOptions()
            else:
                # create a copy of the gen_options to avoid modifying the original
                gen_options = gen_options.model_copy(deep=True)
            original_log_options = gen_options.log.model_copy(deep=True)

            # enable log options
            # it is aggressive, but these are required for tracing
            if (
                not gen_options.log.activated_rails
                or not gen_options.log.llm_calls
                or not gen_options.log.internal_events
            ):
                gen_options.log.activated_rails = True
                gen_options.log.llm_calls = True
                gen_options.log.internal_events = True

        tool_calls = extract_tool_calls_from_events(new_events)
        llm_metadata = get_and_clear_response_metadata_contextvar()
        reasoning_content = extract_bot_thinking_from_events(new_events)
        # If we have generation options, we prepare a GenerationResponse instance.
        if gen_options:
            # If a prompt was used, we only need to return the content of the message.
            if prompt:
                res = GenerationResponse(response=new_message["content"])
            else:
                res = GenerationResponse(response=[new_message])

            if reasoning_content:
                res.reasoning_content = reasoning_content

            if tool_calls:
                res.tool_calls = tool_calls

            if llm_metadata:
                res.llm_metadata = llm_metadata

            if self.config.colang_version == "1.0":
                # If output variables are specified, we extract their values
                if gen_options and gen_options.output_vars:
                    context = compute_context(events)
                    output_vars = gen_options.output_vars
                    if isinstance(output_vars, list):
                        # If we have only a selection of keys, we filter to only that.
                        res.output_data = {k: context.get(k) for k in output_vars}
                    else:
                        # Otherwise, we return the full context
                        res.output_data = context

                _log = compute_generation_log(processing_log)

                # Include information about activated rails and LLM calls if requested
                log_options = gen_options.log if gen_options else None
                if log_options and (log_options.activated_rails or log_options.llm_calls):
                    res.log = GenerationLog()

                    # We always include the stats
                    res.log.stats = _log.stats

                    if log_options.activated_rails:
                        res.log.activated_rails = _log.activated_rails

                    if log_options.llm_calls:
                        res.log.llm_calls = []
                        for activated_rail in _log.activated_rails:
                            for executed_action in activated_rail.executed_actions:
                                res.log.llm_calls.extend(executed_action.llm_calls)

                # Include internal events if requested
                if log_options and log_options.internal_events:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.internal_events = new_events

                # Include the Colang history if requested
                if log_options and log_options.colang_history:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.colang_history = get_colang_history(events)

                # Include the raw llm output if requested
                if gen_options and gen_options.llm_output:
                    # Currently, we include the output from the generation LLM calls.
                    for activated_rail in _log.activated_rails:
                        if activated_rail.type == "generation":
                            for executed_action in activated_rail.executed_actions:
                                for llm_call in executed_action.llm_calls:
                                    res.llm_output = llm_call.raw_response
            else:
                if gen_options and gen_options.output_vars:
                    raise ValueError("The `output_vars` option is not supported for Colang 2.0 configurations.")

                log_options = gen_options.log if gen_options else None
                if log_options and (
                    log_options.activated_rails
                    or log_options.llm_calls
                    or log_options.internal_events
                    or log_options.colang_history
                ):
                    raise ValueError("The `log` option is not supported for Colang 2.0 configurations.")

                if gen_options and gen_options.llm_output:
                    raise ValueError("The `llm_output` option is not supported for Colang 2.0 configurations.")

            # Include the state
            if state is not None:
                res.state = output_state

            if self.config.tracing.enabled:
                # TODO: move it to the top once resolved circular dependency of eval
                # lazy import to avoid circular dependency
                from nemoguardrails.tracing import Tracer

                span_format = getattr(self.config.tracing, "span_format", "opentelemetry")
                enable_content_capture = getattr(self.config.tracing, "enable_content_capture", False)
                # Create a Tracer instance with instantiated adapters and span configuration
                tracer = Tracer(
                    input=messages,
                    response=res,
                    adapters=self._log_adapters,
                    span_format=span_format,
                    enable_content_capture=enable_content_capture,
                )
                await tracer.export_async()

                # respect original log specification, if tracing added information to the output
                if original_log_options:
                    if not any(
                        (
                            original_log_options.internal_events,
                            original_log_options.activated_rails,
                            original_log_options.llm_calls,
                            original_log_options.colang_history,
                        )
                    ):
                        res.log = None
                    else:
                        # Ensure res.log exists before setting attributes
                        if res.log is not None:
                            if not original_log_options.internal_events:
                                res.log.internal_events = []
                            if not original_log_options.activated_rails:
                                res.log.activated_rails = []
                            if not original_log_options.llm_calls:
                                res.log.llm_calls = []

            return res
        else:
            # If a prompt is used, we only return the content of the message.

            if reasoning_content:
                thinking_trace = f"<think>{reasoning_content}</think>\n"
                new_message["content"] = thinking_trace + new_message["content"]

            if prompt:
                return new_message["content"]
            else:
                if tool_calls:
                    new_message["tool_calls"] = tool_calls
                return new_message

    def _validate_streaming_with_output_rails(self) -> None:
        if len(self.config.rails.output.flows) > 0 and (
            not self.config.rails.output.streaming or not self.config.rails.output.streaming.enabled
        ):
            raise StreamingNotSupportedError(
                "stream_async() cannot be used when output rails are configured but "
                "rails.output.streaming.enabled is False. Either set "
                "rails.output.streaming.enabled to True in your configuration, or use "
                "generate_async() instead of stream_async()."
            )

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

        if include_generation_metadata is not None:
            warnings.warn(
                "include_generation_metadata is deprecated, use include_metadata instead. "
                "It will be removed in version 0.22.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            include_metadata = include_generation_metadata

        self._validate_streaming_with_output_rails()
        self._validate_public_state(state)
        # if an external generator is provided, use it directly
        if generator:
            if self.config.rails.output.streaming and self.config.rails.output.streaming.enabled:
                return self._run_output_rails_in_streaming(
                    streaming_handler=generator,
                    output_rails_streaming_config=self.config.rails.output.streaming,
                    messages=messages,
                    prompt=prompt,
                )
            else:
                return generator

        self._explain_info = self._ensure_explain_info()

        streaming_handler = StreamingHandler(include_metadata=include_metadata)

        # Create a properly managed task with exception handling
        async def _generation_task():
            try:
                await self.generate_async(
                    prompt=prompt,
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=options,
                    state=state,
                )
            except Exception as e:
                # If an exception occurs during generation, push it to the streaming handler as a json string
                # This ensures the streaming pipeline is properly terminated
                log.error(f"Error in generation task: {e}", exc_info=True)
                error_message = str(e)
                error_dict = extract_error_json(error_message)
                error_payload = json.dumps(error_dict)
                await streaming_handler.push_chunk(error_payload)
                await streaming_handler.push_chunk(END_OF_STREAM)  # type: ignore

        task = asyncio.create_task(_generation_task())

        # Store task reference to prevent garbage collection and ensure proper cleanup
        if not hasattr(self, "_active_tasks"):
            self._active_tasks = set()
        self._active_tasks.add(task)

        # Clean up task when it's done
        def task_done_callback(task):
            self._active_tasks.discard(task)

        task.add_done_callback(task_done_callback)

        # when we have output rails we wrap the streaming handler
        # if len(self.config.rails.output.flows) > 0:
        #
        if self.config.rails.output.streaming and self.config.rails.output.streaming.enabled:
            base_iterator = self._run_output_rails_in_streaming(
                streaming_handler=streaming_handler,
                output_rails_streaming_config=self.config.rails.output.streaming,
                messages=messages,
                prompt=prompt,
            )
        else:
            base_iterator = streaming_handler

        async def wrapped_iterator():
            try:
                async for chunk in base_iterator:
                    if chunk is not None:
                        yield chunk
            finally:
                await task

        return wrapped_iterator()

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
        if rail_types is not None:
            options: Optional[dict] = {"rails": [r.value for r in rail_types]}
        else:
            options = _determine_rails_from_messages(messages)

        if options is None:
            last_content = messages[-1].get("content", "") if messages else ""
            return RailsResult(status=RailStatus.PASSED, content=last_content)

        rails_to_run = options["rails"]
        if "output" in rails_to_run:
            original_content = _get_last_content_by_role(messages, "assistant")
        else:
            original_content = _get_last_content_by_role(messages, "user")

        messages = _normalize_messages_for_rails(messages, rails_to_run)
        options["log"] = {"activated_rails": True}

        response = await self.generate_async(messages=messages, options=options)

        if not isinstance(response, GenerationResponse):
            raise RuntimeError(f"Expected GenerationResponse, got {type(response).__name__}")

        blocking_rail = _get_blocking_rail(response)
        result_content = _get_last_response_content(response)

        if blocking_rail:
            return RailsResult(status=RailStatus.BLOCKED, content=result_content, rail=blocking_rail)

        if result_content != original_content:
            return RailsResult(status=RailStatus.MODIFIED, content=result_content)
        return RailsResult(status=RailStatus.PASSED, content=result_content)

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
        """Register a custom action for the rails configuration."""
        self.runtime.register_action(action, name)
        return self

    def register_action_param(self, name: str, value: Any) -> Self:
        """Registers a custom action parameter."""
        self.runtime.register_action_param(name, value)
        return self

    def register_filter(self, filter_fn: Callable, name: Optional[str] = None) -> Self:
        """Register a custom filter for the rails configuration."""
        self.runtime.llm_task_manager.register_filter(filter_fn, name)
        return self

    def register_output_parser(self, output_parser: Callable, name: str) -> Self:
        """Register a custom output parser for the rails configuration."""
        self.runtime.llm_task_manager.register_output_parser(output_parser, name)
        return self

    def register_prompt_context(self, name: str, value_or_fn: Any) -> Self:
        """Register a value to be included in the prompt context.

        :name: The name of the variable or function that will be used.
        :value_or_fn: The value or function that will be used to generate the value.
        """
        self.runtime.llm_task_manager.register_prompt_context(name, value_or_fn)
        return self

    def register_embedding_search_provider(self, name: str, cls: Type[EmbeddingsIndex]) -> Self:
        """Register a new embedding search provider.

        Args:
            name: The name of the embedding search provider that will be used.
            cls: The class that will be used to generate and search embedding
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
        """
        register_embedding_provider(engine_name=name, model=cls)
        return self

    def explain(self) -> ExplainInfo:
        """Helper function to return the latest ExplainInfo object."""
        if self._explain_info is None:
            self._explain_info = self._ensure_explain_info()
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

        def _get_last_context_message(
            messages: Optional[List[dict]] = None,
        ) -> dict:
            if messages is None:
                return {}

            for message in reversed(messages):
                if message.get("role") == "context":
                    return message
            return {}

        def _get_latest_user_message(
            messages: Optional[List[dict]] = None,
        ) -> str:
            if messages is None:
                return ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    return message.get("content", "")
            return ""

        def _prepare_context_for_parallel_rails(
            chunk_str: str,
            prompt: Optional[str] = None,
            messages: Optional[List[dict]] = None,
        ) -> dict:
            """Prepare context for parallel rails execution."""
            context_message = _get_last_context_message(messages)
            user_message = prompt or _get_latest_user_message(messages)

            context = {
                "user_message": user_message,
                "bot_message": chunk_str,
            }

            if context_message:
                context.update(context_message["content"])

            return context

        def _create_events_for_chunk(chunk_str: str, context: dict) -> List[dict]:
            """Create events for running output rails on a chunk."""
            return [
                {"type": "ContextUpdate", "data": context},
                {"type": "BotMessage", "text": chunk_str},
            ]

        def _prepare_params(
            flow_id: str,
            action_name: str,
            bot_response_chunk: str,
            prompt: Optional[str] = None,
            messages: Optional[List[dict]] = None,
            action_params: Dict[str, Any] = {},
        ):
            context_message = _get_last_context_message(messages)
            user_message = prompt or _get_latest_user_message(messages)

            context = {
                "user_message": user_message,
                "bot_message": bot_response_chunk,
            }

            if context_message:
                context.update(context_message["content"])

            model_name = flow_id.split("$")[-1].split("=")[-1].strip('"')

            # Resolve $bot_message / $user_message into a new dict. action_params
            # is the shared flow config (reused across chunks and requests) and
            # must not be mutated in place.
            resolved_params = dict(action_params or {})
            for key, value in resolved_params.items():
                if value == "$bot_message":
                    resolved_params[key] = bot_response_chunk
                elif value == "$user_message":
                    resolved_params[key] = user_message

            return {
                # TODO:: are there other context variables that need to be passed?
                # passing events to compute context was not successful
                # context var failed due to different context
                "context": context,
                "llm_task_manager": self.runtime.llm_task_manager,
                "config": self.config,
                "model_name": model_name,
                "llms": self.runtime.registered_action_params.get("llms", {}),
                "llm": self.runtime.registered_action_params.get(f"{action_name}_llm", self.llm),
                **resolved_params,
            }

        buffer_strategy = get_buffer_strategy(output_rails_streaming_config)
        output_rails_flows_id = self.config.rails.output.flows
        stream_first = stream_first or output_rails_streaming_config.stream_first
        get_action_details = partial(get_action_details_from_flow_id, flows=self.config.flows)

        parallel_mode = getattr(self.config.rails.output, "parallel", False)

        async for chunk_batch in buffer_strategy(streaming_handler):
            user_output_chunks = chunk_batch.user_output_chunks
            # format processing_context for output rails processing (needs full context)
            bot_response_chunk = buffer_strategy.format_chunks(chunk_batch.processing_context)

            # check if user_output_chunks is a list of individual chunks
            # or if it's a JSON string, by convention this means an error occurred and the error dict is stored as a JSON
            if not isinstance(user_output_chunks, list):
                try:
                    json.loads(user_output_chunks)
                    yield user_output_chunks
                    return
                except (json.JSONDecodeError, TypeError):
                    # if it's not JSON, treat it as empty list
                    user_output_chunks = []

            if stream_first:
                # yield the individual chunks directly from the buffer strategy
                for chunk in user_output_chunks:
                    yield chunk

            if parallel_mode:
                try:
                    context = _prepare_context_for_parallel_rails(bot_response_chunk, prompt, messages)
                    events = _create_events_for_chunk(bot_response_chunk, context)

                    flows_with_params = {}
                    for flow_id in output_rails_flows_id:
                        action_name, action_params = get_action_details(flow_id)
                        params = _prepare_params(
                            flow_id=flow_id,
                            action_name=action_name,
                            bot_response_chunk=bot_response_chunk,
                            prompt=prompt,
                            messages=messages,
                            action_params=action_params,
                        )
                        flows_with_params[flow_id] = {
                            "action_name": action_name,
                            "params": params,
                        }

                    result_tuple = await self.runtime.action_dispatcher.execute_action(
                        "run_output_rails_in_parallel_streaming",
                        {
                            "flows_with_params": flows_with_params,
                            "events": events,
                        },
                    )

                    # ActionDispatcher.execute_action always returns (result, status)
                    result, status = result_tuple

                    if status != "success":
                        log.error(f"Parallel rails execution failed with status: {status}")
                        # continue processing the chunk even if rails fail
                        pass
                    else:
                        # if there are any stop events, content was blocked or internal error occurred
                        result_events = getattr(result, "events", None)
                        if result_events:
                            # extract the flow info from the first stop event
                            stop_event = result_events[0]
                            blocked_flow = stop_event.get("flow_id", "output rails")
                            error_type = stop_event.get("error_type")

                            if error_type == "internal_error":
                                error_message = stop_event.get("error_message", "Unknown error")
                                reason = f"Internal error in {blocked_flow} rail: {error_message}"
                                error_code = "rail_execution_failure"
                                error_type = "internal_error"
                            else:
                                reason = f"Blocked by {blocked_flow} rails."
                                error_code = "content_blocked"
                                error_type = "guardrails_violation"

                            error_data = {
                                "error": {
                                    "message": reason,
                                    "type": error_type,
                                    "param": blocked_flow,
                                    "code": error_code,
                                }
                            }
                            yield json.dumps(error_data)
                            return

                except Exception as e:
                    log.error(f"Error in parallel rail execution: {e}")
                    # don't block the stream for rail execution errors
                    # continue processing the chunk
                    pass

                # update explain info for parallel mode
                self._explain_info = self._ensure_explain_info()

            else:
                for flow_id in output_rails_flows_id:
                    action_name, action_params = get_action_details(flow_id)

                    params = _prepare_params(
                        flow_id=flow_id,
                        action_name=action_name,
                        bot_response_chunk=bot_response_chunk,
                        prompt=prompt,
                        messages=messages,
                        action_params=action_params,
                    )

                    result = await self.runtime.action_dispatcher.execute_action(action_name, params)
                    self._explain_info = self._ensure_explain_info()

                    action_func = self.runtime.action_dispatcher.get_action(action_name)

                    # Use the mapping to decide if the result indicates blocked content.
                    if is_output_blocked(result, action_func):
                        reason = f"Blocked by {flow_id} rails."

                        # return the error as a plain JSON string (not in SSE format)
                        # NOTE: When integrating with the OpenAI Python client, the server code should:
                        # 1. detect this JSON error object in the stream
                        # 2. terminate the stream
                        # 3. format the error following OpenAI's SSE format
                        # the OpenAI client will then properly raise an APIError with this error message

                        error_data = {
                            "error": {
                                "message": reason,
                                "type": "guardrails_violation",
                                "param": flow_id,
                                "code": "content_blocked",
                            }
                        }

                        # return as plain JSON: the server should detect this JSON and convert it to an HTTP error
                        yield json.dumps(error_data)
                        return

            if not stream_first:
                # yield the individual chunks directly from the buffer strategy
                for chunk in user_output_chunks:
                    yield chunk


def _determine_rails_from_messages(messages: List[dict]) -> Optional[dict]:
    roles = {msg.get("role") for msg in reversed(messages)}
    has_user = "user" in roles
    has_assistant = "assistant" in roles

    if not has_user and not has_assistant:
        log.warning(
            "check() called with no user or assistant messages. "
            "Only system, context, or tool messages found. "
            "Returning passing result without running rails."
        )
        return None

    if has_user and has_assistant:
        return {"rails": ["input", "output"]}
    if has_user:
        return {"rails": ["input"]}
    return {"rails": ["output"]}


def _normalize_messages_for_rails(
    messages: List[dict],
    rails: List[str],
) -> List[dict]:
    if rails == ["output"]:
        has_user = any(msg.get("role") == "user" for msg in messages)
        if not has_user:
            return [{"role": "user", "content": ""}] + messages

    return messages


def _get_last_content_by_role(messages: List[dict], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return msg.get("content", "")
    return ""


def _get_blocking_rail(response: "GenerationResponse") -> Optional[str]:
    if response.log and response.log.activated_rails:
        for rail in response.log.activated_rails:
            if rail.stop:
                return rail.name
    return None


def _get_last_response_content(response: "GenerationResponse") -> str:
    if isinstance(response.response, list) and response.response:
        return response.response[-1].get("content", "")
    if isinstance(response.response, str):
        return response.response
    return ""
