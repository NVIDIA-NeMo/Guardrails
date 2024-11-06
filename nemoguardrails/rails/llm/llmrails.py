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

"""LLM Rails entry point."""

import asyncio
import importlib.util
import logging
import os
import re
import threading
import time
import warnings
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Type, Union, cast

from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import BaseLLM

from nemoguardrails.actions.llm.generation import LLMGenerationActions
from nemoguardrails.actions.llm.utils import get_colang_history
from nemoguardrails.actions.v2_x.generation import LLMGenerationActionsV2dotx
from nemoguardrails.colang import parse_colang_file
from nemoguardrails.colang.v1_0.runtime.flows import compute_context
from nemoguardrails.colang.v1_0.runtime.runtime import Runtime, RuntimeV1_0
from nemoguardrails.colang.v2_x.runtime.flows import Action, State
from nemoguardrails.colang.v2_x.runtime.runtime import RuntimeV2_x
from nemoguardrails.colang.v2_x.runtime.serialization import (
    json_to_state,
    state_to_json,
)
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
from nemoguardrails.kb.kb import KnowledgeBase
from nemoguardrails.llm.providers import get_llm_provider, get_llm_provider_names
from nemoguardrails.logging.explain import ExplainInfo
from nemoguardrails.logging.processing_log import compute_generation_log
from nemoguardrails.logging.stats import LLMStats
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.patch_asyncio import check_sync_call_from_async_loop
from nemoguardrails.rails.llm.config import EmbeddingSearchProvider, Model, RailsConfig
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.rails.llm.utils import get_history_cache_key
from nemoguardrails.streaming import StreamingHandler
from nemoguardrails.utils import get_or_create_event_loop, new_event_dict, new_uuid

log = logging.getLogger(__name__)

process_events_semaphore = asyncio.Semaphore(1)


class LLMRails:
    """Rails based on a given configuration."""

    config: RailsConfig
    llm: Optional[BaseLLM]
    runtime: Runtime

    def __init__(
        self, config: RailsConfig, llm: Optional[BaseLLM] = None, verbose: bool = False
    ):
        """Initializes the LLMRails instance.

        Args:
            config: A rails configuration.
            llm: An optional LLM engine to use.
            verbose: Whether the logging should be verbose or not.
        """
        self.config = config
        self.llm = llm
        self.verbose = verbose

        if self.verbose:
            set_verbose(True, llm_calls=True)

        # We allow the user to register additional embedding search providers, so we keep
        # an index of them.
        self.embedding_search_providers = {}

        # The default embeddings model is using FastEmbed
        self.default_embedding_model = "all-MiniLM-L6-v2"
        self.default_embedding_engine = "FastEmbed"

        # We keep a cache of the events history associated with a sequence of user messages.
        # TODO: when we update the interface to allow to return a "state object", this
        #   should be removed
        self.events_history_cache = {}

        # Weather the main LLM supports streaming
        self.main_llm_supports_streaming = False

        # InteractionLogAdapters used for tracing
        if config.tracing:
            from nemoguardrails.tracing import create_log_adapters

            self._log_adapters = create_log_adapters(config.tracing)

        # We also load the default flows from the `default_flows.yml` file in the current folder.
        # But only for version 1.0.
        # TODO: decide on the default flows for 2.x.
        if config.colang_version == "1.0":
            # We also load the default flows from the `llm_flows.co` file in the current folder.
            current_folder = os.path.dirname(__file__)
            default_flows_file = "llm_flows.co"
            default_flows_path = os.path.join(current_folder, default_flows_file)
            with open(default_flows_path, "r") as f:
                default_flows_content = f.read()
                default_flows = parse_colang_file(
                    default_flows_file, default_flows_content
                )["flows"]

            # We mark all the default flows as system flows.
            for flow_config in default_flows:
                flow_config["is_system_flow"] = True

            # We add the default flows to the config.
            self.config.flows.extend(default_flows)

            # We also need to load the content from the components library.
            library_path = os.path.join(os.path.dirname(__file__), "../../library")
            for root, dirs, files in os.walk(library_path):
                for file in files:
                    # Extract the full path for the file
                    full_path = os.path.join(root, file)
                    if file.endswith(".co"):
                        log.debug(f"Loading file: {full_path}")
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = parse_colang_file(
                                file, content=f.read(), version=config.colang_version
                            )
                            if not content:
                                continue

                        # We mark all the flows coming from the guardrails library as system flows.
                        for flow_config in content["flows"]:
                            flow_config["is_system_flow"] = True

                        # We load all the flows
                        self.config.flows.extend(content["flows"])

                        # And all the messages as well, if they have not been overwritten
                        for message_id, utterances in content.get(
                            "bot_messages", {}
                        ).items():
                            if message_id not in self.config.bot_messages:
                                self.config.bot_messages[message_id] = utterances

        # Last but not least, we mark all the flows that are used in any of the rails
        # as system flows (so they don't end up in the prompt).

        rail_flow_ids = (
            config.rails.input.flows
            + config.rails.output.flows
            + config.rails.retrieval.flows
        )

        for flow_config in self.config.flows:
            if flow_config.get("id") in rail_flow_ids:
                flow_config["is_system_flow"] = True

                # We also mark them as subflows by default, to simplify the syntax
                flow_config["is_subflow"] = True

        # We check if the configuration or any of the imported ones have config.py modules.
        config_modules = []
        for _path in list(self.config.imported_paths.values()) + [
            self.config.config_path
        ]:
            if _path:
                filepath = os.path.join(_path, "config.py")
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    spec = importlib.util.spec_from_file_location(filename, filepath)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    config_modules.append(config_module)

        # First, we initialize the runtime.
        if config.colang_version == "1.0":
            self.runtime = RuntimeV1_0(config=config, verbose=verbose)
        elif config.colang_version == "2.x":
            self.runtime = RuntimeV2_x(config=config, verbose=verbose)
        else:
            raise ValueError(f"Unsupported colang version: {config.colang_version}.")

        # If we have a config_modules with an `init` function, we call it.
        # We need to call this here because the `init` might register additional
        # LLM providers.
        for config_module in config_modules:
            if hasattr(config_module, "init"):
                config_module.init(self)

        # If we have a customized embedding model, we'll use it.
        for model in self.config.models:
            if model.type == "embeddings":
                self.default_embedding_model = model.model
                self.default_embedding_engine = model.engine
                break

        # We run some additional checks on the config
        self._validate_config()

        # Next, we initialize the LLM engines (main engine and action engines if specified).
        self._init_llms()

        # Next, we initialize the LLM Generate actions and register them.
        llm_generation_actions_class = (
            LLMGenerationActions
            if config.colang_version == "1.0"
            else LLMGenerationActionsV2dotx
        )
        self.llm_generation_actions = llm_generation_actions_class(
            config=config,
            llm=self.llm,
            llm_task_manager=self.runtime.llm_task_manager,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
            verbose=verbose,
        )

        # If there's already an action registered, we don't override.
        self.runtime.register_actions(self.llm_generation_actions, override=False)

        # Next, we initialize the Knowledge Base
        # There are still some edge cases not covered by nest_asyncio.
        # Using a separate thread always for now.
        loop = get_or_create_event_loop()
        if True or check_sync_call_from_async_loop():
            t = threading.Thread(target=asyncio.run, args=(self._init_kb(),))
            t.start()
            t.join()
        else:
            loop.run_until_complete(self._init_kb())

        # We also register the kb as a parameter that can be passed to actions.
        self.runtime.register_action_param("kb", self.kb)

        # Reference to the general ExplainInfo object.
        self.explain_info = None

    def update_llm(self, llm):
        """Replace the main LLM with the provided one.

        Arguments:
            llm: The new LLM that should be used.
        """
        self.llm = llm
        self.llm_generation_actions.llm = llm
        self.runtime.register_action_param("llm", llm)

    def _validate_config(self):
        """Runs additional validation checks on the config."""

        if self.config.colang_version == "1.0":
            existing_flows_names = set([flow.get("id") for flow in self.config.flows])
        else:
            existing_flows_names = set([flow.get("name") for flow in self.config.flows])

        for flow_name in self.config.rails.input.flows:
            # content safety check input/output flows are special as they have parameters
            if flow_name.startswith("content safety check"):
                continue
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided input rail flow `{flow_name}` does not exist"
                )

        for flow_name in self.config.rails.output.flows:
            if flow_name.startswith("content safety check"):
                continue
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided output rail flow `{flow_name}` does not exist"
                )

        for flow_name in self.config.rails.retrieval.flows:
            if flow_name not in existing_flows_names:
                raise ValueError(
                    f"The provided retrieval rail flow `{flow_name}` does not exist"
                )

        # If both passthrough mode and single call mode are specified, we raise an exception.
        if self.config.passthrough and self.config.rails.dialog.single_call.enabled:
            raise ValueError(
                f"The passthrough mode and the single call dialog rails mode can't be used at the same time. "
                f"The single call mode needs to use an altered prompt when prompting the LLM. "
            )

    async def _init_kb(self):
        """Initializes the knowledge base."""
        self.kb = None

        if not self.config.docs:
            return

        documents = [doc.content for doc in self.config.docs]
        self.kb = KnowledgeBase(
            documents=documents,
            config=self.config.knowledge_base,
            get_embedding_search_provider_instance=self._get_embeddings_search_provider_instance,
        )
        self.kb.init()
        await self.kb.build()

    @staticmethod
    def get_model_cls_and_kwargs(
        model_config: Model,
    ) -> Tuple[Type[BaseLanguageModel], Dict[str, Any]]:
        """Helper to return the model class and kwargs for initialization."""
        if model_config.engine not in get_llm_provider_names():
            msg = f"Unknown LLM engine: {model_config.engine}."
            if model_config.engine == "openai":
                msg += " Please install langchain-openai using `pip install langchain-openai`."

            raise Exception(msg)

        provider_cls = get_llm_provider(model_config)
        # We need to compute the kwargs for initializing the LLM
        kwargs = model_config.parameters

        # We also need to pass the model, if specified
        if model_config.model:
            # Some LLM providers use `model_name` instead of model. For backward compatibility
            # we keep this hard-coded mapping.
            if model_config.engine in [
                "azure",
                "openai",
                "gooseai",
                "nlpcloud",
                "petals",
                "trt_llm",
                "vertexai",
            ]:
                kwargs["model_name"] = model_config.model
            elif (
                model_config.engine == "nvidia_ai_endpoints"
                or model_config.engine == "nim"
            ):
                kwargs["model"] = model_config.model
            else:
                # The `__fields__` attribute is computed dynamically by pydantic.
                if "model" in provider_cls.__fields__:
                    kwargs["model"] = model_config.model

        return provider_cls, kwargs

    def _init_llms(self):
        """
        Initializes the right LLM engines based on the configuration.
        There can be multiple LLM engines and types that can be specified in the config.
        The main LLM engine is the one that will be used for all the core guardrails generations.
        Other LLM engines can be specified for use in specific actions.

        The reason we provide an option for decoupling the main LLM engine from the action LLM
        is to allow for flexibility in using specialized LLM engines for specific actions.
        """

        # If we already have a pre-configured one,
        # we just need to register the LLM as an action param.
        if self.llm is not None:
            self.runtime.register_action_param("llm", self.llm)
            return

        llms = dict()
        for llm_config in self.config.models:
            if llm_config.type == "embeddings":
                pass
            else:
                provider_cls, kwargs = self.get_model_cls_and_kwargs(llm_config)

                if self.config.streaming:
                    if "streaming" in provider_cls.__fields__:
                        kwargs["streaming"] = True
                        self.main_llm_supports_streaming = True
                    else:
                        log.warning(
                            f"The provider {provider_cls.__name__} does not support streaming."
                        )

                if llm_config.type == "main" or len(self.config.models) == 1:
                    self.llm = provider_cls(**kwargs)
                    self.runtime.register_action_param("llm", self.llm)
                else:
                    model_name = f"{llm_config.type}_llm"
                    setattr(self, model_name, provider_cls(**kwargs))
                    self.runtime.register_action_param(
                        model_name, getattr(self, model_name)
                    )
                    llms[llm_config.type] = getattr(self, model_name)

            self.runtime.register_action_param("llms", llms)

    def _get_embeddings_search_provider_instance(
        self, esp_config: Optional[EmbeddingSearchProvider] = None
    ) -> EmbeddingsIndex:
        if esp_config is None:
            esp_config = EmbeddingSearchProvider()

        if esp_config.name == "default":
            from nemoguardrails.embeddings.basic import BasicEmbeddingsIndex

            return BasicEmbeddingsIndex(
                embedding_model=esp_config.parameters.get(
                    "embedding_model", self.default_embedding_model
                ),
                embedding_engine=esp_config.parameters.get(
                    "embedding_engine", self.default_embedding_engine
                ),
                cache_config=esp_config.cache,
                # We make sure we also pass additional relevant params.
                **{
                    k: v
                    for k, v in esp_config.parameters.items()
                    if k
                    in [
                        "use_batching",
                        "max_batch_size",
                        "matx_batch_hold",
                        "search_threshold",
                    ]
                    and v is not None
                },
            )
        else:
            if esp_config.name not in self.embedding_search_providers:
                raise Exception(f"Unknown embedding search provider: {esp_config.name}")
            else:
                kwargs = esp_config.parameters
                return self.embedding_search_providers[esp_config.name](**kwargs)

    def _get_events_for_messages(self, messages: List[dict], state: Any):
        """Return the list of events corresponding to the provided messages.

        Tries to find a prefix of messages for which we have already a list of events
        in the cache. For the rest, they are converted as is.

        The reason this cache exists is that we want to benefit from events generated in
        previous turns, which can't be computed again because it would be expensive (e.g.,
        involving multiple LLM calls).

        When an explicit state object will be added, this mechanism can be removed.

        Args:
            messages: The list of messages.

        Returns:
            A list of events.
        """
        events = []

        if self.config.colang_version == "1.0":
            # We try to find the longest prefix of messages for which we have a cache
            # of events.
            p = len(messages) - 1
            while p > 0:
                cache_key = get_history_cache_key(messages[0:p])
                if cache_key in self.events_history_cache:
                    events = self.events_history_cache[cache_key].copy()
                    break

                p -= 1

            # For the rest of the messages, we transform them directly into events.
            # TODO: Move this to separate function once more types of messages are supported.
            for idx in range(p, len(messages)):
                msg = messages[idx]
                if msg["role"] == "user":
                    events.append(
                        {
                            "type": "UtteranceUserActionFinished",
                            "final_transcript": msg["content"],
                        }
                    )

                    # If it's not the last message, we also need to add the `UserMessage` event
                    if idx != len(messages) - 1:
                        events.append(
                            {
                                "type": "UserMessage",
                                "text": msg["content"],
                            }
                        )

                elif msg["role"] == "assistant":
                    action_uid = new_uuid()
                    start_event = new_event_dict(
                        "StartUtteranceBotAction",
                        script=msg["content"],
                        action_uid=action_uid,
                    )
                    finished_event = new_event_dict(
                        "UtteranceBotActionFinished",
                        final_script=msg["content"],
                        is_success=True,
                        action_uid=action_uid,
                    )
                    events.extend([start_event, finished_event])
                elif msg["role"] == "context":
                    events.append({"type": "ContextUpdate", "data": msg["content"]})
                elif msg["role"] == "event":
                    events.append(msg["event"])
        else:
            for idx in range(len(messages)):
                msg = messages[idx]
                if msg["role"] == "user":
                    events.append(
                        {
                            "type": "UtteranceUserActionFinished",
                            "final_transcript": msg["content"],
                        }
                    )

                elif msg["role"] == "assistant":
                    raise ValueError(
                        "Providing `assistant` messages as input is not supported for Colang 2.0 configurations."
                    )
                elif msg["role"] == "context":
                    events.append({"type": "ContextUpdate", "data": msg["content"]})
                elif msg["role"] == "event":
                    events.append(msg["event"])
                elif msg["role"] == "tool":
                    action_uid = msg["tool_call_id"]
                    return_value = msg["content"]
                    action: Action = state.actions[action_uid]
                    events.append(
                        new_event_dict(
                            f"{action.name}Finished",
                            action_uid=action_uid,
                            action_name=action.name,
                            status="success",
                            is_success=True,
                            return_value=return_value,
                            events=[],
                        )
                    )

        return events

    async def generate_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        options: Optional[Union[dict, GenerationOptions]] = None,
        state: Optional[Union[dict, State]] = None,
        streaming_handler: Optional[StreamingHandler] = None,
        return_context: bool = False,
    ) -> Union[str, dict, GenerationResponse, Tuple[dict, dict]]:
        """Generate a completion or a next message.

        The format for messages is the following:

        ```python
            [
                {"role": "context", "content": {"user_name": "John"}},
                {"role": "user", "content": "Hello! How are you?"},
                {"role": "assistant", "content": "I am fine, thank you!"},
                {"role": "event", "event": {"type": "UserSilent"}},
                ...
            ]
        ```

        Args:
            prompt: The prompt to be used for completion.
            messages: The history of messages to be used to generate the next message.
            options: Options specific for the generation.
            state: The state object that should be used as the starting point.
            streaming_handler: If specified, and the config supports streaming, the
              provided handler will be used for streaming.
            return_context: Whether to return the context at the end of the run.

        Returns:
            The completion (when a prompt is provided) or the next message.

        System messages are not yet supported."""
        # If a state object is specified, then we switch to "generation options" mode.
        # This is because we want the output to be a GenerationResponse which will contain
        # the output state.
        if state is not None:
            # We deserialize the state if needed.
            if isinstance(state, dict) and state.get("version", "1.0") == "2.x":
                state = json_to_state(state["state"])

            if options is None:
                options = GenerationOptions()

        # We allow options to be specified both as a dict and as an object.
        if options and isinstance(options, dict):
            options = GenerationOptions(**options)

        # Save the generation options in the current async context.
        generation_options_var.set(options)

        if return_context:
            warnings.warn(
                "The `return_context` argument is deprecated and will be removed in 0.9.0. "
                "Use `GenerationOptions.output_vars = True` instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # And we use the generation options mechanism instead.
            if options is None:
                options = GenerationOptions()
            options.output_vars = True

        if streaming_handler:
            streaming_handler_var.set(streaming_handler)

        # Initialize the object with additional explanation information.
        # We allow this to also be set externally. This is useful when multiple parallel
        # requests are made.
        explain_info = explain_info_var.get()
        if explain_info is None:
            explain_info = ExplainInfo()
            explain_info_var.set(explain_info)

            # We also keep a general reference to this object
            self.explain_info = explain_info

        if prompt is not None:
            # Currently, we transform the prompt request into a single turn conversation
            messages = [{"role": "user", "content": prompt}]
            raw_llm_request.set(prompt)
        else:
            raw_llm_request.set(messages)

        # If we have generation options, we also add them to the context
        if options:
            messages = [
                {"role": "context", "content": {"generation_options": options.dict()}}
            ] + messages

        # If the last message is from the assistant, rather than the user, then
        # we move that to the `$bot_message` variable. This is to enable a more
        # convenient interface. (only when dialog rails are disabled)
        if (
            messages[-1]["role"] == "assistant"
            and options
            and options.rails.dialog is False
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
        events = self._get_events_for_messages(messages, state)

        if self.config.colang_version == "1.0":
            # If we had a state object, we also need to prepend the events from the state.
            state_events = []
            if state:
                assert isinstance(state, dict)
                state_events = state["events"]

            # Compute the new events.
            new_events = await self.runtime.generate_events(
                state_events + events, processing_log=processing_log
            )
            output_state = None
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
            new_events, output_state = await runtime.process_events(
                events, state=state, instant_actions=instant_actions, blocking=True
            )
            # We also encode the output state as a JSON
            output_state = {"state": state_to_json(output_state), "version": "2.x"}

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
            new_message = {"role": "exception", "content": exception}
        else:
            new_message = {"role": "assistant", "content": "\n".join(responses)}
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
                cache_key = get_history_cache_key(messages + [new_message])
                self.events_history_cache[cache_key] = events
            else:
                output_state = {"events": events}

        # If logging is enabled, we log the conversation
        # TODO: add support for logging flag
        explain_info.colang_history = get_colang_history(events)
        if self.verbose:
            log.info(f"Conversation history so far: \n{explain_info.colang_history}")

        total_time = time.time() - t0
        log.info(
            "--- :: Total processing took %.2f seconds. LLM Stats: %s"
            % (total_time, llm_stats)
        )

        # If there is a streaming handler, we make sure we close it now
        streaming_handler = streaming_handler_var.get()
        if streaming_handler:
            # print("Closing the stream handler explicitly")
            await streaming_handler.push_chunk(None)

        # IF tracing is enabled we need to set GenerationLog attrs
        if self.config.tracing.enabled:
            if options is None:
                options = GenerationOptions()
            if (
                not options.log.activated_rails
                or not options.log.llm_calls
                or not options.log.internal_events
            ):
                options.log.activated_rails = True
                options.log.llm_calls = True
                options.log.internal_events = True

        # If we have generation options, we prepare a GenerationResponse instance.
        if options:
            # If a prompt was used, we only need to return the content of the message.
            if prompt:
                res = GenerationResponse(response=new_message["content"])
            else:
                res = GenerationResponse(response=[new_message])

            if self.config.colang_version == "1.0":
                # If output variables are specified, we extract their values
                if options.output_vars:
                    context = compute_context(events)
                    if isinstance(options.output_vars, list):
                        # If we have only a selection of keys, we filter to only that.
                        res.output_data = {
                            k: context.get(k) for k in options.output_vars
                        }
                    else:
                        # Otherwise, we return the full context
                        res.output_data = context

                    # If the `return_context` is used, then we return a tuple to keep
                    # the interface compatible.
                    # TODO: remove this in 0.10.0.
                    if return_context:
                        return new_message, context

                _log = compute_generation_log(processing_log)

                # Include information about activated rails and LLM calls if requested
                if options.log.activated_rails or options.log.llm_calls:
                    res.log = GenerationLog()

                    # We always include the stats
                    res.log.stats = _log.stats

                    if options.log.activated_rails:
                        res.log.activated_rails = _log.activated_rails

                    if options.log.llm_calls:
                        res.log.llm_calls = []
                        for activated_rail in _log.activated_rails:
                            for executed_action in activated_rail.executed_actions:
                                res.log.llm_calls.extend(executed_action.llm_calls)

                # Include internal events if requested
                if options.log.internal_events:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.internal_events = new_events

                # Include the Colang history if requested
                if options.log.colang_history:
                    if res.log is None:
                        res.log = GenerationLog()

                    res.log.colang_history = get_colang_history(events)

                # Include the raw llm output if requested
                if options.llm_output:
                    # Currently, we include the output from the generation LLM calls.
                    for activated_rail in _log.activated_rails:
                        if activated_rail.type == "generation":
                            for executed_action in activated_rail.executed_actions:
                                for llm_call in executed_action.llm_calls:
                                    res.llm_output = llm_call.raw_response
            else:
                if options.output_vars:
                    raise ValueError(
                        "The `output_vars` option is not supported for Colang 2.0 configurations."
                    )

                if (
                    options.log.activated_rails
                    or options.log.llm_calls
                    or options.log.internal_events
                    or options.log.colang_history
                ):
                    raise ValueError(
                        "The `log` option is not supported for Colang 2.0 configurations."
                    )

                if options.llm_output:
                    raise ValueError(
                        "The `llm_output` option is not supported for Colang 2.0 configurations."
                    )

            # Include the state
            if state is not None:
                res.state = output_state

            if self.config.tracing.enabled:
                # TODO: move it to the top once resolved circular dependency of eval
                # lazy import to avoid circular dependency
                from nemoguardrails.tracing import Tracer

                # Create a Tracer instance with instantiated adapters
                tracer = Tracer(
                    input=messages, response=res, adapters=self._log_adapters
                )
                await tracer.export_async()
                res = res.response[0]
            return res
        else:
            # If a prompt is used, we only return the content of the message.
            if prompt:
                return new_message["content"]
            else:
                return new_message

    def stream_async(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        """Simplified interface for getting directly the streamed tokens from the LLM."""
        streaming_handler = StreamingHandler()

        asyncio.create_task(
            self.generate_async(
                prompt=prompt,
                messages=messages,
                streaming_handler=streaming_handler,
            )
        )

        return streaming_handler

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
        return_context: bool = False,
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
                return_context=return_context,
            )
        )

    async def generate_events_async(
        self,
        events: List[dict],
    ) -> List[dict]:
        """Generate the next events based on the provided history.

        The format for events is the following:

        ```python
            [
                {"type": "...", ...},
                ...
            ]
        ```

        Args:
            events: The history of events to be used to generate the next events.
            options: The options to be used for the generation.

        Returns:
            The newly generate event(s).

        """
        t0 = time.time()

        # Initialize the LLM stats
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        # Compute the new events.
        processing_log = []
        new_events = await self.runtime.generate_events(
            events, processing_log=processing_log
        )

        # If logging is enabled, we log the conversation
        # TODO: add support for logging flag
        if self.verbose:
            history = get_colang_history(events)
            log.info(f"Conversation history so far: \n{history}")

        log.info("--- :: Total processing took %.2f seconds." % (time.time() - t0))
        log.info("--- :: Stats: %s" % llm_stats)

        return new_events

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
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
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
        t0 = time.time()
        llm_stats = LLMStats()
        llm_stats_var.set(llm_stats)

        # Compute the new events.
        # We need to protect 'process_events' to be called only once at a time
        # TODO (cschueller): Why is this?
        async with process_events_semaphore:
            output_events, output_state = await self.runtime.process_events(
                events, state, blocking
            )

        took = time.time() - t0
        # Small tweak, disable this when there were no events (or it was just too fast).
        if took > 0.1:
            log.info("--- :: Total processing took %.2f seconds." % took)
            log.info("--- :: Stats: %s" % llm_stats)

        return output_events, output_state

    def process_events(
        self,
        events: List[dict],
        state: Optional[dict] = None,
        blocking: bool = False,
    ) -> Tuple[List[dict], dict]:
        """Synchronous version of `LLMRails.process_events_async`."""

        if check_sync_call_from_async_loop():
            raise RuntimeError(
                "You are using the sync `generate_events` inside async code. "
                "You should replace with `await generate_events_async(...)."
            )

        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            self.process_events_async(events, state, blocking)
        )

    def register_action(self, action: callable, name: Optional[str] = None):
        """Register a custom action for the rails configuration."""
        self.runtime.register_action(action, name)

    def register_action_param(self, name: str, value: Any):
        """Registers a custom action parameter."""
        self.runtime.register_action_param(name, value)

    def register_filter(self, filter_fn: callable, name: Optional[str] = None):
        """Register a custom filter for the rails configuration."""
        self.runtime.llm_task_manager.register_filter(filter_fn, name)

    def register_output_parser(self, output_parser: callable, name: str):
        """Register a custom output parser for the rails configuration."""
        self.runtime.llm_task_manager.register_output_parser(output_parser, name)

    def register_prompt_context(self, name: str, value_or_fn: Any):
        """Register a value to be included in the prompt context.

        :name: The name of the variable or function that will be used.
        :value_or_fn: The value or function that will be used to generate the value.
        """
        self.runtime.llm_task_manager.register_prompt_context(name, value_or_fn)

    def register_embedding_search_provider(
        self, name: str, cls: Type[EmbeddingsIndex]
    ) -> None:
        """Register a new embedding search provider.

        Args:
            name: The name of the embedding search provider that will be used.
            cls: The class that will be used to generate and search embedding
        """

        self.embedding_search_providers[name] = cls

    def register_embedding_provider(
        self, cls: Type[EmbeddingModel], name: Optional[str] = None
    ) -> None:
        """Register a custom embedding provider.

        Args:
            model (Type[EmbeddingModel]): The embedding model class.
            name (str): The name of the embedding engine. If available in the model, it will be used.

        Raises:
            ValueError: If the engine name is not provided and the model does not have an engine name.
            ValueError: If the model does not have 'encode' or 'encode_async' methods.
        """
        register_embedding_provider(engine_name=name, model=cls)

    def explain(self) -> ExplainInfo:
        """Helper function to return the latest ExplainInfo object."""
        return self.explain_info

    def __getstate__(self):
        return {"config": self.config}

    def __setstate__(self, state):
        if state["config"].config_path:
            config = RailsConfig.from_path(state["config"].config_path)
        else:
            config = state["config"]
        self.__init__(config=config, verbose=False)
