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

import asyncio
import contextvars
import importlib.util
import json
import logging
import os.path
import re
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel, Field, root_validator, validator
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.rails.llm.options import GenerationOptions, GenerationResponse
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.server.schemas.openai import (
    GuardrailsModel,
    ModelsResponse,
    OpenAIRequestFields,
    ResponseBody,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GuardrailsApp(FastAPI):
    """Custom FastAPI subclass with additional attributes for Guardrails server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize custom attributes
        self.default_config_id: Optional[str] = None
        self.rails_config_path: str = ""
        self.disable_chat_ui: bool = False
        self.auto_reload: bool = False
        self.stop_signal: bool = False
        self.single_config_mode: bool = False
        self.single_config_id: Optional[str] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Future] = None


# The list of registered loggers. Can be used to send logs to various
# backends and storage engines.
registered_loggers: List[Callable] = []

api_description = """Guardrails Sever API."""

# The headers for each request
api_request_headers: contextvars.ContextVar = contextvars.ContextVar("headers")

# The datastore that the Server should use.
# This is currently used only for storing threads.
# TODO: refactor to wrap the FastAPI instance inside a RailsServer class
#  and get rid of all the global attributes.
datastore: Optional[DataStore] = None


@asynccontextmanager
async def lifespan(app: GuardrailsApp):
    # Startup logic here
    """Register any additional challenges, if available at startup."""
    challenges_files = os.path.join(app.rails_config_path, "challenges.json")

    if os.path.exists(challenges_files):
        with open(challenges_files) as f:
            register_challenges(json.load(f))

    # If there is a `config.yml` in the root `app.rails_config_path`, then
    # that means we are in single config mode.
    if os.path.exists(
        os.path.join(app.rails_config_path, "config.yml")
    ) or os.path.exists(os.path.join(app.rails_config_path, "config.yaml")):
        app.single_config_mode = True
        app.single_config_id = os.path.basename(app.rails_config_path)
    else:
        # If we're not in single-config mode, we check if we have a config.py for the
        # server configuration.
        filepath = os.path.join(app.rails_config_path, "config.py")
        if os.path.exists(filepath):
            filename = os.path.basename(filepath)
            spec = importlib.util.spec_from_file_location(filename, filepath)
            if spec is not None and spec.loader is not None:
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
            else:
                config_module = None

            # If there is an `init` function, we call it with the reference to the app.
            if config_module is not None and hasattr(config_module, "init"):
                config_module.init(app)

    # Finally, we register the static frontend UI serving

    if not app.disable_chat_ui:
        FRONTEND_DIR = utils.get_chat_ui_data_path("frontend")

        app.mount(
            "/",
            StaticFiles(
                directory=FRONTEND_DIR,
                html=True,
            ),
            name="chat",
        )
    else:

        @app.get("/")
        async def root_handler():
            return {"status": "ok"}

    if app.auto_reload:
        app.loop = asyncio.get_running_loop()
        # Store the future directly as task
        app.task = app.loop.run_in_executor(None, start_auto_reload_monitoring)

    yield

    # Shutdown logic here
    if app.auto_reload:
        app.stop_signal = True
        if hasattr(app, "task") and app.task is not None:
            app.task.cancel()
        log.info("Shutting down file observer")
    else:
        pass


app = GuardrailsApp(
    title="Guardrails Server API",
    description=api_description,
    version="0.1.0",
    license_info={"name": "Apache License, Version 2.0"},
    lifespan=lifespan,
)

ENABLE_CORS = os.getenv("NEMO_GUARDRAILS_SERVER_ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("NEMO_GUARDRAILS_SERVER_ALLOWED_ORIGINS", "*")

if ENABLE_CORS:
    # Split origins by comma
    origins = ALLOWED_ORIGINS.split(",")

    log.info(f"CORS enabled with the following origins: {origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.default_config_id = None

# By default, we use the rails in the examples folder
app.rails_config_path = utils.get_examples_data_path("bots")

# Weather the chat UI is enabled or not.
app.disable_chat_ui = False

# auto reload flag
app.auto_reload = False

# stop signal for observer
app.stop_signal = False

# Whether the server is pointed to a directory containing a single config.
app.single_config_mode = False
app.single_config_id = None


class RequestBody(BaseModel):
    config_id: Optional[str] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", None),
        description="The id of the configuration to be used. If not set, the default configuration will be used.",
    )
    config_ids: Optional[List[str]] = Field(
        default=None,
        description="The list of configuration ids to be used. If set, the configurations will be combined.",
        # alias="guardrails",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The id of an existing thread to which the messages should be added.",
    )
    messages: Optional[List[dict]] = Field(
        default=None, description="The list of messages in the current conversation."
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data to be added to the conversation.",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent, like in ChatGPT. "
        "Tokens will be sent as data-only server-sent events as they become "
        "available, with the stream terminated by a data: [DONE] message.",
    )
    options: GenerationOptions = Field(
        default_factory=GenerationOptions,
        description="Additional options for controlling the generation.",
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction.",
    )
    # Standard OpenAI completion parameters
    model: Optional[str] = Field(
        default="main",
        description="The model to use for chat completion. Maps to the main model in the config.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature to use.",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter.",
    )
    stop: Optional[str] = Field(
        default=None,
        description="Stop sequences.",
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty parameter.",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty parameter.",
    )
    function_call: Optional[dict] = Field(
        default=None,
        description="Function call parameter.",
    )
    logit_bias: Optional[dict] = Field(
        default=None,
        description="Logit bias parameter.",
    )
    log_probs: Optional[bool] = Field(
        default=None,
        description="Log probabilities parameter.",
    )

    @root_validator(pre=True)
    def ensure_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("model") is not None and data.get("config_id") is None:
                data["config_id"] = data["model"]
            if data.get("config_id") is not None and data.get("config_ids") is not None:
                raise ValueError(
                    "Only one of config_id or config_ids should be specified"
                )
            if data.get("config_id") is None and data.get("config_ids") is not None:
                data["config_id"] = None
            if data.get("config_id") is None and data.get("config_ids") is None:
                warnings.warn(
                    "No config_id or config_ids provided, using default config_id"
                )
        return data

    @validator("config_ids", pre=True, always=True)
    def ensure_config_ids(cls, v, values):
        if v is None and values.get("config_id") and values.get("config_ids") is None:
            # populate config_ids with config_id if only config_id is provided
            return [values["config_id"]]
        return v


@app.get(
    "/v1/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Lists the currently available models, mapping guardrails configurations to OpenAI-compatible model format.",
)
async def get_models():
    """Returns the list of available models (guardrails configurations) in OpenAI-compatible format."""

    # Use the same logic as get_rails_configs to find available configurations
    if app.single_config_mode:
        config_ids = [app.single_config_id] if app.single_config_id else []

    else:
        config_ids = [
            f
            for f in os.listdir(app.rails_config_path)
            if os.path.isdir(os.path.join(app.rails_config_path, f))
            and f[0] != "."
            and f[0] != "_"
            # Filter out all the configs for which there is no `config.yml` file.
            and (
                os.path.exists(os.path.join(app.rails_config_path, f, "config.yml"))
                or os.path.exists(os.path.join(app.rails_config_path, f, "config.yaml"))
            )
        ]

    models = []
    for config_id in config_ids:
        try:
            # Load the RailsConfig to extract model information
            if app.single_config_mode:
                config_path = app.rails_config_path
            else:
                config_path = os.path.join(app.rails_config_path, config_id)

            rails_config = RailsConfig.from_path(config_path)
            # Extract all models from this config
            config_models = rails_config.models

            if len(config_models) == 0:
                guardrails_model = GuardrailsModel(
                    id=config_id,
                    object="model",
                    created=int(time.time()),
                    owned_by="nemo-guardrails",
                    guardrails_config_id=config_id,
                )
                models.append(guardrails_model)
            else:
                for model in config_models:
                    # Only include models with a model name
                    if model.model:
                        guardrails_model = GuardrailsModel(
                            id=model.model,
                            object="model",
                            created=int(time.time()),
                            owned_by="nemo-guardrails",
                            guardrails_config_id=config_id,
                        )
                        models.append(guardrails_model)
        except Exception as ex:
            log.warning(f"Could not load model info for config {config_id}: {ex}")
            continue

    return ModelsResponse(data=models)


@app.get(
    "/v1/rails/configs",
    summary="Get List of available rails configurations.",
)
async def get_rails_configs():
    """Returns the list of available rails configurations."""

    # In single-config mode, we return a single config.
    if app.single_config_mode:
        # And we use the name of the root folder as the id of the config.
        return [{"id": app.single_config_id}]

    # We extract all folder names as config names
    config_ids = [
        f
        for f in os.listdir(app.rails_config_path)
        if os.path.isdir(os.path.join(app.rails_config_path, f))
        and f[0] != "."
        and f[0] != "_"
        # We filter out all the configs for which there is no `config.yml` file.
        and (
            os.path.exists(os.path.join(app.rails_config_path, f, "config.yml"))
            or os.path.exists(os.path.join(app.rails_config_path, f, "config.yaml"))
        )
    ]

    return [{"id": config_id} for config_id in config_ids]


# One instance of LLMRails per config id
llm_rails_instances: dict[str, LLMRails] = {}
llm_rails_events_history_cache: dict[str, dict] = {}


def _generate_cache_key(config_ids: List[str]) -> str:
    """Generates a cache key for the given config ids."""

    return "-".join((config_ids))  # remove sorted


def _get_main_model_name(rails_config: RailsConfig) -> Optional[str]:
    """Extracts the main model name from a RailsConfig."""
    main_models = [m for m in rails_config.models if m.type == "main"]
    if main_models and main_models[0].model:
        return main_models[0].model
    return None


def _get_rails(config_ids: List[str]) -> LLMRails:
    """Returns the rails instance for the given config id."""

    # If we have a single config id, we just use it as the key
    configs_cache_key = _generate_cache_key(config_ids)

    if configs_cache_key in llm_rails_instances:
        return llm_rails_instances[configs_cache_key]

    # In single-config mode, we only load the main config directory
    if app.single_config_mode:
        if config_ids != [app.single_config_id]:
            raise ValueError(f"Invalid configuration ids: {config_ids}")

        # We set this to an empty string so tha when joined with the root path, we
        # get the same thing.
        config_ids = [""]

    full_llm_rails_config: Optional[RailsConfig] = None

    for config_id in config_ids:
        base_path = os.path.abspath(app.rails_config_path)
        full_path = os.path.normpath(os.path.join(base_path, config_id))

        # @NOTE: (Rdinu) Reject config_ids that contain dangerous characters or sequences
        if re.search(r"[\\/]|(\.\.)", config_id):
            raise ValueError("Invalid config_id.")

        if os.path.commonprefix([full_path, base_path]) != base_path:
            raise ValueError("Access to the specified path is not allowed.")

        rails_config = RailsConfig.from_path(full_path)

        if not full_llm_rails_config:
            full_llm_rails_config = rails_config
        else:
            full_llm_rails_config += rails_config

    if full_llm_rails_config is None:
        raise ValueError("No valid rails configuration found.")

    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    llm_rails_instances[configs_cache_key] = llm_rails

    # If we have a cache for the events, we restore it
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(
        configs_cache_key, {}
    )

    return llm_rails


async def _format_streaming_response(
    streaming_handler: StreamingHandler, model_name: Optional[str]
) -> AsyncIterator[str]:
    while True:
        try:
            chunk = await streaming_handler.__anext__()
        except StopAsyncIteration:
            # When the stream ends, yield the [DONE] message
            yield "data: [DONE]\n\n"
            break

        # Determine the payload format based on chunk type
        if isinstance(chunk, dict):
            # If chunk is a dict, wrap it in OpenAI chunk format with delta
            payload = {
                "id": None,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "delta": chunk,
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }
        elif isinstance(chunk, str):
            try:
                # Try parsing as JSON - if it parses, it might be a pre-formed payload
                payload = json.loads(chunk)
            except Exception:
                # treat as plain text content token
                payload = {
                    "id": None,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "delta": {"content": chunk},
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
        else:
            # For any other type, treat as plain content
            payload = {
                "id": None,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "delta": {"content": str(chunk)},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
            }

        # Send the payload as JSON
        data = json.dumps(payload, ensure_ascii=False)
        yield f"data: {data}\n\n"


@app.post(
    "/v1/chat/completions",
    response_model=ResponseBody,
    response_model_exclude_none=True,
)
async def chat_completion(body: RequestBody, request: Request):
    """Chat completion for the provided conversation.

    TODO: add support for explicit state object.
    """
    log.info("Got request for config %s", body.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(
            logger({"endpoint": "/v1/chat/completions", "body": body.json()})
        )

    # Save the request headers in a context variable.
    api_request_headers.set(request.headers)

    # Use Request config_ids if set, otherwise use the FastAPI default config.
    # If neither is available we can't generate any completions as we have no config_id
    config_ids = body.config_ids

    if not config_ids:
        if app.default_config_id:
            config_ids = [app.default_config_id]
        else:
            raise GuardrailsConfigurationError(
                "No request config_ids provided and server has no default configuration"
            )

    try:
        llm_rails = _get_rails(config_ids)

    except ValueError as ex:
        log.exception(ex)
        return ResponseBody(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=config_ids[0] if config_ids else "unknown",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        content=f"Could not load the {config_ids} guardrails configuration. "
                        f"An internal error has occurred.",
                        role="assistant",
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
        )

    try:
        main_model_name = _get_main_model_name(llm_rails.config)
        if main_model_name is None:
            main_model_name = config_ids[0] if config_ids else "unknown"

        messages = body.messages or []
        if body.context:
            messages.insert(0, {"role": "context", "content": body.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")
            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.thread_id) < 16:
                return ResponseBody(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=main_model_name,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                content="The `thread_id` must have a minimum length of 16 characters.",
                                role="assistant",
                            ),
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                )

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        generation_options = body.options

        # Initialize llm_params if not already set
        if generation_options.llm_params is None:
            generation_options.llm_params = {}

        # Set OpenAI-compatible parameters in llm_params
        if body.max_tokens:
            generation_options.llm_params["max_tokens"] = body.max_tokens
        if body.temperature is not None:
            generation_options.llm_params["temperature"] = body.temperature
        if body.top_p is not None:
            generation_options.llm_params["top_p"] = body.top_p
        if body.stop:
            generation_options.llm_params["stop"] = body.stop
        if body.presence_penalty is not None:
            generation_options.llm_params["presence_penalty"] = body.presence_penalty
        if body.frequency_penalty is not None:
            generation_options.llm_params["frequency_penalty"] = body.frequency_penalty

        if body.stream:
            # Use stream_async for streaming
            streaming_handler = llm_rails.stream_async(
                messages=messages,
                options=generation_options,
                state=body.state,
            )

            return StreamingResponse(
                _format_streaming_response(
                    streaming_handler, model_name=main_model_name
                ),
                media_type="text/event-stream",
            )
        else:
            res = await llm_rails.generate_async(
                messages=messages, options=generation_options, state=body.state
            )

            if isinstance(res, GenerationResponse):
                bot_message_content = res.response[0]
                # Ensure bot_message is always a dict
                if isinstance(bot_message_content, str):
                    bot_message = {"role": "assistant", "content": bot_message_content}
                else:
                    bot_message = bot_message_content
            else:
                assert isinstance(res, dict)
                bot_message = res

            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.thread_id and datastore is not None and datastore_key is not None:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            # Build the response with OpenAI-compatible format
            response_kwargs = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": main_model_name,
                "choices": [
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=bot_message["content"],
                        ),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
            }

            # If we have additional GenerationResponse fields, include them for backward compatibility
            if isinstance(res, GenerationResponse):
                response_kwargs["llm_output"] = res.llm_output
                response_kwargs["output_data"] = res.output_data
                response_kwargs["log"] = res.log
                response_kwargs["state"] = res.state

            return ResponseBody(**response_kwargs)

    except Exception as ex:
        log.exception(ex)
        return ResponseBody(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=config_ids[0] if config_ids else "unknown",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        content="Internal server error",
                        role="assistant",
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
        )


# By default, there are no challenges
challenges = []


def register_challenges(additional_challenges: List[dict]):
    """Register additional challenges

    Args:
        additional_challenges: The new challenges to be registered.
    """
    challenges.extend(additional_challenges)


@app.get(
    "/v1/challenges",
    summary="Get list of available challenges.",
)
async def get_challenges():
    """Returns the list of available challenges for red teaming."""

    return challenges


def register_datastore(datastore_instance: DataStore):
    """Registers a DataStore to be used by the server."""
    global datastore

    datastore = datastore_instance


def register_logger(logger: Callable):
    """Register an additional logger"""
    registered_loggers.append(logger)


def start_auto_reload_monitoring():
    """Start a thread that monitors the config folder for changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return None

                elif event.event_type == "created" or event.event_type == "modified":
                    log.info(
                        f"Watchdog received {event.event_type} event for file {event.src_path}"
                    )

                    # Compute the relative path
                    src_path_str = str(event.src_path)
                    rel_path = os.path.relpath(src_path_str, app.rails_config_path)

                    # The config_id is the first component
                    parts = rel_path.split(os.path.sep)
                    config_id = parts[0]

                    if (
                        not parts[-1].startswith(".")
                        and ".ipynb_checkpoints" not in parts
                        and os.path.isfile(src_path_str)
                    ):
                        # We just remove the config from the cache so that a new one is used next time
                        if config_id in llm_rails_instances:
                            instance = llm_rails_instances[config_id]
                            del llm_rails_instances[config_id]
                            if instance:
                                val = instance.events_history_cache
                                # We save the events history cache, to restore it on the new instance
                                llm_rails_events_history_cache[config_id] = val

                            log.info(
                                f"Configuration {config_id} has changed. Clearing cache."
                            )

        observer = Observer()
        event_handler = Handler()
        observer.schedule(event_handler, app.rails_config_path, recursive=True)
        observer.start()
        try:
            while not app.stop_signal:
                time.sleep(5)
        finally:
            observer.stop()
            observer.join()

    except ImportError:
        # Since this is running in a separate thread, we just print the error.
        print(
            "The auto-reload feature requires `watchdog`. Please install using `pip install watchdog`."
        )
        # Force close everything.
        os._exit(-1)


def set_default_config_id(config_id: str):
    app.default_config_id = config_id


class GuardrailsConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    pass


# # Register a nicer error message for 422 error
# def register_exception(app: FastAPI):
#     @app.exception_handler(RequestValidationError)
#     async def validation_exception_handler(
#         request: Request, exc: RequestValidationError
#     ):
#         exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
#         # or logger.error(f'{exc}')
#         log.error(request, exc_str)
#         content = {"status_code": 10422, "message": exc_str, "data": None}
#         return JSONResponse(
#             content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
#         )
#
#
# register_exception(app)
