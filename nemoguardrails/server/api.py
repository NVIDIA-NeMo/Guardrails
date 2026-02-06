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
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pydantic import BaseModel, ValidationError
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.rails.llm.config import Model
from nemoguardrails.rails.llm.options import GenerationResponse
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.server.schemas.openai import (
    GuardrailsChatCompletion,
    GuardrailsChatCompletionRequest,
)
from nemoguardrails.server.schemas.utils import (
    create_error_chat_completion,
    extract_bot_message_from_response,
    format_streaming_chunk_as_sse,
    generation_response_to_chat_completion,
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
    if os.path.exists(os.path.join(app.rails_config_path, "config.yml")) or os.path.exists(
        os.path.join(app.rails_config_path, "config.yaml")
    ):
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


def _generate_cache_key(config_ids: List[str], model_name: Optional[str] = None) -> str:
    """Generates a cache key for the given config ids and model name."""
    key = "-".join(config_ids)
    if model_name:
        key = f"{key}:{model_name}"
    return key


def _update_models_in_config(config: RailsConfig, main_model: Model) -> RailsConfig:
    """Update the main model in the RailsConfig.

    If a model with type="main" exists, it replaces it. Otherwise, adds it.
    """
    models = config.models.copy()
    main_model_index = None

    for index, model in enumerate(models):
        if model.type == main_model.type:
            main_model_index = index
            break

    if main_model_index is not None:
        parameters = {**models[main_model_index].parameters, **main_model.parameters}
        models[main_model_index] = main_model
        models[main_model_index].parameters = parameters
    else:
        models.append(main_model)

    return config.model_copy(update={"models": models})


def _get_rails(config_ids: List[str], model_name: Optional[str] = None) -> LLMRails:
    """Returns the rails instance for the given config id and model.

    Args:
        config_ids: List of configuration IDs to load
        model_name: The model name from the request (overrides config's main model)
    """
    configs_cache_key = _generate_cache_key(config_ids, model_name)

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

    if model_name:
        engine = os.environ.get("MAIN_MODEL_ENGINE")
        if not engine:
            engine = "openai"
            log.warning("MAIN_MODEL_ENGINE not set, defaulting to 'openai'. ")

        parameters = {}
        base_url = os.environ.get("MAIN_MODEL_BASE_URL")
        if base_url:
            parameters["base_url"] = base_url

        main_model = Model(model=model_name, type="main", engine=engine, parameters=parameters)
        full_llm_rails_config = _update_models_in_config(full_llm_rails_config, main_model)

    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    llm_rails_instances[configs_cache_key] = llm_rails

    # If we have a cache for the events, we restore it
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(configs_cache_key, {})

    return llm_rails


class ChunkErrorMetadata(BaseModel):
    message: str
    type: str
    param: str
    code: str


class ChunkError(BaseModel):
    error: ChunkErrorMetadata


async def _format_streaming_response(
    stream_iterator: AsyncIterator[Union[str, dict]], model_name: str
) -> AsyncIterator[str]:
    """
    Format streaming chunks from LLMRails.stream_async() as SSE events.

    Args:
        stream_iterator: AsyncIterator from stream_async() that yields str or dict chunks
        model_name: The model name to include in the chunks

    Yields:
        SSE-formatted strings (data: {...}\n\n)
    """
    # Use "unknown" as default if model_name is None
    model = model_name or "unknown"
    chunk_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        async for chunk in stream_iterator:
            # Format the chunk as SSE using the utility function
            processed_chunk = process_chunk(chunk)
            if isinstance(processed_chunk, ChunkError):
                # Yield the error and stop streaming
                yield f"data: {json.dumps(processed_chunk.model_dump())}\n\n"
                return
            else:
                yield format_streaming_chunk_as_sse(processed_chunk, model, chunk_id)

    finally:
        # Always send [DONE] event when stream ends
        yield "data: [DONE]\n\n"


def process_chunk(chunk: Any) -> Union[Any, ChunkError]:
    """
    Processes a single chunk from the stream.

    Args:
        chunk: A single chunk from the stream (can be str, dict, or other type).
        model: The model name (not used in processing but kept for signature consistency).

    Returns:
        Union[Any, StreamingError]: StreamingError instance for errors or the original chunk.
    """
    # Convert chunk to string for JSON parsing if needed
    chunk_str = chunk if isinstance(chunk, str) else json.dumps(chunk) if isinstance(chunk, dict) else str(chunk)

    try:
        validated_data = ChunkError.model_validate_json(chunk_str)
        return validated_data  # Return the StreamingError instance directly
    except ValidationError:
        # Not an error, just a normal token
        pass
    except json.JSONDecodeError:
        # Invalid JSON format, treat as normal token
        pass
    except Exception as e:
        log.warning(
            f"Unexpected error processing stream chunk: {type(e).__name__}: {str(e)}",
            extra={"chunk": chunk_str},
        )

    # Return the original chunk
    return chunk


@app.post(
    "/v1/chat/completions",
    response_model=GuardrailsChatCompletion,
    response_model_exclude_none=True,
)
async def chat_completion(body: GuardrailsChatCompletionRequest, request: Request):
    """Chat completion for the provided conversation.

    TODO: add support for explicit state object.
    """
    log.info("Got request for config %s", body.guardrails.config_id)
    for logger in registered_loggers:
        asyncio.get_event_loop().create_task(logger({"endpoint": "/v1/chat/completions", "body": body.json()}))

    # Save the request headers in a context variable.
    api_request_headers.set(request.headers)

    # Use Request config_ids if set, otherwise use the FastAPI default config.
    # If neither is available we can't generate any completions as we have no config_id
    config_ids = body.guardrails.config_ids

    if not config_ids:
        if app.default_config_id:
            config_ids = [app.default_config_id]
        else:
            raise HTTPException(
                status_code=422,
                detail="No guardrails config_id provided and server has no default configuration",
            )

    try:
        llm_rails = _get_rails(config_ids, model_name=body.model)

    except ValueError as ex:
        log.exception(ex)
        return create_error_chat_completion(
            model=body.model,
            error_message=f"Could not load the {config_ids} guardrails configuration. An internal error has occurred.",
            config_id=config_ids[0] if config_ids else None,
        )

    try:
        messages = body.messages or []
        if body.guardrails.context:
            messages.insert(0, {"role": "context", "content": body.guardrails.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.guardrails.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")
            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.guardrails.thread_id) < 16:
                return create_error_chat_completion(
                    model=body.model,
                    error_message="The `thread_id` must have a minimum length of 16 characters.",
                    config_id=config_ids[0] if config_ids else None,
                )

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.guardrails.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        generation_options = body.guardrails.options

        # Validate state format if provided
        if body.guardrails.state is not None and body.guardrails.state != {}:
            if "events" not in body.guardrails.state and "state" not in body.guardrails.state:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid state format: state must contain 'events' or 'state' key. Use an empty dict {} to start a new conversation.",
                )

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
            # Use stream_async for streaming with output rails support
            stream_iterator = llm_rails.stream_async(
                messages=messages,
                options=generation_options,
                state=body.guardrails.state,
            )

            return StreamingResponse(
                _format_streaming_response(stream_iterator, model_name=body.model),
                media_type="text/event-stream",
            )
        else:
            res = await llm_rails.generate_async(
                messages=messages,
                options=generation_options,
                state=body.guardrails.state,
            )

            # Extract bot message for thread storage if needed
            bot_message = extract_bot_message_from_response(res)

            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.guardrails.thread_id and datastore is not None and datastore_key is not None:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            # Build the response with OpenAI-compatible format using utility function
            if isinstance(res, GenerationResponse):
                return generation_response_to_chat_completion(
                    response=res,
                    model=body.model,
                    config_id=config_ids[0] if config_ids else None,
                )
            else:
                # For dict responses, convert to basic chat completion
                return GuardrailsChatCompletion(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content=bot_message.get("content", ""),
                            ),
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                )

    except HTTPException:
        raise
    except Exception as ex:
        log.exception(ex)
        return create_error_chat_completion(
            model=body.model,
            error_message="Internal server error",
            config_id=config_ids[0] if config_ids else None,
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
                    log.info(f"Watchdog received {event.event_type} event for file {event.src_path}")

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

                            log.info(f"Configuration {config_id} has changed. Clearing cache.")

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
        print("The auto-reload feature requires `watchdog`. Please install using `pip install watchdog`.")
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
