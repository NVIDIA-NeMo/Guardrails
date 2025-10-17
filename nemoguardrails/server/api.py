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
import asyncio
import contextvars
import importlib.util
import json
import logging
import os.path
import re
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator, validator
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles

from nemoguardrails import LLMRails, RailsConfig, utils
from nemoguardrails.rails.llm.options import (
    GenerationLog,
    GenerationOptions,
    GenerationResponse,
)
from nemoguardrails.server.datastore.datastore import DataStore
from nemoguardrails.streaming import StreamingHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# The list of registered loggers. Can be used to send logs to various
# backends and storage engines.
registered_loggers = []

api_description = """Guardrails Sever API."""

# The headers for each request
api_request_headers = contextvars.ContextVar("headers")

# The datastore that the Server should use.
# This is currently used only for storing threads.
# TODO: refactor to wrap the FastAPI instance inside a RailsServer class
#  and get rid of all the global attributes.
datastore: Optional[DataStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)

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
        app.task = app.loop.run_in_executor(None, start_auto_reload_monitoring)

    yield

    # Shutdown logic here
    if app.auto_reload:
        app.stop_signal = True
        if hasattr(app, "task"):
            app.task.cancel()
        log.info("Shutting down file observer")
    else:
        pass


app = FastAPI(
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
        description="The list of configuration ids to be used. "
        "If set, the configurations will be combined.",
        # alias="guardrails",
        validate_default=True,
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The id of an existing thread to which the messages should be added.",
    )
    messages: List[dict] = Field(
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

    @root_validator(pre=True)
    def ensure_config_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
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


class ResponseBody(BaseModel):
    messages: List[dict] = Field(
        default=None, description="The new messages in the conversation"
    )
    llm_output: Optional[dict] = Field(
        default=None,
        description="Contains any additional output coming from the LLM.",
    )
    output_data: Optional[dict] = Field(
        default=None,
        description="The output data, i.e. a dict with the values corresponding to the `output_vars`.",
    )
    log: Optional[GenerationLog] = Field(
        default=None, description="Additional logging information."
    )
    state: Optional[dict] = Field(
        default=None,
        description="A state object that should be used to continue the interaction in the future.",
    )


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
llm_rails_instances = {}
llm_rails_events_history_cache = {}


def _generate_cache_key(config_ids: List[str]) -> str:
    """Generates a cache key for the given config ids."""

    return "-".join((config_ids))  # remove sorted


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

    full_llm_rails_config = None

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

    llm_rails = LLMRails(config=full_llm_rails_config, verbose=True)
    llm_rails_instances[configs_cache_key] = llm_rails

    # If we have a cache for the events, we restore it
    llm_rails.events_history_cache = llm_rails_events_history_cache.get(
        configs_cache_key, {}
    )

    return llm_rails


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

    config_ids = body.config_ids
    if not config_ids and app.default_config_id:
        config_ids = [app.default_config_id]
    elif not config_ids and not app.default_config_id:
        raise GuardrailsConfigurationError(
            "No 'config_id' provided and no default configuration is set for the server. "
            "You must set a 'config_id' in your request or set use --default-config-id when . "
        )
    try:
        llm_rails = _get_rails(config_ids)
    except ValueError as ex:
        log.exception(ex)
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Could not load the {config_ids} guardrails configuration. "
                    f"An internal error has occurred.",
                }
            ]
        }

    try:
        messages = body.messages
        if body.context:
            messages.insert(0, {"role": "context", "content": body.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")

            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.thread_id) < 16:
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "The `thread_id` must have a minimum length of 16 characters.",
                        }
                    ]
                }

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.thread_id
            thread_messages = json.loads(await datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        if (
            body.stream
            and llm_rails.config.streaming_supported
            and llm_rails.main_llm_supports_streaming
        ):
            # Create the streaming handler instance
            streaming_handler = StreamingHandler()

            # Start the generation
            asyncio.create_task(
                llm_rails.generate_async(
                    messages=messages,
                    streaming_handler=streaming_handler,
                    options=body.options,
                    state=body.state,
                )
            )

            # TODO: Add support for thread_ids in streaming mode

            return StreamingResponse(streaming_handler)
        else:
            res = await llm_rails.generate_async(
                messages=messages, options=body.options, state=body.state
            )

            if isinstance(res, GenerationResponse):
                bot_message = res.response[0]
            else:
                assert isinstance(res, dict)
                bot_message = res

            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.thread_id:
                await datastore.set(datastore_key, json.dumps(messages + [bot_message]))

            result = {"messages": [bot_message]}

            # If we have additional GenerationResponse fields, we return as well
            if isinstance(res, GenerationResponse):
                result["llm_output"] = res.llm_output
                result["output_data"] = res.output_data
                result["log"] = res.log
                result["state"] = res.state

            return result

    except Exception as ex:
        log.exception(ex)
        return {
            "messages": [{"role": "assistant", "content": "Internal server error."}]
        }


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


def register_logger(logger: callable):
    """Register an additional logger"""
    registered_loggers.append(logger)


def start_auto_reload_monitoring():
    """Start a thread that monitors the config folder for changes."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            @staticmethod
            def on_any_event(event):
                if event.is_directory:
                    return None

                elif event.event_type == "created" or event.event_type == "modified":
                    log.info(
                        f"Watchdog received {event.event_type} event for file {event.src_path}"
                    )

                    # Compute the relative path
                    rel_path = os.path.relpath(event.src_path, app.rails_config_path)

                    # The config_id is the first component
                    parts = rel_path.split(os.path.sep)
                    config_id = parts[0]

                    if (
                        not parts[-1].startswith(".")
                        and ".ipynb_checkpoints" not in parts
                        and os.path.isfile(event.src_path)
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
            "The auto-reload feature requires `watchdog`. "
            "Please install using `pip install watchdog`."
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
