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
import os
from enum import Enum
from typing import Any, List, Literal, Optional

import typer
import uvicorn
from fastapi import FastAPI

from nemoguardrails import __version__
from nemoguardrails.actions_server import actions_server
from nemoguardrails.cli.chat import run_chat
from nemoguardrails.cli.migration import migrate
from nemoguardrails.cli.providers import _list_providers, select_provider_with_type
from nemoguardrails.eval import cli as eval_cli
from nemoguardrails.logging.verbose import set_verbose
from nemoguardrails.utils import init_random_seed


class ColangVersions(str, Enum):
    one = "1.0"
    two_alpha = "2.0-alpha"


_COLANG_VERSIONS = [version.value for version in ColangVersions]


app = typer.Typer()

app.add_typer(
    eval_cli.app, name="eval", short_help="Evaluation a guardrail configuration."
)
app.pretty_exceptions_enable = False

logging.getLogger().setLevel(logging.WARNING)


@app.command()
def chat(
    config: List[str] = typer.Option(
        default=["config"],
        exists=True,
        help="Path to a directory containing configuration files to use. "
        "Can also point to a single configuration file.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="If the chat should be verbose and output detailed logging information.",
    ),
    verbose_no_llm: bool = typer.Option(
        default=False,
        help="If the chat should be verbose and exclude the prompts and responses for the LLM calls.",
    ),
    verbose_simplify: bool = typer.Option(
        default=False,
        help="Simplify further the verbose output.",
    ),
    debug_level: List[str] = typer.Option(
        default=[],
        help="Enable debug mode which prints rich information about the flows execution. "
        "Available levels: WARNING, INFO, DEBUG",
    ),
    streaming: bool = typer.Option(
        default=False,
        help="If the chat should use the streaming mode, if possible.",
    ),
    server_url: Optional[str] = typer.Option(
        default=None,
        help="If specified, the chat CLI will interact with a server, rather than load the config. "
        "In this case, the --config-id must also be specified.",
    ),
    config_id: Optional[str] = typer.Option(
        default=None, help="The config_id to be used when interacting with the server."
    ),
):
    """Start an interactive chat session."""
    if len(config) > 1:
        typer.secho("Multiple configurations are not supported.", fg=typer.colors.RED)
        typer.echo("Please provide a single folder.")
        raise typer.Exit(1)

    # We enable verbose mode automatically when a debug level is specified.
    # If the `--verbose-no-llm` mode is used, we activate the verbose mode as well.
    # This means that the user doesn't have to use both options at the same time.
    verbose = verbose or verbose_no_llm or len(debug_level) > 0

    if len(debug_level) > 0 or os.environ.get("DEBUG_MODE"):
        init_random_seed(0)

    if verbose:
        set_verbose(
            True,
            llm_calls=not verbose_no_llm,
            debug=len(debug_level) > 0,
            debug_level=debug_level[0] if debug_level else "INFO",
            simplify=verbose_simplify,
        )

    run_chat(
        config_path=config[0],
        verbose=verbose,
        verbose_llm_calls=not verbose_no_llm,
        streaming=streaming,
        server_url=server_url,
        config_id=config_id,
    )


@app.command()
def server(
    port: int = typer.Option(
        default=8000, help="The port that the server should listen on. "
    ),
    config: List[str] = typer.Option(
        default=[],
        exists=True,
        help="Path to a directory containing multiple configuration sub-folders.",
    ),
    default_config_id: Optional[str] = typer.Option(
        default=None,
        help="The default configuration to use when no config is specified.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="If the server should be verbose and output detailed logs including prompts.",
    ),
    disable_chat_ui: bool = typer.Option(
        default=False,
        help="Weather the ChatUI should be disabled",
    ),
    auto_reload: bool = typer.Option(default=False, help="Enable auto reload option."),
    prefix: str = typer.Option(
        default="",
        help="A prefix that should be added to all server paths. Should start with '/'.",
    ),
):
    """Start a NeMo Guardrails server."""

    from nemoguardrails.server import api

    if config:
        # We make sure there is no trailing separator, as that might break things in
        # single config mode.
        setattr(
            api.app,
            "rails_config_path",
            os.path.expanduser(config[0].rstrip(os.path.sep)),
        )
    else:
        # If we don't have a config, we try to see if there is a local config folder
        local_path = os.getcwd()
        local_configs_path = os.path.join(local_path, "config")

        if os.path.exists(local_configs_path):
            setattr(api.app, "rails_config_path", local_configs_path)

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    if disable_chat_ui:
        setattr(api.app, "disable_chat_ui", True)

    if auto_reload:
        setattr(api.app, "auto_reload", True)

    if prefix:
        server_app = FastAPI()
        server_app.mount(prefix, api.app)
    else:
        server_app = api.app

    if default_config_id:
        api.set_default_config_id(default_config_id)  # Call function

    uvicorn.run(server_app, port=port, log_level="info", host="0.0.0.0")


@app.command()
def convert(
    path: str = typer.Argument(
        ..., help="The path to the file or directory to migrate."
    ),
    from_version: ColangVersions = typer.Option(
        default=ColangVersions.one,
        help=f"The version of the colang files to migrate from. Available options: {_COLANG_VERSIONS}.",
    ),
    verbose: bool = typer.Option(
        default=False,
        help="If the migration should be verbose and output detailed logs.",
    ),
    validate: bool = typer.Option(
        default=False,
        help="If the migration should validate the output using Colang Parser.",
    ),
    use_active_decorator: bool = typer.Option(
        default=True,
        help="If the migration should use the active decorator.",
    ),
    include_main_flow: bool = typer.Option(
        default=True,
        help="If the migration should add a main flow to the config.",
    ),
):
    """Convert Colang files and configs from older version to the latest."""

    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    absolute_path = os.path.abspath(path)

    # Typer CLI args have to use an enum, not literal. Convert to Literal here
    from_version_literal: Literal["1.0", "2.0-alpha"] = from_version.value

    migrate(
        path=absolute_path,
        include_main_flow=include_main_flow,
        use_active_decorator=use_active_decorator,
        from_version=from_version_literal,
        validate=validate,
    )


@app.command("actions-server")
def action_server(
    port: int = typer.Option(
        default=8001, help="The port that the server should listen on. "
    ),
):
    """Start a NeMo Guardrails actions server."""

    uvicorn.run(actions_server.app, port=port, log_level="info", host="0.0.0.0")


@app.command()
def find_providers(
    list_only: bool = typer.Option(
        False, "--list", "-l", help="Just list all available providers"
    ),
):
    """List and select LLM providers interactively.

    This command provides an interactive interface to explore and select LLM providers.
    It supports both text completion and chat completion model providers.

    When run without options:
    1. First, you'll be prompted to select a provider type:
    - Type to filter between "text completion" and "chat completion"
    - Use arrow keys to navigate through matches
    - Press Tab to autocomplete
    - Press Enter to select

    2. Then, you'll be prompted to select a specific provider:
    - Type to filter through available providers
    - Use arrow keys to navigate through matches
    - Press Tab to autocomplete
    - Press Enter to select

    When run with --list:
    - Simply lists all available providers
    - No selection is made
    """
    if list_only:
        _list_providers()
        return

    result = select_provider_with_type()
    if result:
        provider_type, provider = result
        typer.echo(f"\nSelected {provider_type} provider: {provider}")
    else:
        typer.echo("No provider selected.")


def version_callback(value: bool):
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def cli(
    _: Optional[bool] = typer.Option(
        None, "-v", "--version", callback=version_callback, is_eager=True
    ),
):
    pass
