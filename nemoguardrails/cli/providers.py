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
from typing import List, Literal, Optional, Tuple, cast

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter

from nemoguardrails.llm.providers import get_chat_provider_names, get_llm_provider_names
from nemoguardrails.utils import console

log = logging.getLogger(__name__)


ProviderType = Literal["text completion", "chat completion"]


def _list_providers() -> None:
    """List all available providers."""
    console.print("\n[bold]Text Completion Providers:[/]")
    for provider in sorted(get_llm_provider_names()):
        console.print(f"  • {provider}")

    console.print("\n[bold]Chat Completion Providers:[/]")
    for provider in sorted(get_chat_provider_names()):
        console.print(f"  • {provider}")


def _get_provider_completions(
    provider_type: Optional[ProviderType] = None,
) -> List[str]:
    """Get list of providers based on type."""
    if provider_type == "text completion":
        return sorted(get_llm_provider_names())
    elif provider_type == "chat completion":
        return sorted(get_chat_provider_names())
    return []


def select_provider_type() -> Optional[ProviderType]:
    """Let user select between text completion and chat completion providers."""
    provider_types = ["chat completion", "text completion"]

    # session with fuzzy completion
    session = PromptSession()
    completer = FuzzyWordCompleter(provider_types)

    console.print(
        "\n[bold]Available Provider Types:[/] (type to filter, use arrows to select)"
    )
    for provider_type in provider_types:
        console.print(f"  • {provider_type}")

    try:
        result = session.prompt(
            "\nSelect provider type: ",
            completer=completer,
            complete_while_typing=True,
        ).strip()

        # None for empty input
        if not result:
            return None

        # exact match only
        if result in provider_types:
            return cast(ProviderType, result)  # type: ignore

        # fuzzy match
        matches = [t for t in provider_types if result.lower() in t.lower()]
        if len(matches) == 1:
            return matches[0]  # type: ignore

        return None
    except (EOFError, KeyboardInterrupt):
        return None


def select_provider(
    provider_type: Optional[ProviderType] = None,
) -> Optional[str]:
    """Let user select a specific provider based on the type."""
    providers = _get_provider_completions(provider_type)

    # session with fuzzy completion
    session = PromptSession()
    completer = FuzzyWordCompleter(providers)

    console.print(
        f"\n[bold]Available {provider_type} providers:[/] (type to filter, use arrows to select)"
    )
    for provider in providers:
        console.print(f"  • {provider}")

    try:
        result = session.prompt(
            "\nSelect provider: ",
            completer=completer,
            complete_while_typing=True,
        ).strip()

        # Return None for empty input
        if not result:
            return None

        # Return exact match only
        if result in providers:
            return result

        # Try fuzzy match
        matches = [p for p in providers if result.lower() in p.lower()]
        if len(matches) == 1:
            return matches[0]

        return None
    except (EOFError, KeyboardInterrupt):
        return None


def select_provider_with_type() -> Optional[Tuple[str, str]]:
    """Let user select both provider type and specific provider."""
    provider_type = select_provider_type()
    if not provider_type:
        return None

    provider = select_provider(provider_type)
    if not provider:
        return None

    return (provider_type, provider)


def find_providers(
    list_only: bool = typer.Option(
        False, "--list", "-l", help="Just list all available providers"
    ),
):
    """List and select LLM providers interactively.

    This command provides an interactive interface to explore and select LLM providers.
    It supports both text completion and chat completion providers.

    When run without options:
    - Type to search for provider type (text/chat completion)
    - Type to search for specific provider
    - Use arrows to navigate and Tab to complete

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
