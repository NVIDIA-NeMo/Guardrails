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

import logging
from typing import List, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter

from nemoguardrails.llm.providers import get_chat_provider_names
from nemoguardrails.utils import console

log = logging.getLogger(__name__)


def _list_providers() -> None:
    """List all available providers."""
    console.print("\n[bold]Chat Completion Providers:[/]")
    for provider in sorted(get_chat_provider_names()):
        console.print(f"  • {provider}")


def _get_provider_completions() -> List[str]:
    """Get list of available chat-completion providers."""
    return sorted(get_chat_provider_names())


def select_provider() -> Optional[str]:
    """Let user select a specific provider."""
    providers = _get_provider_completions()

    # session with fuzzy completion
    session = PromptSession()
    completer = FuzzyWordCompleter(providers)

    console.print("\n[bold]Available chat completion providers:[/] (type to filter, use arrows to select)")
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


def find_providers(
    list_only: bool = typer.Option(False, "--list", "-l", help="Just list all available providers"),
):
    """List and select LLM providers interactively.

    This command provides an interactive interface to explore and select LLM providers.

    When run without options:
    - Type to search for a specific provider
    - Use arrows to navigate and Tab to complete

    When run with --list:
    - Simply lists all available providers
    - No selection is made
    """
    if list_only:
        _list_providers()
        return

    provider = select_provider()
    if provider:
        typer.echo(f"\nSelected chat completion provider: {provider}")
    else:
        typer.echo("No provider selected.")
