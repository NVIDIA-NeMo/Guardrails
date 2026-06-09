# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 1: Fast LLM-based domain hallucination detection.

This module provides a lightweight, zero-network-dependency layer that
uses LLM judgment to detect hallucinated URLs, domains, and GitHub repos.
It is designed to be the default first-pass check.

Key characteristics:
- Zero external network calls (only LLM calls)
- Fast (~100-200ms, LLM-bound)
- No dependency on DNS, HTTP, TLS, or other network libraries
- Extracts entities via simple regex patterns
- Returns simple yes/no judgment

The LLM evaluates five dimensions:
1. Domain Existence - is this a well-known, real domain?
2. GitHub Verification - is this a real repository?
3. URL Path Plausibility - does the path look realistic?
4. Typosquatting - does it resemble a known name with misspelling?
5. Suspicious Patterns - are there red flags?
"""

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity Extraction Helpers (Zero External Dependencies)
# ---------------------------------------------------------------------------

_URL_RE = re.compile(
    r"(?:https?://[^\s<>()\[\]{}\"']+|www\.[^\s<>()\[\]{}\"']+)",
    re.IGNORECASE,
)

_GITHUB_RESERVED = frozenset(
    {
        "features",
        "topics",
        "search",
        "marketplace",
        "pricing",
        "login",
        "signup",
        "explore",
        "collections",
        "events",
        "sponsors",
        "about",
        "enterprise",
        "trending",
        "new",
        "organizations",
        "settings",
        "notifications",
        "codespaces",
        "site",
        "contact",
        "orgs",
        "users",
        "apps",
        "readme",
    }
)

_TRAILING = " \t\r\n.。，,;；:：!！?？)）]】}>\"'`"


def _clean(raw: str) -> str:
    """Clean and normalize a URL string."""
    return raw.strip().strip("`'\"").rstrip(_TRAILING)


def extract_urls(text: str) -> List[str]:
    """Extract and normalize URLs from text.

    Returns a list of deduplicated URLs with proper http(s) prefix.
    """
    seen: set = set()
    urls: List[str] = []
    for m in _URL_RE.finditer(text):
        url = _clean(m.group(0))
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def extract_domains(urls: List[str]) -> List[str]:
    """Extract unique domain names from URLs.

    Removes www. prefix and returns deduplicated hostnames.
    """
    seen: set = set()
    domains: List[str] = []
    for url in urls:
        host = (urlparse(url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in seen:
            seen.add(host)
            domains.append(host)
    return domains


def extract_github_repos(urls: List[str]) -> List[str]:
    """Extract GitHub repository references from URLs.

    Returns a list of "owner/repo" strings found in github.com URLs.
    Filters out reserved GitHub path segments.
    """
    seen: set = set()
    repos: List[str] = []
    for url in urls:
        parsed = urlparse(url)
        if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
            continue
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            continue
        owner = parts[0]
        repo = parts[1].removesuffix(".git")
        if owner.lower() in _GITHUB_RESERVED:
            continue
        key = f"{owner}/{repo}"
        if key not in seen:
            seen.add(key)
            repos.append(key)
    return repos


def extract_entities(text: str) -> Dict[str, Any]:
    """Extract all external entities from text.

    Returns a dict with:
    - urls: list of extracted URLs
    - domains: list of extracted domain names
    - github_repos: list of "owner/repo" references
    - has_entities: bool indicating if any entities were found
    """
    urls = extract_urls(text)
    domains = extract_domains(urls)
    github_repos = extract_github_repos(urls)
    return {
        "urls": urls,
        "domains": domains,
        "github_repos": github_repos,
        "has_entities": bool(urls or domains or github_repos),
    }


# ---------------------------------------------------------------------------
# Layer 1 LLM Check
# ---------------------------------------------------------------------------


async def layer1_check_domain_hallucination(
    bot_response: str,
    user_message: str,
    llm_call_func,
    llm_task_manager,
    config,
    llm=None,
) -> Dict[str, Any]:
    """Perform Layer 1 (fast LLM-based) domain hallucination check.

    Args:
        bot_response: The bot's response text to check
        user_message: The user's original message (for context)
        llm_call_func: Async function to call the LLM (typically llm_call)
        llm_task_manager: LLMTaskManager for rendering prompts
        config: RailsConfig object

    Returns:
        Dict with structure:
        {
            "layer": "layer1",
            "entities": {...},  # extracted entities
            "llm_response": "yes" | "no",
            "is_hallucinated": bool,
            "status": "clean" | "suspicious" | "error",
        }
    """
    _MAX_TOKENS = 1024

    # Fast path: no external references at all
    entities = extract_entities(bot_response)
    if not entities["has_entities"]:
        log.info("Layer 1: No URLs/domains found, passing.")
        return {
            "layer": "layer1",
            "entities": entities,
            "llm_response": "no",
            "is_hallucinated": False,
            "status": "clean",
        }

    log.debug(
        f"Layer 1: Found {len(entities['urls'])} URLs, "
        f"{len(entities['domains'])} domains, "
        f"{len(entities['github_repos'])} GitHub repos"
    )

    # Render Layer 1 prompt
    # Use custom task name for domain hallucination
    task_name = "self_check_domain_hallucination"

    prompt = llm_task_manager.render_task_prompt(
        task=task_name,
        context={
            "user_input": user_message,
            "bot_response": bot_response,
            "extracted_urls": ", ".join(entities["urls"]) if entities["urls"] else "none",
            "extracted_domains": ", ".join(entities["domains"]) if entities["domains"] else "none",
            "extracted_github_repos": (", ".join(entities["github_repos"]) if entities["github_repos"] else "none"),
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=task_name)
    max_tokens = llm_task_manager.get_max_tokens(task=task_name)
    max_tokens = max_tokens or _MAX_TOKENS

    # Call LLM
    try:
        from nemoguardrails.context import llm_call_info_var
        from nemoguardrails.logging.explain import LLMCallInfo

        llm_call_info_var.set(LLMCallInfo(task=task_name))

        llm_response = await llm_call_func(
            llm,
            prompt,
            stop=stop,
            llm_params={
                "temperature": config.lowest_temperature,
                "max_tokens": max_tokens,
            },
        )

        from nemoguardrails.actions.llm.utils import warn_if_truncated

        warn_if_truncated(llm_response, task_name)

        response_text = llm_response.content.strip().lower()

    except Exception as e:
        log.error(f"Layer 1 LLM call failed: {e}")
        # Default to "no" on error (permissive)
        return {
            "layer": "layer1",
            "entities": entities,
            "llm_response": "error",
            "is_hallucinated": False,
            "status": "error",
            "error": str(e),
        }

    # Parse output
    is_hallucinated = response_text.startswith("yes")

    log.info(f"Layer 1: LLM response='{response_text}', is_hallucinated={is_hallucinated}")

    return {
        "layer": "layer1",
        "entities": entities,
        "llm_response": response_text,
        "is_hallucinated": is_hallucinated,
        "status": "suspicious" if is_hallucinated else "clean",
    }
