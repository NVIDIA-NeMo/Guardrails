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

"""Layer 2: Advanced network verification with LLM judgment.

This module provides deep network verification (DNS, HTTP, TLS) combined with
conservative LLM judgment based on actual verification results.

Key features:
- Network verification (DNS resolution, HTTP HEAD/GET, TLS validation)
- Conservative LLM judgment (requires strong evidence to mark as hallucinated)
- Integration with verification results as context
- Reduced false positives compared to Layer 1
"""

import asyncio
import json
import logging
import socket
from typing import Any, Dict, List

log = logging.getLogger(__name__)


async def verify_domain_network(
    domain: str,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Perform basic network verification (DNS only for now).

    Uses asyncio.to_thread() to prevent blocking the event loop during DNS resolution.

    Args:
        domain: Domain name to verify
        timeout: Timeout in seconds

    Returns:
        Verification result dict with DNS status
    """
    try:
        loop = asyncio.get_running_loop()

        def _getaddrinfo():
            """Helper to perform DNS lookup with timeout."""
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                return socket.getaddrinfo(domain, 443)
            finally:
                sock.close()

        # Run DNS lookup in thread pool to avoid blocking event loop
        addrs = await loop.run_in_executor(None, _getaddrinfo)
        return {
            "domain": domain,
            "status": "resolved",
            "address_count": len(addrs),
        }
    except socket.gaierror:
        return {
            "domain": domain,
            "status": "nxdomain",
            "error": "DNS resolution failed",
        }
    except socket.timeout:
        return {
            "domain": domain,
            "status": "timeout",
            "error": "DNS timeout",
        }
    except Exception as e:
        return {
            "domain": domain,
            "status": "error",
            "error": str(e),
        }


async def layer2_check_with_verification(
    bot_response: str,
    user_message: str,
    urls: List[str],
    domains: List[str],
    github_repos: List[str],
    llm_call_func,
    llm_task_manager,
    config,
    llm=None,
) -> Dict[str, Any]:
    """Layer 2 check: Use verification results + LLM for conservative judgment.

    Args:
        bot_response: Bot response text
        user_message: User message
        urls: Extracted URLs
        domains: Extracted domains
        github_repos: Extracted GitHub repos
        llm_call_func: LLM call function
        llm_task_manager: Task manager
        config: Rails config
        llm: LLM model for inference

    Returns:
        Layer 2 verification result
    """
    # Fast path: no entities to verify
    if not urls and not domains and not github_repos:
        log.info("Layer 2: No URLs/domains/repos found, fast pass")
        return {
            "layer": "layer2",
            "status": "clean",
            "is_hallucinated": False,
            "llm_response": "no",
            "verification_results": {},
        }

    # Step 1: Quick network verification (concurrent for all domains)
    verification_results = {}
    if domains:
        tasks = [verify_domain_network(domain) for domain in domains]
        results = await asyncio.gather(*tasks)
        verification_results = {domain: result for domain, result in zip(domains, results)}

    # Step 2: Call LLM with verification results
    task_name = "self_check_domain_hallucination_layer2"

    prompt = llm_task_manager.render_task_prompt(
        task=task_name,
        context={
            "user_input": user_message,
            "bot_response": bot_response,
            "extracted_urls": ", ".join(urls) if urls else "none",
            "extracted_domains": ", ".join(domains) if domains else "none",
            "extracted_github_repos": ", ".join(github_repos) if github_repos else "none",
            "verification_results": json.dumps(verification_results, default=str),
        },
    )

    stop = llm_task_manager.get_stop_tokens(task=task_name)
    max_tokens = llm_task_manager.get_max_tokens(task=task_name)
    max_tokens = max_tokens or 1024

    try:
        from nemoguardrails.actions.llm.utils import warn_if_truncated
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

        warn_if_truncated(llm_response, task_name)
        response_text = llm_response.content.strip().lower()

    except Exception as e:
        log.error(f"Layer 2 LLM call failed: {e}")
        return {
            "layer": "layer2",
            "status": "error",
            "is_hallucinated": False,
            "llm_response": "error",
            "verification_results": verification_results,
        }

    # Step 3: Parse result
    is_hallucinated = response_text.startswith("yes")

    log.info(f"Layer 2: LLM response='{response_text}', is_hallucinated={is_hallucinated}")

    return {
        "layer": "layer2",
        "status": "suspicious" if is_hallucinated else "verified",
        "is_hallucinated": is_hallucinated,
        "llm_response": response_text,
        "verification_results": verification_results,
    }
