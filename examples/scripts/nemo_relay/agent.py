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

"""A Deep Agents agent whose every tool call NeMo Relay checks with the NeMo Guardrails hooks server.

Relay's NeMo Guardrails plugin does the check. From the repository root, start the hooks server, then the agent with
the plugin's bundle:

    nemoguardrails hooks-server --config examples/scripts/nemo_relay/configs/policy
    python examples/scripts/nemo_relay/agent.py "Why is checkout failing? Check the logs and runbooks." \
        --plugin-bundle <bundle directory>
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import nemo_relay
import relay_plugin
from deepagents import create_deep_agent
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from deny_feedback import DenyFeedbackMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from nemo_relay.integrations.deepagents import NemoRelayDeepAgentsCallbackHandler, add_nemo_relay_integration

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

_RUNBOOKS = {
    "checkout-5xx": "Checkout 5xx: read the last 200 lines of the checkout log, then check the payments upstream.",
    "payments-timeout": "Payments timeout: confirm with GET https://status.internal.example/payments.",
}

_LOG = [
    f"2026-11-02T03:{minute:02d}:00Z checkout ERROR upstream payments timed out after 3000ms" for minute in range(60)
]


@tool
def search_runbooks(query: str) -> str:
    """Search the team's runbooks, returning each runbook whose name or text matches the query."""
    words = query.lower().split()
    hits = {name: text for name, text in _RUNBOOKS.items() if any(word in f"{name} {text}".lower() for word in words)}
    return json.dumps(hits or {"result": "no runbook matches"})


@tool
def read_log(service: str, lines: int) -> str:
    """Read the last `lines` lines of a service's log. Policy allows at most 500 lines per read."""
    if service != "checkout":
        return f"no log for service {service!r}"
    return "\n".join(_LOG[-lines:]) if lines > 0 else ""


@tool
def http_request(method: str, url: str, body: str = "") -> str:
    """Make an HTTP request. Policy allows only GET to https://*.internal.example. Sandboxed: nothing is sent."""
    host = urlparse(url).hostname
    return json.dumps({"status": 200, "method": method, "host": host, "body": "payments: degraded"})


TOOLS = [search_runbooks, read_log, http_request]


def build_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    system_prompt: str,
    name: str = "relay-guarded-agent",
    on_rejection: Optional[Callable[[str, str], None]] = None,
    **deep_agent_kwargs: Any,
) -> Any:
    """A Deep Agents agent with Relay on every tool call, its general-purpose subagent's included.

    Deep Agents doesn't give its built-in general-purpose subagent the main agent's middleware, and Relay's
    integration only instruments subagents passed to it, so the subagent is passed explicitly here. Without it,
    the subagent's tool calls would run unchecked. *on_rejection* is called with the tool name and reason of each
    call a guardrail rejects.
    """
    general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "middleware": [DenyFeedbackMiddleware(on_rejection=on_rejection)]}
    kwargs = add_nemo_relay_integration(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        name=name,
        # DenyFeedback first, so it wraps Relay and sees its rejections.
        middleware=[DenyFeedbackMiddleware(on_rejection=on_rejection)],
        subagents=[general_purpose, *deep_agent_kwargs.pop("subagents", [])],
        position="inner",
        **deep_agent_kwargs,
    )
    return create_deep_agent(**kwargs)


def print_rejection(tool_name: str, reason: str) -> None:
    print(f"  [guardrail] DENY {tool_name}: {reason}")


async def run(
    task: str, *, hooks_url: str, plugin_bundle: Path, model: BaseChatModel, timeout: float = 25.0
) -> Optional[str]:
    """Run *task* with every tool call checked by the hooks server at *hooks_url*; the agent's final answer."""
    async with relay_plugin.activate(plugin_bundle, hooks_url, timeout_millis=round(timeout * 1000)):
        agent = build_agent(
            model,
            TOOLS,
            system_prompt=(
                "You are an on-call assistant. Use the tools to investigate. If a tool call is blocked by "
                "NeMo Guardrails, read the reason, adjust the call to comply, and continue."
            ),
            on_rejection=print_rejection,
        )
        with nemo_relay.scope.scope("relay-guarded-agent", nemo_relay.ScopeType.Agent):
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]},
                config={"callbacks": [NemoRelayDeepAgentsCallbackHandler()], "recursion_limit": 60},
            )
    return result["messages"][-1].content


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task")
    parser.add_argument("--hooks-url", default="http://127.0.0.1:8100", help="The hooks server's base URL.")
    parser.add_argument(
        "--plugin-bundle",
        type=Path,
        default=os.environ.get("NEMO_GUARDRAILS_RELAY_PLUGIN_BUNDLE"),
        help="The NeMo Guardrails Relay plugin's bundle directory (default: $NEMO_GUARDRAILS_RELAY_PLUGIN_BUNDLE).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="A build.nvidia.com model with tool calling.")
    parser.add_argument("--timeout", type=float, default=25.0, help="Seconds to wait for each check.")
    args = parser.parse_args(argv)
    if args.plugin_bundle is None:
        parser.error("--plugin-bundle is required (or set NEMO_GUARDRAILS_RELAY_PLUGIN_BUNDLE)")

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    try:
        answer = asyncio.run(
            run(
                args.task,
                hooks_url=args.hooks_url,
                plugin_bundle=args.plugin_bundle,
                model=ChatNVIDIA(model=args.model),
                timeout=args.timeout,
            )
        )
    except RuntimeError as exc:
        if relay_plugin.failed_check_tool(exc) is None:
            raise
        print(f"Stopped, failing closed: {exc}", file=sys.stderr)
        return 2
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
