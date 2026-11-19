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

"""Load NeMo Guardrails' native Relay plugin, which checks every tool call Relay manages with the hooks server.

The plugin is `crates/nemo-guardrails-relay-plugin`, packaged as a bundle by its `scripts/package_bundle.py`.
"""

import json
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import nemo_relay

PLUGIN_ID = "nvidia.nemo_guardrails"

# Relay raises a plain RuntimeError when the plugin's check fails (an unreachable server, a timeout, an error status),
# and the call doesn't run. A deny is also a plain RuntimeError, prefixed "guardrail rejected: " (see deny_feedback).
_FAILED_CHECK = re.compile(r"internal error: NeMo Guardrails check failed for tool call '(?P<tool>[^']*)': ")


def plugins_toml(bundle: Path, endpoint: str, *, timeout_millis: int = 25000) -> str:
    """A `plugins.toml` that loads the plugin from *bundle* and points it at the hooks server at *endpoint*.

    Bundles aren't signed, so the plugin is allowed integrity-only attestation, and a plugin that fails to load stops
    activation instead of leaving tool calls unchecked.
    """
    manifest = Path(bundle).resolve() / "relay-plugin.toml"
    return (
        "version = 1\n\n"
        f'[plugins.policy.overrides."{PLUGIN_ID}"]\n'
        'startup = "required"\n'
        'attestation = "integrity_only"\n\n'
        "[[plugins.dynamic]]\n"
        f"manifest = {json.dumps(str(manifest))}\n\n"
        "[plugins.dynamic.config]\n"
        f"endpoint = {json.dumps(endpoint)}\n"
        f"timeout_millis = {int(timeout_millis)}\n"
    )


@asynccontextmanager
async def activate(bundle: Path, endpoint: str, *, timeout_millis: int = 25000) -> AsyncIterator[None]:
    """Check every tool call Relay manages with the hooks server at *endpoint*, until the block exits."""
    if not (Path(bundle) / "relay-plugin.toml").is_file():
        raise FileNotFoundError(f"no NeMo Guardrails Relay plugin bundle at {bundle}")
    with tempfile.TemporaryDirectory(prefix="nemo-relay-plugins-") as directory:
        path = Path(directory) / "plugins.toml"
        path.write_text(plugins_toml(bundle, endpoint, timeout_millis=timeout_millis))
        async with nemo_relay.plugin.activate({}, str(path)) as activation:
            report = activation.report
            loaded = {entry["plugin_id"] for entry in report["dynamic_plugins"] if entry["selected"]}
            if PLUGIN_ID not in loaded:
                raise RuntimeError(f"NeMo Guardrails plugin not loaded: {report['config']['diagnostics']}")
            yield


def failed_check_tool(exc: BaseException) -> Optional[str]:
    """The tool whose check failed, when *exc* is Relay's error for a failed plugin check; otherwise None."""
    if type(exc) is not RuntimeError:
        return None
    match = _FAILED_CHECK.match(str(exc))
    return match["tool"] if match else None
