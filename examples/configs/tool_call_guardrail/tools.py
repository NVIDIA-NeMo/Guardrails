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

"""Mock agent tools the guardrail sits in front of.

Stand-ins for real agent capabilities. They perform no authorization of their
own — that is exactly the point: the guard in `policy.py` decides whether a call
is permitted, and only then does dispatch reach these functions.
"""

from __future__ import annotations


def read_file(path: str) -> str:
    return f"<contents of {path}>"


def write_file(path: str, content: str = "") -> str:
    return f"wrote {len(content)} bytes to {path}"


def run_shell(command: str, timeout_seconds: float = 60) -> str:
    return f"ran {command!r} (timeout {timeout_seconds:g}s)"


def http_request(url: str, method: str = "GET") -> str:
    return f"{method} {url} -> 200"


def git_push(remote: str, branch: str) -> str:
    return f"pushed {branch} to {remote}"


def install_package(name: str, version: str = "latest") -> str:
    return f"installed {name}=={version}"


# Dispatch table mapping tool names (as referenced by policies) to implementations.
TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "http_request": http_request,
    "git_push": git_push,
    "install_package": install_package,
}
