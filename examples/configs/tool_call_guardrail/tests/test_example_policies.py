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

"""Behavioral contract for the example guards.

`VULNERABLE_GUARD` is the deliberately-gappy "before" state the scanner
discovers; `HARDENED_GUARD` is the "after" state with the human-approved output
of the scanner->synthesis pipeline folded in (the guard the rails serve). These
tests pin both so the before/after story can't silently drift.
"""

from example_policies import HARDENED_GUARD, PRINCIPALS, VULNERABLE_GUARD
from policy import Principal, ToolCall


def _authorize(guard, principal_id, tool, **args):
    principal = PRINCIPALS.get(principal_id, Principal(principal_id))
    return guard.authorize(ToolCall(tool, args), principal)


# --- The "before" gaps the scanner is meant to surface ---------------------


def test_vulnerable_guard_has_the_known_gaps():
    # write_file is unpoliced -> default-deny (the coverage gap, not a real control).
    assert not _authorize(VULNERABLE_GUARD, "dev-alice", "write_file", path="/etc/hosts").allowed
    # http_request is unpoliced -> default-deny.
    assert not _authorize(VULNERABLE_GUARD, "dev-alice", "http_request", url="http://169.254.169.254/").allowed
    # run_shell ceiling is loose: 600s passes the old 3600 ceiling.
    assert _authorize(VULNERABLE_GUARD, "dev-alice", "run_shell", command="x", timeout_seconds=600).allowed
    # git_push has no ownership check: dev-bob can push to a remote it does not own.
    assert _authorize(VULNERABLE_GUARD, "dev-bob", "git_push", remote="origin", branch="main").allowed


# --- The "after" controls, one finding at a time ---------------------------


def test_hardened_read_file_rejects_malformed_path():
    assert _authorize(HARDENED_GUARD, "dev-alice", "read_file", path="src/app.py").allowed
    assert not _authorize(HARDENED_GUARD, "dev-alice", "read_file", path="app.py; rm -rf /").allowed


def test_hardened_run_shell_tightened_ceiling():
    assert _authorize(HARDENED_GUARD, "dev-alice", "run_shell", command="x", timeout_seconds=100).allowed
    assert not _authorize(HARDENED_GUARD, "dev-alice", "run_shell", command="x", timeout_seconds=600).allowed


def test_hardened_git_push_requires_owned_remote():
    assert _authorize(HARDENED_GUARD, "dev-alice", "git_push", remote="origin", branch="main").allowed
    assert not _authorize(HARDENED_GUARD, "dev-bob", "git_push", remote="origin", branch="main").allowed


def test_hardened_install_package_blocks_denylisted_name():
    assert _authorize(HARDENED_GUARD, "dev-alice", "install_package", name="numpy", version="2.0").allowed
    assert not _authorize(HARDENED_GUARD, "dev-alice", "install_package", name="leftpad-evil", version="1.0").allowed


def test_hardened_http_request_blocks_metadata_egress():
    assert _authorize(HARDENED_GUARD, "dev-alice", "http_request", url="https://api.example.com/v1").allowed
    assert not _authorize(HARDENED_GUARD, "dev-alice", "http_request", url="http://169.254.169.254/latest").allowed


def test_hardened_write_file_requires_prefix_and_stepup():
    # dev-alice is elevated and writes inside her owned workspace prefix -> allowed.
    assert _authorize(HARDENED_GUARD, "dev-alice", "write_file", path="/workspace/alice/notes.txt").allowed
    # Outside the owned prefix -> blocked (prefix-ownership rule).
    assert not _authorize(HARDENED_GUARD, "dev-alice", "write_file", path="/etc/hosts").allowed
    # dev-bob writes inside his own prefix but is NOT elevated -> blocked (step-up rule).
    assert not _authorize(HARDENED_GUARD, "dev-bob", "write_file", path="/workspace/bob/x.txt").allowed
