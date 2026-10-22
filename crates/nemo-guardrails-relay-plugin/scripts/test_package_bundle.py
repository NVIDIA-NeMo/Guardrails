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

"""Tests for package_bundle.py: archive paths are checked before writing, and archives are reproducible."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

import pytest

SCRIPT = Path(__file__).with_name("package_bundle.py")


def _package(library: Path, output: Path, archive: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run the packaging script as the release pipeline does."""
    command = [sys.executable, str(SCRIPT), "--library", str(library), "--output", str(output)]
    if archive is not None:
        command += ["--archive", str(archive)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A stand-in for the compiled plugin library."""
    path = tmp_path / "libnemo_guardrails_relay_plugin.so"
    path.write_bytes(b"not really a library")
    return path


@pytest.mark.parametrize("archive_name, exists", [("bundle.tgz", False), ("bundle.tar.gz", True)])
def test_unusable_archive_is_refused_before_the_bundle_is_written(tmp_path, library, archive_name, exists):
    """An archive path with an unknown suffix, or one that exists, is refused before the output directory is written."""
    archive = tmp_path / archive_name
    if exists:
        archive.write_bytes(b"")
    output = tmp_path / "bundle"

    result = _package(library, output, archive)

    assert result.returncode == 2, result.stderr
    assert not output.exists() or not any(output.iterdir())


@pytest.mark.parametrize(
    "output_name, archive_name",
    [
        ("bundle.zip", "bundle.zip"),
        ("bundle.tar.gz/inner", "bundle.tar.gz"),
        ("bundle", "bundle/bundle.zip"),
    ],
    ids=["archive-at-output", "output-inside-archive", "archive-inside-output"],
)
def test_archive_overlapping_the_output_is_refused_before_writing(tmp_path, library, output_name, archive_name):
    """An archive path that is, contains, or sits inside the output directory is refused before anything is written."""
    output = tmp_path / output_name
    archive = tmp_path / archive_name

    result = _package(library, output, archive)

    assert result.returncode == 2, result.stderr
    assert not output.exists()
    assert not archive.exists()


@pytest.mark.parametrize("suffix", [".tar.gz", ".zip"])
def test_archives_of_the_same_files_are_identical(tmp_path, library, suffix):
    """Two archives of the same files have the same bytes, whatever the files' modification times."""
    digests = []
    for run, mtime in enumerate([1_000_000_000, 1_700_000_000]):
        os.utime(library, (mtime, mtime))
        archive = tmp_path / f"run{run}{suffix}"

        result = _package(library, tmp_path / f"bundle{run}", archive)

        assert result.returncode == 0, result.stderr
        digests.append(hashlib.sha256(archive.read_bytes()).hexdigest())

    assert digests[0] == digests[1]


def test_tar_archive_records_no_build_host_owner(tmp_path, library):
    """The .tar.gz archive records no user or group of the machine that built it."""
    archive = tmp_path / "bundle.tar.gz"

    result = _package(library, tmp_path / "bundle", archive)

    assert result.returncode == 0, result.stderr
    with tarfile.open(archive) as stream:
        owners = {(member.uid, member.gid, member.uname, member.gname) for member in stream.getmembers()}
    assert owners == {(0, 0, "", "")}
