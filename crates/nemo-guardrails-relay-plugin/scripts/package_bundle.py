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

"""Materialize the NVIDIA NeMo Guardrails plugin's bundle from a compiled cdylib."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

CRATE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CRATE_ROOT.parents[1]
PACKAGE_NAME = "nemo-guardrails-relay-plugin"
# LICENSE.md holds the license notice; LICENSE-Apache-2.0.txt holds the full license text.
LICENSE_FILES = ("LICENSE.md", "LICENSE-Apache-2.0.txt")
ARCHIVE_SUFFIXES = (".tar.gz", ".zip")
# Every archive entry records this time and no owner, so the same files always give the same archive bytes.
# It's the earliest time a zip entry can record: 1980-01-01T00:00:00Z.
ARCHIVE_TIMESTAMP = 315_532_800
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def digest(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def archive_problem(archive: Path, bundle: Path) -> Optional[str]:
    """Why *archive* can't hold a bundle written to *bundle*, or None when it can."""
    # Equal paths included: the output directory would be created where the archive goes.
    if bundle.is_relative_to(archive):
        return f"bundle output directory must not be the archive path or inside it: {bundle}"
    if archive.parent.is_relative_to(bundle):
        return f"bundle archive must be outside output directory: {archive}"
    if not archive.name.endswith(ARCHIVE_SUFFIXES):
        return "bundle archive must end in .tar.gz or .zip"
    if archive.exists():
        return f"bundle archive already exists: {archive}"
    return None


def archive_bundle(bundle: Path, archive: Path) -> None:
    """Archive a materialized bundle under the stable package directory name, as the same bytes for the same files."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.gz"):
        _write_tar_gz(bundle, archive)
    elif archive.suffix == ".zip":
        _write_zip(bundle, archive)
    else:
        raise ValueError("bundle archive must end in .tar.gz or .zip")


def _entry_mode(executable: bool) -> int:
    """The permissions an archive entry records, which depend only on whether it is executable."""
    return 0o755 if executable else 0o644


def _reproducible_tar_entry(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """*info* with the fixed time, no owner, and permissions that don't depend on the build host's umask."""
    info.mtime = ARCHIVE_TIMESTAMP
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = _entry_mode(info.isdir() or bool(info.mode & 0o111))
    return info


def _write_tar_gz(bundle: Path, archive: Path) -> None:
    """Write *bundle* to a new .tar.gz *archive*."""
    # The gzip header would also record the archive's name and the current time.
    with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=ARCHIVE_TIMESTAMP) as gz:
        with tarfile.open(fileobj=gz, mode="w") as stream:
            stream.add(bundle, arcname=PACKAGE_NAME, filter=_reproducible_tar_entry)


def _write_zip(bundle: Path, archive: Path) -> None:
    """Write *bundle* to a new .zip *archive*."""
    with zipfile.ZipFile(archive, "x") as stream:
        for path in sorted(bundle.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo((Path(PACKAGE_NAME) / path.relative_to(bundle)).as_posix(), date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Unix, the only system whose permission bits zip readers take from external_attr.
            info.create_system = 3
            info.external_attr = _entry_mode(bool(path.stat().st_mode & 0o111)) << 16
            stream.writestr(info, path.read_bytes())


def main() -> None:
    """Materialize a Relay-loadable plugin bundle in an empty directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    library = args.library.resolve()
    if not library.is_file():
        parser.error(f"compiled plugin library does not exist: {library}")

    manifest = (CRATE_ROOT / "relay-plugin.toml").read_text(encoding="utf-8")
    placeholders = ("<platform-library-file>", "<artifact-sha256>")
    missing = [placeholder for placeholder in placeholders if placeholder not in manifest]
    if missing:
        parser.error(f"plugin manifest is missing placeholders: {', '.join(missing)}")

    output = args.output.resolve()
    if output.exists() and not output.is_dir():
        parser.error(f"bundle output exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        parser.error(f"bundle output directory must be empty: {output}")

    archive = args.archive.resolve() if args.archive is not None else None
    # Checked before anything is written, so a bad --archive leaves --output as it was.
    problem = archive_problem(archive, output) if archive is not None else None
    if problem is not None:
        parser.error(problem)
    output.mkdir(parents=True, exist_ok=True)

    artifact = output / library.name
    shutil.copy2(library, artifact)
    shutil.copy2(CRATE_ROOT / "config.schema.json", output / "config.schema.json")
    for filename in LICENSE_FILES:
        shutil.copy2(REPOSITORY_ROOT / filename, output / filename)

    artifact_digest = digest(artifact)
    manifest = manifest.replace("<platform-library-file>", artifact.name)
    manifest = manifest.replace("<artifact-sha256>", artifact_digest)
    (output / "relay-plugin.toml").write_text(manifest, encoding="utf-8")

    if archive is None:
        print(output)
        return

    archive_bundle(output, archive)
    print(archive)


if __name__ == "__main__":
    main()
