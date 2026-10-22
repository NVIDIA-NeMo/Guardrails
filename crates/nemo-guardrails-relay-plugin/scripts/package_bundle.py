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

"""Materialize the NeMo Relay plugin bundle from a compiled cdylib."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import zipfile
from pathlib import Path

CRATE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CRATE_ROOT.parents[1]
PACKAGE_NAME = "nemo-guardrails-relay-plugin"
# LICENSE.md holds the license notice; LICENSE-Apache-2.0.txt holds the full license text.
LICENSE_FILES = ("LICENSE.md", "LICENSE-Apache-2.0.txt")


def digest(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def archive_bundle(bundle: Path, archive: Path) -> None:
    """Archive a materialized bundle under the stable package directory name."""
    if archive.parent.is_relative_to(bundle):
        raise ValueError(f"bundle archive must be outside output directory: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise ValueError(f"bundle archive already exists: {archive}")

    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(bundle, arcname=PACKAGE_NAME)
        return

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(bundle.rglob("*")):
                if path.is_file():
                    stream.write(path, Path(PACKAGE_NAME) / path.relative_to(bundle))
        return

    raise ValueError("bundle archive must end in .tar.gz or .zip")


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
    if archive is not None and archive.parent.is_relative_to(output):
        parser.error(f"bundle archive must be outside output directory: {archive}")
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

    try:
        archive_bundle(output, archive)
    except ValueError as error:
        parser.error(str(error))
    print(archive)


if __name__ == "__main__":
    main()
