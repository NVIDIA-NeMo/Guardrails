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

import argparse
import hashlib
from pathlib import Path

import yaml

NON_STRUCTURAL_KEYS = frozenset(
    {"description", "summary", "title", "example", "examples", "x-oaiMeta", "externalDocs", "tags"}
)


def structural_fields(value, location=()):
    """Remove documentation, never user-named members or literal schema values."""
    if location and location[-1] in {"default", "const", "enum"}:
        return value
    if isinstance(value, dict):
        named_members = bool(location) and (
            location[-1] in {"properties", "patternProperties", "$defs", "dependentSchemas"}
            or (len(location) >= 2 and location[-2] == "components")
        )
        return {
            key: structural_fields(child, location + (key,))
            for key, child in value.items()
            if named_members
            or key not in NON_STRUCTURAL_KEYS
            or (key == "description" and len(location) >= 2 and location[-2] == "responses")
        }
    if isinstance(value, list):
        return [structural_fields(child, location) for child in value]
    return value


def extract_chat_source(source_path: Path, descriptor_path: Path) -> dict:
    """Extract the pinned Chat operations and their reachable local components."""
    source_bytes = source_path.read_bytes()
    descriptor = yaml.safe_load(descriptor_path.read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != descriptor["sha256"]:
        raise ValueError("The OpenAI source does not match its pinned SHA-256 digest.")

    source = yaml.safe_load(source_bytes)
    paths = {
        path: structural_fields(value, ("paths", path))
        for path, value in source["paths"].items()
        if path == "/chat/completions" or path.startswith("/chat/completions/")
    }
    components = source["components"]
    references = set()

    def visit(value):
        if isinstance(value, dict):
            reference = value.get("$ref")
            if reference is not None and not reference.startswith("#/components/"):
                raise ValueError(f"Only local component references are supported: {reference}")
            if isinstance(reference, str) and reference.startswith("#/components/") and reference not in references:
                references.add(reference)
                _, _, section, name = reference.split("/", 3)
                visit(components[section][name])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(paths)
    selected_components = {}
    for section in components:
        names = sorted(
            reference.split("/", 3)[-1] for reference in references if reference.startswith(f"#/components/{section}/")
        )
        if names:
            selected_components[section] = {
                name: structural_fields(components[section][name], ("components", section, name)) for name in names
            }

    return {
        "openapi": source["openapi"],
        "info": {
            "title": source["info"]["title"],
            "version": source["info"]["version"],
            "license": source["info"]["license"],
        },
        "x-nemo-source": {"revision": descriptor["revision"], "sha256": descriptor["sha256"]},
        "paths": paths,
        "components": selected_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extracted = extract_chat_source(args.source, args.descriptor)
    args.output.write_text(yaml.safe_dump(extracted, sort_keys=False, allow_unicode=True), encoding="utf-8")


if __name__ == "__main__":
    main()
