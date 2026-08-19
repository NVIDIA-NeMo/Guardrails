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
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

FORMAT_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "nemoguardrails"
OUTPUT_PATH = PACKAGE_ROOT / "manifests/builtin_rails.json"


def _load_rail_catalog():
    if "nemoguardrails" not in sys.modules:
        package = ModuleType("nemoguardrails")
        setattr(package, "__path__", [str(PACKAGE_ROOT)])
        sys.modules["nemoguardrails"] = package

    return importlib.import_module("nemoguardrails.manifests").RailCatalog


def generate_builtin_rail_catalog() -> str:
    catalog = _load_rail_catalog().discover_built_ins()
    records = [
        {
            "manifest": record.manifest.model_dump(mode="json"),
            "source": record.source,
        }
        for _, record in sorted(catalog.records.items())
    ]
    return (
        json.dumps(
            {"format_version": FORMAT_VERSION, "records": records},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def check_builtin_rail_catalog(output_path: Path = OUTPUT_PATH) -> None:
    try:
        current = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"{output_path} is missing.\nRun: make generate-builtin-rail-catalog") from None

    try:
        json.loads(current)
    except json.JSONDecodeError:
        raise SystemExit(f"{output_path} is malformed.\nRun: make generate-builtin-rail-catalog") from None

    if current != generate_builtin_rail_catalog():
        raise SystemExit(f"{output_path} is stale.\nRun: make generate-builtin-rail-catalog")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if args.check:
        check_builtin_rail_catalog(args.output)
    else:
        args.output.write_text(generate_builtin_rail_catalog(), encoding="utf-8")


if __name__ == "__main__":
    main()
