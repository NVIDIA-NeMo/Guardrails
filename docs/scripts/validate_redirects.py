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

"""Validate that all redirect targets in conf.py resolve to existing source files.

Usage:
    python docs/scripts/validate_redirects.py          # from repo root
    python scripts/validate_redirects.py               # from docs/
    python scripts/validate_redirects.py --verbose      # show each check
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def extract_redirects(conf_path: Path) -> dict[str, str]:
    """Parse conf.py with AST and extract the redirects dict (no imports needed)."""
    tree = ast.parse(conf_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "redirects":
                    return ast.literal_eval(node.value)
    return {}


def resolve_target(docs_dir: Path, html_target: str) -> Path | None:
    """Return the source file for an HTML redirect target, or None if missing."""
    stem = html_target.removesuffix(".html")

    candidates = [
        docs_dir / f"{stem}.md",
        docs_dir / f"{stem}.rst",
        docs_dir / f"{stem}/index.md",
        docs_dir / f"{stem}/index.rst",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Case-insensitive fallback (README.md vs readme)
    parent = docs_dir / Path(stem).parent
    name = Path(stem).name
    if parent.is_dir():
        for entry in parent.iterdir():
            if entry.stem.lower() == name.lower() and entry.suffix in (".md", ".rst"):
                return entry

    return None


def main() -> int:
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Resolve docs/ directory regardless of where the script is invoked from
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts":
        docs_dir = script_dir.parent
    else:
        docs_dir = script_dir

    conf_path = docs_dir / "conf.py"
    if not conf_path.exists():
        print(f"Error: {conf_path} not found", file=sys.stderr)
        return 1

    redirects = extract_redirects(conf_path)
    if not redirects:
        print("No redirects found in conf.py")
        return 1

    broken: list[tuple[str, str]] = []

    for source, target in sorted(redirects.items()):
        if target.startswith(("http://", "https://")):
            if verbose:
                print(f"  SKIP (external) {source} → {target}")
            continue

        resolved = resolve_target(docs_dir, target)
        if resolved:
            if verbose:
                print(f"  OK   {source} → {resolved.relative_to(docs_dir)}")
        else:
            broken.append((source, target))
            if verbose:
                print(f"  FAIL {source} → {target}")

    print(f"\nChecked {len(redirects)} redirects.")

    if broken:
        print(f"\n{len(broken)} broken redirect(s):\n")
        for source, target in broken:
            print(f"  {source}")
            print(f"    → {target}  (no matching source file)\n")
        return 1

    print("All redirect targets resolved to existing source files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
