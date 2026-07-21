#!/usr/bin/env python3
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

"""OpenAI API conformance analyzer for NeMo Guardrails.

Compares the NeMo Guardrails API spec against the upstream OpenAI spec using
oasdiff. Supports two spec sources for the guardrails side:

  --guardrails-spec fern/openapi.yml   (default, the hand-written docs spec)
  --fastapi                            (exports app.openapi() at runtime)

The OpenAI spec can be a local file or fetched from upstream:

  --openai-spec /path/to/spec.yml      (local file)
  --fetch                              (download from openai/openai-openapi)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

OPENAI_SPEC_URL = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"

MATCH_PATH = "/(chat/completions|models)"


def _run_oasdiff(subcommand: str, base: str, revision: str, *, extra_flags: list[str] | None = None) -> Any:
    cmd = [
        "oasdiff",
        subcommand,
        base,
        revision,
        "--format",
        "json",
        "--auto-upgrade",
        "--flatten-allof",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"oasdiff {subcommand} failed: {result.stderr}")
    return json.loads(result.stdout) if result.stdout else {}


def _check_breaking(spec: str) -> bool:
    if not Path(spec).exists():
        return True
    result = subprocess.run(
        [
            "oasdiff",
            "breaking",
            f"HEAD:{spec}",
            spec,
            "--fail-on",
            "ERR",
            "--auto-upgrade",
            "--flatten-allof",
            "--match-path",
            "^/v1/",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout or result.stderr)
        return False
    return True


def _fetch_openai_spec(url: str = OPENAI_SPEC_URL) -> Path:
    tmp = Path(tempfile.mktemp(suffix=".yml"))
    with urlopen(url) as resp:  # noqa: S310
        tmp.write_bytes(resp.read())
    return tmp


def _export_fastapi_spec() -> Path:
    import yaml

    from nemoguardrails.server.api import app

    spec = app.openapi()
    tmp = Path(tempfile.mktemp(suffix=".yml"))
    tmp.write_text(yaml.dump(spec, default_flow_style=False, sort_keys=False))
    return tmp


def analyze(
    openai_spec: Path,
    guardrails_spec: Path,
    match_path: str | None = None,
    strip_prefix: str = "/v1",
) -> dict[str, Any]:
    import yaml

    flags = ["--strip-prefix-revision", strip_prefix]
    if match_path:
        flags.extend(["--match-path", match_path])
    changelog = _run_oasdiff("changelog", str(openai_spec), str(guardrails_spec), extra_flags=flags)

    changes = [
        {k: v for k, v in entry.items() if k not in ("baseSource", "revisionSource", "fingerprint")}
        for entry in (changelog if isinstance(changelog, list) else [])
        if entry.get("section") == "paths"
    ]
    changes.sort(key=lambda e: (e.get("path", ""), e.get("operation", ""), e.get("text", "")))

    spec_text = openai_spec.read_text()
    spec_data = yaml.safe_load(spec_text) if openai_spec.suffix in (".yml", ".yaml") else json.loads(spec_text)

    return {
        "openai_version": spec_data.get("info", {}).get("version", "unknown"),
        "changes": changes,
    }


def _print_report(report: dict[str, Any], *, verbose: bool = False) -> None:
    ver = report["openai_version"]
    changes = report["changes"]
    print(f"OpenAI v{ver}: {len(changes)} conformance gap(s)")

    if not verbose:
        return

    current_endpoint = ""
    for c in changes:
        endpoint = f"{c.get('operation', '?')} {c.get('path', '?')}"
        if endpoint != current_endpoint:
            current_endpoint = endpoint
            print(f"  {endpoint}:")
        print(f"    {c['text']}")


def main():
    parser = argparse.ArgumentParser(description="OpenAI API conformance analyzer for NeMo Guardrails")
    parser.add_argument(
        "--openai-spec",
        type=Path,
        default=None,
        help="Path to OpenAI spec file (default: fetched from upstream if --fetch)",
    )
    parser.add_argument(
        "--guardrails-spec",
        type=Path,
        default=Path("fern/openapi.yml"),
        help="Path to guardrails spec (ignored when --fastapi is set)",
    )
    parser.add_argument(
        "--fastapi",
        action="store_true",
        help="Use app.openapi() as the guardrails spec instead of a file",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch the OpenAI spec from upstream instead of using a local file",
    )
    parser.add_argument("--match-path", type=str, default=MATCH_PATH)
    parser.add_argument("--quiet", action="store_true", help="Only print the summary line")
    parser.add_argument(
        "--check-breaking",
        action="store_true",
        help="Fail on breaking API changes vs HEAD (fern spec only)",
    )
    args = parser.parse_args()

    openai_spec = args.openai_spec
    guardrails_spec = args.guardrails_spec
    tmp_files: list[Path] = []

    try:
        if args.fetch:
            if openai_spec is not None:
                parser.error("--fetch and --openai-spec are mutually exclusive")
            openai_spec = _fetch_openai_spec()
            tmp_files.append(openai_spec)
        elif openai_spec is None:
            parser.error("either --openai-spec or --fetch is required")

        if args.fastapi:
            guardrails_spec = _export_fastapi_spec()
            tmp_files.append(guardrails_spec)

        if args.check_breaking and not args.fastapi:
            print("Breaking changes (HEAD vs working tree):", end=" ")
            if _check_breaking(str(args.guardrails_spec)):
                print("none")
            else:
                sys.exit(1)

        report = analyze(openai_spec, guardrails_spec, match_path=args.match_path)
        source = "app.openapi()" if args.fastapi else str(args.guardrails_spec)
        print(f"[{source}]", end=" ")
        _print_report(report, verbose=not args.quiet)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        for f in tmp_files:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
