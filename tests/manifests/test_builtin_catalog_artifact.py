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

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemoguardrails.manifests import RailCatalog, registry
from scripts import generate_builtin_rail_catalog as catalog_generator


def test_generated_builtin_catalog_matches_source():
    artifact = registry.resources.files("nemoguardrails.manifests").joinpath("builtin_rails.json")

    assert artifact.read_text(encoding="utf-8") == catalog_generator.generate_builtin_rail_catalog()


def test_builtin_catalog_freshness_check_accepts_current_output(tmp_path, monkeypatch):
    output_path = tmp_path / "builtin_rails.json"
    current = '{"format_version": 1, "records": []}\n'
    output_path.write_text(current, encoding="utf-8")
    monkeypatch.setattr(catalog_generator, "generate_builtin_rail_catalog", lambda: current)

    catalog_generator.check_builtin_rail_catalog(output_path)


@pytest.mark.parametrize(
    ("current", "reason"),
    (
        (None, "missing"),
        ("{", "malformed"),
        ('{"format_version": 1, "records": []}\n', "stale"),
    ),
)
def test_builtin_catalog_freshness_check_fails_clearly(tmp_path, monkeypatch, current, reason):
    output_path = tmp_path / "builtin_rails.json"
    if current is not None:
        output_path.write_text(current, encoding="utf-8")
    monkeypatch.setattr(
        catalog_generator,
        "generate_builtin_rail_catalog",
        lambda: '{"format_version": 1, "records": [{}]}\n',
    )

    with pytest.raises(SystemExit) as exc_info:
        catalog_generator.check_builtin_rail_catalog(output_path)

    assert str(exc_info.value) == (f"{output_path} is {reason}.\nRun: make generate-builtin-rail-catalog")


@pytest.mark.parametrize(("current", "reason"), ((None, "missing"), ("{", "malformed")))
def test_generator_subprocess_recovers_invalid_artifact(tmp_path, current, reason):
    output_path = tmp_path / "builtin_rails.json"
    if current is not None:
        output_path.write_text(current, encoding="utf-8")
    script_path = Path(catalog_generator.__file__)

    check_result = subprocess.run(
        [sys.executable, str(script_path), "--check", "--output", str(output_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert check_result.returncode == 1
    assert f"{output_path} is {reason}." in check_result.stderr
    assert "Traceback" not in check_result.stderr

    generate_result = subprocess.run(
        [sys.executable, str(script_path), "--output", str(output_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert generate_result.returncode == 0, generate_result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 1
    assert payload["records"]


def test_runtime_catalog_does_not_scan_or_import_rail_modules(monkeypatch):
    registry._reset_rail_manifest_cache()
    imported_rail_modules = {name for name in sys.modules if name.endswith(".rail")}
    real_import_module = importlib.import_module

    def reject_scan(*args, **kwargs):
        raise AssertionError("runtime catalog loading performed source discovery")

    def reject_rail_import(name, package=None):
        if name.startswith("nemoguardrails.library.") and name.endswith(".rail"):
            raise AssertionError(f"runtime catalog loading imported {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(RailCatalog, "discover_built_ins", reject_scan)
    monkeypatch.setattr(importlib, "import_module", reject_rail_import)
    try:
        catalog = registry.default_rail_catalog()
    finally:
        registry._reset_rail_manifest_cache()

    assert catalog.records
    assert {name for name in sys.modules if name.endswith(".rail")} == imported_rail_modules


class _Resource:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    def joinpath(self, name):
        return self

    def read_text(self, encoding):
        if self.error is not None:
            raise self.error
        return self.content


@pytest.mark.parametrize(
    ("resource", "message"),
    (
        (_Resource(error=FileNotFoundError()), "is missing"),
        (_Resource("{"), "is malformed"),
        (_Resource("[]"), "must contain an object"),
        (_Resource('{"format_version": 2, "records": []}'), "unsupported format version"),
        (_Resource('{"format_version": 1}'), "must contain a records list"),
        (_Resource('{"format_version": 1, "records": [null]}'), "record 0 must be an object"),
        (
            _Resource('{"format_version": 1, "records": [{"manifest": {}, "source": "source"}]}'),
            "record 0 has an invalid manifest",
        ),
    ),
)
def test_invalid_builtin_catalog_artifacts_fail_clearly(monkeypatch, resource, message):
    monkeypatch.setattr(registry.resources, "files", lambda package: resource)

    with pytest.raises(RuntimeError, match=message):
        registry._load_builtin_catalog()
