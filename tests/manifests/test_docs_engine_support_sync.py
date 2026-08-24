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

"""Keeps the published rail support matrix honest about the shipped catalog.

The matrix in ``docs/reference/rail-engine-support.mdx`` is hand-maintained, so
adding a ``rail.py`` or lifting an IORails limitation would otherwise leave the
page quietly wrong. These tests read the page back and compare it against the
live catalog.
"""

import re
from pathlib import Path

import pytest

from nemoguardrails.guardrails.compiled_rail import unsupported_surface_reason
from nemoguardrails.manifests import RailDirection, default_rail_catalog

DOCS_PAGE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "rail-engine-support.mdx"

SUPPORTED = "✓"
UNSUPPORTED = "✗"

_SECTION_HEADINGS = {
    "### Input Surfaces": RailDirection.INPUT,
    "### Output Surfaces": RailDirection.OUTPUT,
    "### Retrieval Surfaces": RailDirection.RETRIEVAL,
}

# | `flow name` | Rail | LLMRails | IORails | Notes |
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|[^|]*\|\s*(\S+)\s*\|\s*(\S+)\s*\|")


def _documented_rows() -> dict[tuple[RailDirection, str], tuple[str, str]]:
    """Parse the matrix tables into {(direction, flow): (llmrails mark, iorails mark)}."""
    rows: dict[tuple[RailDirection, str], tuple[str, str]] = {}
    direction = None
    for line in DOCS_PAGE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped in _SECTION_HEADINGS:
            direction = _SECTION_HEADINGS[stripped]
            continue
        if stripped.startswith("#"):
            # Any other heading ends the table that the last surface heading opened.
            direction = None
            continue
        if direction is None:
            continue
        match = _ROW.match(stripped)
        if match is not None:
            flow, llmrails, iorails = match.groups()
            rows[(direction, flow)] = (llmrails, iorails)
    return rows


def _catalog_rows() -> dict[tuple[RailDirection, str], tuple[str, str]]:
    """Build the same mapping from the shipped manifests."""
    return {
        key: (SUPPORTED, UNSUPPORTED if unsupported_surface_reason(surface) else SUPPORTED)
        for key, surface in default_rail_catalog().surfaces().items()
    }


@pytest.fixture(scope="module")
def documented():
    return _documented_rows()


@pytest.fixture(scope="module")
def catalog():
    return _catalog_rows()


def test_matrix_lists_every_catalog_surface(documented, catalog):
    """Every surface the catalog declares appears in the documented matrix."""
    missing = sorted((direction.value, flow) for direction, flow in catalog.keys() - documented.keys())
    assert not missing, (
        f"{DOCS_PAGE.name} is missing {len(missing)} surface(s) the catalog declares: {missing}. "
        "Add a row for each to the matching direction table."
    )


def test_matrix_lists_no_surface_the_catalog_dropped(documented, catalog):
    """The documented matrix does not list a surface the catalog no longer declares."""
    stale = sorted((direction.value, flow) for direction, flow in documented.keys() - catalog.keys())
    assert not stale, (
        f"{DOCS_PAGE.name} documents {len(stale)} surface(s) the catalog does not declare: {stale}. "
        "Remove the rows, or fix the flow name if the surface was renamed."
    )


def test_matrix_engine_marks_match_the_catalog(documented, catalog):
    """Each documented row carries the support marks the catalog implies."""
    drifted = {
        (key[0].value, key[1]): {"documented": documented[key], "expected": marks}
        for key, marks in catalog.items()
        if key in documented and documented[key] != marks
    }
    assert not drifted, (
        f"{DOCS_PAGE.name} disagrees with the catalog on {len(drifted)} row(s), as (LLMRails, IORails): {drifted}"
    )


def test_summary_counts_match_the_catalog(catalog):
    """The support summary totals agree with the per-direction tables."""
    text = DOCS_PAGE.read_text(encoding="utf-8")
    for direction in RailDirection:
        total = sum(1 for key in catalog if key[0] is direction)
        servable = sum(1 for key, marks in catalog.items() if key[0] is direction and marks[1] == SUPPORTED)
        row = f"| {direction.value.capitalize()} | {total} | {total} | {servable} |"
        assert row in text, f"{DOCS_PAGE.name} is missing the summary row {row!r}"

    total = len(catalog)
    servable = sum(1 for marks in catalog.values() if marks[1] == SUPPORTED)
    assert f"| **Total** | **{total}** | **{total}** | **{servable}** |" in text
