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

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDES = tuple(
    ROOT / ".agents" / "skills" / name / "SKILL.md"
    for name in ("add-library-rail", "review-library-rail", "recorded-tests")
) + (
    ROOT / "nemoguardrails/library/AGENTS.md",
    ROOT / "tests/recorded/AGENTS.md",
    ROOT / "tests/recorded/README.md",
)
REFERENCE = re.compile(r"`((?:\.agents/skills|nemoguardrails|tests|scripts|schemas|docs)/[\w./-]+(?:::\w+)*)`")


def _references():
    for guide in GUIDES:
        for line_number, line in enumerate(guide.read_text().splitlines(), 1):
            for match in REFERENCE.finditer(line):
                yield pytest.param(
                    guide,
                    line_number,
                    match[1],
                    id=f"{guide.relative_to(ROOT)}:{line_number}:{match[1]}",
                )


def _definitions(body):
    definitions = {}
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            definitions[node.target.id] = node
    return definitions


@pytest.mark.parametrize(("guide", "line_number", "reference"), _references())
def test_rail_skill_references_resolve(guide, line_number, reference):
    filename, *symbols = reference.split("::")
    target = ROOT / filename
    location = f"{guide.relative_to(ROOT)}:{line_number}"
    assert target.exists(), f"{location}: missing repository path {filename}"
    if not symbols:
        return

    assert target.suffix == ".py", f"{location}: symbol reference requires a Python file"
    node = ast.parse(target.read_text(), filename=str(target))
    for symbol in symbols:
        definitions = _definitions(getattr(node, "body", []))
        assert symbol in definitions, f"{location}: missing Python definition in {reference}"
        node = definitions[symbol]
