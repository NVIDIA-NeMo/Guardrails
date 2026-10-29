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
MODULE_REFERENCE = re.compile(r"`(nemoguardrails(?:\.[A-Za-z_]\w*)*)`")
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
PYTHON_BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _module_reference(module, symbol=None):
    path = ROOT.joinpath(*module.split("."))
    filename = path.with_suffix(".py")
    if not filename.is_file() and path.is_dir():
        filename = path / "__init__.py"
    reference = str(filename.relative_to(ROOT))
    return f"{reference}::{symbol}" if symbol else reference


def _import_references(code):
    # Guides use single-line imports; surrounding examples can contain pseudocode.
    for line_number, line in enumerate(code.splitlines()):
        statement = line.strip()
        if not statement.startswith(("from nemoguardrails", "import nemoguardrails")):
            continue
        for node in ast.parse(statement).body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "nemoguardrails":
                for alias in node.names:
                    yield line_number, _module_reference(node.module, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "nemoguardrails":
                        yield line_number, _module_reference(alias.name)


def _markdown_references(text):
    for match in REFERENCE.finditer(text):
        yield text.count("\n", 0, match.start()) + 1, match[1]
    for match in MODULE_REFERENCE.finditer(text):
        yield text.count("\n", 0, match.start()) + 1, _module_reference(match[1])
    for pattern in (INLINE_CODE, PYTHON_BLOCK):
        for match in pattern.finditer(text):
            start_line = text.count("\n", 0, match.start(1)) + 1
            for offset, reference in _import_references(match[1]):
                yield start_line + offset, reference


def _references():
    for guide in GUIDES:
        for line_number, reference in _markdown_references(guide.read_text()):
            yield pytest.param(
                guide,
                line_number,
                reference,
                id=f"{guide.relative_to(ROOT)}:{line_number}:{reference}",
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
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                definitions[name] = node
        else:
            for field in ("body", "orelse", "finalbody", "handlers"):
                definitions.update(_definitions(getattr(node, field, [])))
    return definitions


def _assert_reference(reference, location):
    filename, *symbols = reference.split("::")
    target = ROOT / filename
    assert target.exists(), f"{location}: missing repository path {filename}"
    if not symbols:
        return

    assert target.suffix == ".py", f"{location}: symbol reference requires a Python file"
    node = ast.parse(target.read_text(), filename=str(target))
    for symbol in symbols:
        definitions = _definitions(getattr(node, "body", []))
        assert symbol in definitions, f"{location}: missing Python definition in {reference}"
        node = definitions[symbol]


@pytest.mark.parametrize(("guide", "line_number", "reference"), _references())
def test_rail_skill_references_resolve(guide, line_number, reference):
    _assert_reference(reference, f"{guide.relative_to(ROOT)}:{line_number}")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("`nemoguardrails.llm.call`", {"nemoguardrails/llm/call.py"}),
        (
            "`from nemoguardrails.llm.call import llm_call, warn_if_truncated`",
            {"nemoguardrails/llm/call.py::llm_call", "nemoguardrails/llm/call.py::warn_if_truncated"},
        ),
        (
            "```python\nfrom nemoguardrails.llm.call import llm_call, warn_if_truncated\n```",
            {"nemoguardrails/llm/call.py::llm_call", "nemoguardrails/llm/call.py::warn_if_truncated"},
        ),
        ("`import nemoguardrails`", {"nemoguardrails/__init__.py"}),
        ("`from nemoguardrails.actions import action`", {"nemoguardrails/actions/__init__.py::action"}),
        ("`from nemoguardrails import LLMRails`", {"nemoguardrails/__init__.py::LLMRails"}),
    ],
)
def test_python_module_and_import_references(text, expected):
    references = {reference for _, reference in _markdown_references(text)}
    assert references == expected
    for reference in references:
        _assert_reference(reference, "example")


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("`nemoguardrails.missing_module`", "missing repository path"),
        ("`from nemoguardrails.llm.call import missing_symbol`", "missing Python definition"),
    ],
)
def test_stale_python_import_references_fail(text, error):
    references = list(_markdown_references(text))
    assert references
    with pytest.raises(AssertionError, match=error):
        for _, reference in references:
            _assert_reference(reference, "example")


def test_import_references_are_checked_without_execution(tmp_path, monkeypatch):
    module = tmp_path / "nemoguardrails" / "example.py"
    module.parent.mkdir()
    module.write_text("raise RuntimeError('must not execute')\ndef check():\n    pass\n")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    references = list(_markdown_references("`from nemoguardrails.example import check`"))
    assert references == [(1, "nemoguardrails/example.py::check")]
    _assert_reference(references[0][1], "example")
