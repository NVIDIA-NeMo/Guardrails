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
from pathlib import Path

import pytest

LIBRARY_ROOT = Path(__file__).parents[2] / "nemoguardrails" / "library"
FORBIDDEN_HTTP_MODULES = frozenset({"aiohttp", "httpx", "requests", "urllib3"})


def _direct_http_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.partition(".")[0])
    return imports & FORBIDDEN_HTTP_MODULES


@pytest.mark.unit
def test_library_uses_canonical_http_boundary():
    violations = {
        str(path.relative_to(LIBRARY_ROOT)): sorted(imports)
        for path in sorted(LIBRARY_ROOT.rglob("*.py"))
        if (imports := _direct_http_imports(path))
    }

    assert violations == {}
