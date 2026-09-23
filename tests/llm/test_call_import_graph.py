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

"""The LLM call path must not depend on the Colang runtime.

``llm_call`` is the shared entry point every built-in rail action uses to reach a model.
Engines without a Colang runtime execute those same actions, so the module defining
``llm_call`` must be reachable without importing ``nemoguardrails.colang``.

This is checked statically rather than by inspecting ``sys.modules``: importing any
submodule runs ``nemoguardrails/__init__.py``, which loads ``RailsConfig`` and with it
both Colang parsers, so a runtime check could never observe the property under test. The
static walk asks the question that actually matters — what does *this module* declare a
dependency on, transitively, through first-party imports.
"""

import ast
from pathlib import Path
from typing import List, Optional, Set, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "nemoguardrails"
PACKAGE = "nemoguardrails"
FORBIDDEN_PREFIX = "nemoguardrails.colang"


def _in_scope_rail_action_modules() -> List[str]:
    """Action modules backing an input or output surface in the built-in rail catalog.

    Driven off the manifest catalog rather than a hand-kept list, so a newly added rail
    is covered the moment its manifest lands. Retrieval surfaces are excluded: engines
    without a retrieval stage never execute them.
    """
    from nemoguardrails.manifests import default_rail_catalog

    return sorted(
        {
            surface.action.target.split(":")[0]
            for manifest in default_rail_catalog().manifests.values()
            for surface in manifest.surfaces
            if surface.direction.value != "retrieval"
        }
    )


def _module_path(module: str) -> Optional[Path]:
    """Return the file backing a dotted module name, preferring the module over the package."""
    relative = Path(*module.split(".")[1:])
    for candidate in (PACKAGE_ROOT / relative.with_suffix(".py"), PACKAGE_ROOT / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _resolve_relative(module: Optional[str], level: int, importer: str) -> str:
    """Resolve a relative import to its absolute dotted name."""
    parts = importer.split(".")
    is_package_init = _module_path(importer) == PACKAGE_ROOT / Path(*parts[1:]) / "__init__.py"
    base = parts[: len(parts) - level + 1] if is_package_init else parts[: len(parts) - level]
    return ".".join(base + ([module] if module else []))


def _imports_of(module: str, path: Path) -> Set[str]:
    """Return the first-party modules imported at any level by *module*."""
    found: Set[str] = set()

    def visit(nodes) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_relative(node.module, node.level, module) if node.level else (node.module or "")
                if target:
                    found.add(target)
                    # `from x import y` may name a submodule rather than an attribute.
                    found.update(f"{target}.{alias.name}" for alias in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A function-local import runs only when that function is called, so it
                # does not load the module at import time. Deferring a Colang import into
                # the one code path that needs it is a valid way to cut an edge here.
                continue
            elif isinstance(node, ast.If) and _is_type_checking_guard(node.test):
                # `if TYPE_CHECKING:` bodies never execute.
                visit(node.orelse)
            else:
                visit(ast.iter_child_nodes(node))

    # Read bytes so ``ast`` applies the PEP 263 source encoding rather than the platform
    # default. ``read_text()`` decodes as cp1252 on Windows, which fails on the non-ASCII
    # refusal strings in library/content_safety/actions.py.
    visit(ast.iter_child_nodes(ast.parse(path.read_bytes(), filename=str(path))))
    return {name for name in found if name == PACKAGE or name.startswith(f"{PACKAGE}.")}


def _is_type_checking_guard(test: ast.expr) -> bool:
    """True for `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:`."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_chain_to_colang(entry: str) -> Optional[List[str]]:
    """Return the shortest first-party import chain from *entry* to a Colang module.

    The parent package ``nemoguardrails`` is skipped: its ``__init__`` imports the world,
    so every submodule would trivially reach Colang through it. That pre-existing
    condition is not what these tests are about.
    """
    queue: List[Tuple[str, List[str]]] = [(entry, [entry])]
    seen: Set[str] = {entry, PACKAGE}
    while queue:
        module, chain = queue.pop(0)
        path = _module_path(module)
        if path is None:
            continue
        for imported in sorted(_imports_of(module, path)):
            if imported.startswith(FORBIDDEN_PREFIX):
                return chain + [imported]
            if imported in seen:
                continue
            seen.add(imported)
            queue.append((imported, chain + [imported]))
    return None


def test_llm_call_module_does_not_depend_on_colang():
    """nemoguardrails.llm.call reaches no Colang module through first-party imports."""
    chain = _import_chain_to_colang("nemoguardrails.llm.call")

    assert chain is None, "nemoguardrails.llm.call must not depend on Colang; chain:\n  " + "\n  -> ".join(chain or [])


def test_compiled_rail_does_not_depend_on_colang():
    """``compiled_rail`` emits Colang-shaped *events* for actions but doesn't require
    a Colang runtime
    """
    chain = _import_chain_to_colang("nemoguardrails.guardrails.compiled_rail")

    assert chain is None, "compiled_rail must not depend on Colang; chain:\n  " + "\n  -> ".join(chain or [])


def test_rail_guard_does_not_depend_on_colang():
    """The shared rail error envelope stays Colang-free alongside compiled_rail."""
    chain = _import_chain_to_colang("nemoguardrails.guardrails.rail_guard")

    assert chain is None, "rail_guard must not depend on Colang; chain:\n  " + "\n  -> ".join(chain or [])


def test_import_graph_walker_detects_a_known_colang_dependent():
    """The walker is not vacuously passing: a known Colang-dependent module is flagged."""
    chain = _import_chain_to_colang("nemoguardrails.actions.llm.utils")

    assert chain is not None
    assert chain[0] == "nemoguardrails.actions.llm.utils"
    assert chain[-1].startswith(FORBIDDEN_PREFIX)


def test_no_in_scope_rail_action_loads_colang_at_import_time():
    """Importing any input/output rail action must not pull in the Colang runtime.

    This is the invariant that lets an engine without a Colang runtime execute the
    built-in rails. Colang is still loaded process-wide by ``nemoguardrails/__init__``,
    so this is about the action's own dependency graph, not about what happens to be
    resident in ``sys.modules``.
    """
    modules = _in_scope_rail_action_modules()
    assert modules, "manifest catalog produced no in-scope action modules"

    offenders = {module: chain for module in modules if (chain := _import_chain_to_colang(module)) is not None}

    assert not offenders, "rail actions must not load Colang at import time:\n" + "\n".join(
        f"  {module}:\n    " + "\n    -> ".join(chain) for module, chain in offenders.items()
    )


def test_moved_helpers_are_still_importable_from_their_previous_locations():
    """Pre-move import paths keep resolving to the same objects."""
    from nemoguardrails.actions.llm.utils import (
        get_multiline_response,
        llm_call,
        remove_action_intent_identifiers,
        strip_quotes,
        warn_if_truncated,
    )
    from nemoguardrails.colang.v1_0.runtime.flows import _get_flow_params, _normalize_flow_id
    from nemoguardrails.llm import call, completion_parsing
    from nemoguardrails.utils import _get_flow_params as canonical_params
    from nemoguardrails.utils import _normalize_flow_id as canonical_normalize

    assert llm_call is call.llm_call
    assert warn_if_truncated is call.warn_if_truncated
    assert strip_quotes is completion_parsing.strip_quotes
    assert get_multiline_response is completion_parsing.get_multiline_response
    assert remove_action_intent_identifiers is completion_parsing.remove_action_intent_identifiers
    assert _normalize_flow_id is canonical_normalize
    assert _get_flow_params is canonical_params


def test_function_local_imports_do_not_count_as_import_time_dependencies(tmp_path):
    """The walker distinguishes a deferred import from an eager one.

    Several Colang edges were cut by moving the import into the one function that needs
    it. That is only a real cut if the walker ignores function bodies, so this pins the
    behavior the other assertions rely on.
    """
    source = "import nemoguardrails.context\n\ndef f():\n    import nemoguardrails.colang.v1_0.runtime.flows\n"
    probe = tmp_path / "_import_graph_probe.py"
    probe.write_text(source, encoding="utf-8")

    imports = _imports_of("nemoguardrails._import_graph_probe", probe)

    assert "nemoguardrails.context" in imports
    assert not any(name.startswith(FORBIDDEN_PREFIX) for name in imports)
