# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Runtime import-cost guarantees for the top-level package.

The top-level ``nemoguardrails/__init__`` and ``nemoguardrails.rails/__init__``
resolve their public names lazily (PEP 562). Because of that, a lightweight
import no longer boots the Colang runtime, and ``sys.modules`` becomes a
meaningful place to observe the property. Every check runs in a fresh
interpreter so that Colang loaded by one assertion (for example, resolving
``LLMRails``) cannot leak into another and mask a regression.

The companion module ``tests/llm/test_call_import_graph.py`` proves the same
Colang-independence statically from the module dependency graph; these tests
prove it dynamically for the assembled package.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# A built-in input/output rail action; the circular import that motivated lazy
# ``rails`` resolution first surfaced while importing this module.
RAIL_ACTION_MODULE = "nemoguardrails.library.content_safety.actions"


def _run(body: str, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    """Run *body* in a fresh interpreter and return the completed process."""
    env = dict(os.environ)
    # Keep child startup cheap and deterministic regardless of the caller's env.
    env.pop("NEMO_GUARDRAILS_IORAILS_ENGINE", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def _assert_colang_free(import_statement: str) -> None:
    """Assert *import_statement* loads no ``nemoguardrails.colang`` module."""
    result = _run(
        f"""
        import sys

        {import_statement}

        loaded = sorted(m for m in sys.modules if m.startswith("nemoguardrails.colang"))
        print("\\n".join(loaded))
        sys.exit(1 if loaded else 0)
        """
    )
    assert result.returncode == 0, (
        f"`{import_statement}` should not import Colang, but loaded:\n{result.stdout}{result.stderr}"
    )


# 1-3: lightweight package imports do not boot Colang.


def test_import_package_does_not_load_colang():
    _assert_colang_free("import nemoguardrails")


def test_import_lightweight_type_does_not_load_colang():
    _assert_colang_free("from nemoguardrails import ChatMessage")


def test_import_rails_config_does_not_load_colang():
    _assert_colang_free("from nemoguardrails import RailsConfig")


# 4-5: rail-action and llm_call paths do not boot Colang (relies on #2241 having
# decoupled the action/llm_call dependency graph from Colang).


def test_import_builtin_rail_action_does_not_load_colang():
    _assert_colang_free(f"import {RAIL_ACTION_MODULE}")


def test_import_llm_call_does_not_load_colang():
    _assert_colang_free("import nemoguardrails.llm.call")


# 6-7: the legacy-runtime entry points still resolve; loading Colang here is
# expected because they directly back the Colang-based engine.


def test_accessing_llmrails_loads_legacy_runtime():
    result = _run(
        """
        import sys

        from nemoguardrails import LLMRails
        from nemoguardrails.rails.llm.llmrails import LLMRails as Real

        assert LLMRails is Real, "top-level LLMRails must be the legacy runtime class"
        assert any(m.startswith("nemoguardrails.colang") for m in sys.modules), (
            "resolving LLMRails is expected to load the Colang runtime"
        )
        print("ok")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_accessing_guardrails_resolves():
    result = _run(
        """
        from nemoguardrails import Guardrails
        from nemoguardrails.guardrails.guardrails import Guardrails as Real

        assert Guardrails is Real
        print("ok")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


# 8-9: the NEMO_GUARDRAILS_IORAILS_ENGINE alias, including re-evaluation on reload.


def test_env_var_aliases_llmrails_to_guardrails():
    result = _run(
        """
        import nemoguardrails

        assert nemoguardrails.LLMRails is nemoguardrails.Guardrails
        print("ok")
        """,
        env_overrides={"NEMO_GUARDRAILS_IORAILS_ENGINE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_reload_reevaluates_env_var_alias():
    result = _run(
        """
        import importlib
        import os

        import nemoguardrails
        from nemoguardrails.rails.llm.llmrails import LLMRails as Real

        assert nemoguardrails.LLMRails is Real

        os.environ["NEMO_GUARDRAILS_IORAILS_ENGINE"] = "1"
        importlib.reload(nemoguardrails)
        assert nemoguardrails.LLMRails is nemoguardrails.Guardrails

        del os.environ["NEMO_GUARDRAILS_IORAILS_ENGINE"]
        importlib.reload(nemoguardrails)
        assert nemoguardrails.LLMRails is Real
        print("ok")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


# 10: attribute-surface contracts are preserved.


def test_public_surface_contracts_preserved():
    result = _run(
        """
        import nemoguardrails

        exported = set(nemoguardrails.__all__)
        listed = set(dir(nemoguardrails))
        assert exported <= listed, exported - listed

        # Every advertised name resolves.
        for name in nemoguardrails.__all__:
            getattr(nemoguardrails, name)

        # Unknown attributes still raise AttributeError.
        try:
            nemoguardrails.DefinitelyNotAThing
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError for unknown attribute")
        print("ok")
        """
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
