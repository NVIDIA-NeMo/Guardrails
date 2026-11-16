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

"""Removal warnings for direct LLMRails use and Colang 2.x, and the UserWarnings for config IORails ignores."""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import nemoguardrails
import nemoguardrails.cli.chat  # noqa: F401
from nemoguardrails import Guardrails, LLMRails, LLMRailsRemovalWarning, RailsConfig, set_default_framework
from nemoguardrails.deprecation import DEPRECATED_SINCE, REMOVAL_RELEASE
from nemoguardrails.testing import FakeLLMModel, TestChat

COLANG_1_CONFIG = {"models": []}
IORAILS_CONFIG = {"models": [{"type": "main", "engine": "nim", "model": "meta/llama-3.3-70b-instruct"}]}
COLANG_2_CONTENT = 'import core\n\nflow main\n  user said "hi"\n  bot say "Hello"\n'
COLANG_2_YAML = 'colang_version: "2.x"\nmodels: []\n'
DIALOG_FLOWS_CONTENT = (
    'define user express greeting\n  "hi"\n\n'
    'define bot express greeting\n  "Hello!"\n\n'
    "define flow greeting\n  user express greeting\n  bot express greeting\n"
)
FLOW_ONLY_CONTENT = 'define flow greeting\n  user express greeting\n  bot "Hello!"\n'
DIALOG_CONTENTS = pytest.mark.parametrize(
    "colang_content", [DIALOG_FLOWS_CONTENT, FLOW_ONLY_CONTENT], ids=["with-define-user", "flow-only"]
)
SELF_CHECK_INPUT_OVERRIDE = (
    "define subflow self check input\n"
    "  $allowed = execute self_check_input\n"
    "  if not $allowed\n"
    "    bot refuse to respond\n"
    "    stop\n"
)
CUSTOM_OUTPUT_RAIL = "define subflow check blocked terms\n  bot refuse to respond\n  stop\n"
SELF_CHECK_INPUT_CONFIG = {
    **IORAILS_CONFIG,
    "rails": {"input": {"flows": ["self check input"]}},
    "prompts": [{"task": "self_check_input", "content": "Should this be blocked? {{ user_input }}"}],
}
LIBRARY_SELF_CHECK_INPUT = (
    Path(nemoguardrails.__file__).parent / "library" / "self_check" / "input_check" / "flows.v1.co"
).read_text()
_LLMRAILS_DEFAULT_FLOWS = (Path(nemoguardrails.__file__).parent / "rails" / "llm" / "llm_flows.co").read_text()
RUN_DIALOG_RAILS_COPY = _LLMRAILS_DEFAULT_FLOWS[
    _LLMRAILS_DEFAULT_FLOWS.index("define flow run dialog rails") : _LLMRAILS_DEFAULT_FLOWS.index(
        "define subflow generate user intent"
    )
]
LLMRAILS_ONLY_CONFIG = {"models": [], "rails": {"input": {"flows": ["jailbreak detection heuristics"]}}}
DIALOG_FLOWS_DIAGNOSTIC = "IORails does not run this config's Colang dialog flows"
CONFIG_PY_DIAGNOSTIC = "IORails does not import this config's config.py"

# `nemoguardrails.cli.chat` is shadowed by the `chat` command function on the package.
chat_module = sys.modules["nemoguardrails.cli.chat"]


def _record_warnings(action) -> list[warnings.WarningMessage]:
    """Run *action* with every warning shown and return all the warnings it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        action()
    return list(caught)


def _removals(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, LLMRailsRemovalWarning)]


def _diagnostics(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if w.category is UserWarning and str(w.message).startswith("IORails does not")]


def _removal_subjects(action) -> list[str]:
    """Run *action* and return the subject of each LLMRailsRemovalWarning it emitted."""
    return [str(w.message).split(" is deprecated")[0] for w in _removals(_record_warnings(action))]


def _diagnostic_prefixes(caught: list[warnings.WarningMessage]) -> list[str]:
    """Map each IORails diagnostic in *caught* to the known prefix it starts with."""
    prefixes = (DIALOG_FLOWS_DIAGNOSTIC, CONFIG_PY_DIAGNOSTIC)
    return [next(p for p in prefixes if str(w.message).startswith(p)) for w in _diagnostics(caught)]


def _colang_1_config() -> RailsConfig:
    return RailsConfig.from_content(config=COLANG_1_CONFIG)


def _colang_2_config() -> RailsConfig:
    return RailsConfig.from_content(colang_content=COLANG_2_CONTENT, yaml_content=COLANG_2_YAML)


def _config_dir(tmp_path, *, config_py: bool, colang_content: str = "") -> RailsConfig:
    """Write an IORails-compatible config directory, optionally with config.py and Colang flows, and load it."""
    (tmp_path / "config.yml").write_text(
        "models:\n  - type: main\n    engine: nim\n    model: meta/llama-3.3-70b-instruct\n"
    )
    if config_py:
        (tmp_path / "config.py").write_text("def init(app):\n    pass\n")
    if colang_content:
        (tmp_path / "flows.co").write_text(colang_content)
    return RailsConfig.from_path(str(tmp_path))


def _build_llmrails_first(config: RailsConfig) -> None:
    """Construct LLMRails on *config*, which adds its default and library flows to the same config object."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LLMRailsRemovalWarning)
        LLMRails(config)


class _UserSubclassOfLLMRails(LLMRails):
    def __init__(self, config: RailsConfig):
        super().__init__(config, llm=FakeLLMModel(responses=[]))


def test_removal_warning_is_a_deprecation_warning():
    """LLMRailsRemovalWarning is a DeprecationWarning, shown in tests and scripts and hidden in imported modules."""
    assert issubclass(LLMRailsRemovalWarning, DeprecationWarning)


def test_message_says_removed_no_earlier_than_the_release():
    """The LLMRails warning names the deprecating release and a removal lower bound, and ends at the alternative."""
    (warning,) = _removals(_record_warnings(lambda: LLMRails(_colang_1_config(), llm=FakeLLMModel(responses=[]))))

    message = str(warning.message)
    assert message.startswith("LLMRails is deprecated")
    assert f"since NeMo Guardrails {DEPRECATED_SINCE}" in message
    assert f"will be removed no earlier than {REMOVAL_RELEASE}." in message
    assert message.endswith(
        f"no earlier than {REMOVAL_RELEASE}. Use Guardrails(config), which runs the IORails engine."
    )


def test_llmrails_warning_points_at_the_constructing_line():
    """Constructing LLMRails directly warns once, attributed to the caller's file."""
    caught = _removals(_record_warnings(lambda: LLMRails(_colang_1_config(), llm=FakeLLMModel(responses=[]))))

    assert [w.filename for w in caught] == [__file__]


def test_llmrails_subclass_defined_by_a_user_still_warns():
    """A user's LLMRails subclass warns, because its super().__init__ call comes from outside the package."""
    caught = _removals(_record_warnings(lambda: _UserSubclassOfLLMRails(_colang_1_config())))

    assert [w.filename for w in caught] == [__file__]


COLANG_2_ROUTES = {
    "llmrails": lambda config: LLMRails(config, llm=FakeLLMModel(responses=[])),
    "guardrails-fallback": lambda config: Guardrails(config),
    "guardrails-use-iorails-false": lambda config: Guardrails(config, use_iorails=False),
    "testchat": lambda config: TestChat(config, llm_completions=[]),
}


@pytest.mark.parametrize("route", sorted(COLANG_2_ROUTES))
def test_every_route_into_colang_2_warns_once_about_colang_2(route):
    """Each way of running a Colang 2.x config emits one Colang 2.x removal warning, attributed to the caller."""
    config = _colang_2_config()

    caught = _removals(_record_warnings(lambda: COLANG_2_ROUTES[route](config)))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == ["Colang 2.x"]
    assert caught[0].filename == __file__
    assert "IORails will not support Colang 2.x, so there is no replacement." in str(caught[0].message)


def test_runnable_rails_on_colang_2_warns_about_colang_2():
    """RunnableRails on a Colang 2.x config emits the Colang 2.x removal warning, not one about RunnableRails."""
    pytest.importorskip("langchain_core")
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

    config = _colang_2_config()

    assert _removal_subjects(lambda: RunnableRails(config)) == ["Colang 2.x"]


KEPT_SURFACES = {
    "guardrails-llm": lambda: Guardrails(_colang_1_config(), llm=FakeLLMModel(responses=[])),
    "guardrails-use-iorails-false": lambda: Guardrails(_colang_1_config(), use_iorails=False),
    "testchat": lambda: TestChat(_colang_1_config(), llm_completions=[]),
}


@pytest.mark.parametrize("surface", sorted(KEPT_SURFACES))
def test_surfaces_without_a_removal_decision_emit_no_removal_warning(surface):
    """Guardrails(llm=...), Guardrails(use_iorails=False) and TestChat on a Colang 1.0 config emit no removal warning."""
    assert _removal_subjects(KEPT_SURFACES[surface]) == []


def test_langchain_surfaces_emit_no_removal_warning(langchain_framework):
    """The LangChain framework, adapter, RunnableRails and middleware on a Colang 1.0 config emit no removal warning."""
    pytest.importorskip("langchain.agents.middleware")
    from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework, LangChainLLMAdapter
    from nemoguardrails.integrations.langchain.middleware import InputRailsMiddleware
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

    constructions = [
        lambda: set_default_framework("langchain"),
        LangChainFramework,
        lambda: LangChainLLMAdapter(object()),
        lambda: RunnableRails(_colang_1_config()),
        lambda: InputRailsMiddleware(config_yaml="models: []"),
    ]

    assert [_removal_subjects(construct) for construct in constructions] == [[]] * len(constructions)


def test_llmrails_on_the_langchain_framework_warns_once(langchain_framework, monkeypatch):
    """LLMRails on the LangChain model framework emits only its own removal warning."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = RailsConfig.from_content(config={"models": [{"type": "main", "engine": "openai", "model": "gpt-4o-mini"}]})

    assert _removal_subjects(lambda: LLMRails(config)) == ["LLMRails"]


@pytest.mark.parametrize(
    ("colang_content", "config_dict"),
    [
        ("", LLMRAILS_ONLY_CONFIG),
        (CUSTOM_OUTPUT_RAIL, {**IORAILS_CONFIG, "rails": {"output": {"flows": ["check blocked terms"]}}}),
    ],
    ids=["unavailable-rail", "custom-rail-flow"],
)
def test_fallback_for_a_reason_other_than_colang_2_only_logs(monkeypatch, colang_content, config_dict):
    """A config that falls back to LLMRails for a reason other than Colang 2.x gets no warning, only the existing log."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=colang_content, config=config_dict)
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _record_warnings(construct)

    assert rails is not None and not rails.use_iorails_engine
    assert _removals(caught) == []
    assert _diagnostics(caught) == []


def test_guardrails_on_iorails_without_ignored_config_emits_nothing(monkeypatch):
    """Guardrails with a config IORails runs in full emits neither a removal warning nor a diagnostic."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    caught = _record_warnings(lambda: Guardrails(RailsConfig.from_content(config=IORAILS_CONFIG)))

    assert _removals(caught) == []
    assert _diagnostics(caught) == []


@DIALOG_CONTENTS
def test_dialog_flows_on_iorails_raise_a_user_warning_without_a_removal_date(monkeypatch, colang_content):
    """IORails running a config with its own dialog flows raises one undated UserWarning at the caller's line."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=colang_content, config=IORAILS_CONFIG)
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _record_warnings(construct)

    assert rails is not None and rails.use_iorails_engine
    assert _diagnostic_prefixes(caught) == [DIALOG_FLOWS_DIAGNOSTIC]
    (diagnostic,) = _diagnostics(caught)
    assert diagnostic.filename == __file__
    assert "deprecated" not in str(diagnostic.message)
    assert REMOVAL_RELEASE not in str(diagnostic.message)
    assert _removals(caught) == []


def test_config_py_on_iorails_raises_a_user_warning(tmp_path, monkeypatch):
    """IORails running a config directory with config.py raises a UserWarning that the file is not imported."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=True)

    assert _diagnostic_prefixes(_record_warnings(lambda: Guardrails(config))) == [CONFIG_PY_DIAGNOSTIC]


def test_dialog_flows_and_config_py_both_raise_a_user_warning(tmp_path, monkeypatch):
    """A config with both dialog flows and config.py raises one UserWarning for each."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=True, colang_content=FLOW_ONLY_CONTENT)

    caught = _record_warnings(lambda: Guardrails(config))

    assert _diagnostic_prefixes(caught) == [DIALOG_FLOWS_DIAGNOSTIC, CONFIG_PY_DIAGNOSTIC]


def test_config_directory_without_config_py_raises_no_diagnostic(tmp_path, monkeypatch):
    """An IORails config loaded from a directory without config.py raises no diagnostic."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=False)

    assert _diagnostics(_record_warnings(lambda: Guardrails(config))) == []


def test_redefinition_of_a_library_rail_is_reported_as_an_ignored_dialog_flow(monkeypatch):
    """A user flow that redefines a library rail stays on IORails, which runs its own rail, so it is reported."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=SELF_CHECK_INPUT_OVERRIDE, config=SELF_CHECK_INPUT_CONFIG)

    assert _diagnostic_prefixes(_record_warnings(lambda: Guardrails(config))) == [DIALOG_FLOWS_DIAGNOSTIC]


@pytest.mark.parametrize(
    ("colang_content", "config_dict"),
    [(LIBRARY_SELF_CHECK_INPUT, SELF_CHECK_INPUT_CONFIG), (RUN_DIALOG_RAILS_COPY, IORAILS_CONFIG)],
    ids=["library-rail-flow", "llmrails-default-dialog-flow"],
)
def test_verbatim_copy_of_a_bundled_flow_is_reported(monkeypatch, colang_content, config_dict):
    """A flow the user copied unchanged from the library or llm_flows.co is still the user's own and is reported."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=colang_content, config=config_dict)

    assert _diagnostic_prefixes(_record_warnings(lambda: Guardrails(config))) == [DIALOG_FLOWS_DIAGNOSTIC]


def test_reused_config_after_llmrails_raises_no_dialog_flow_diagnostic(monkeypatch):
    """Flows LLMRails added to a reused config are not the user's dialog flows, so Guardrails reports none."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(config=IORAILS_CONFIG)
    _build_llmrails_first(config)

    assert _diagnostics(_record_warnings(lambda: Guardrails(config))) == []


def test_config_py_diagnostic_survives_llmrails_loading_the_same_config(tmp_path, monkeypatch):
    """Flows LLMRails added to a reused config do not add a dialog-flow diagnostic next to the config.py one."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=True)
    _build_llmrails_first(config)

    assert _diagnostic_prefixes(_record_warnings(lambda: Guardrails(config))) == [CONFIG_PY_DIAGNOSTIC]


def test_library_rail_redefinition_is_still_reported_after_llmrails_loaded_the_config(monkeypatch):
    """A user's redefinition of a library rail is still reported after LLMRails marked it as a system flow."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=SELF_CHECK_INPUT_OVERRIDE, config=SELF_CHECK_INPUT_CONFIG)
    _build_llmrails_first(config)

    assert _diagnostic_prefixes(_record_warnings(lambda: Guardrails(config))) == [DIALOG_FLOWS_DIAGNOSTIC]


def test_colang_2_chat_prints_the_removal_notice(capsys):
    """Chatting with a Colang 2.x config prints the removal notice on stderr."""
    with (
        patch.object(chat_module, "RailsConfig") as mock_rails_config,
        patch.object(chat_module, "LLMRails"),
        patch.object(chat_module, "get_or_create_event_loop") as mock_get_loop,
    ):
        mock_rails_config.from_path.return_value = MagicMock(colang_version="2.x")
        mock_get_loop.return_value.run_until_complete.side_effect = lambda coroutine: coroutine.close()

        chat_module.run_chat(config_path="config")

    notice = capsys.readouterr().err
    assert "Colang 2.x chat, including its debugger commands, is deprecated" in notice
    assert f"will be removed no earlier than {REMOVAL_RELEASE}." in notice
    assert notice.strip().endswith("IORails will not support Colang 2.x, so there is no replacement.")


@pytest.mark.parametrize("engine", [LLMRails, Guardrails], ids=["llmrails", "iorails-alias"])
def test_colang_2_chat_emits_one_message(monkeypatch, capsys, engine):
    """Colang 2.x chat prints its notice and silences the Colang 2.x warning its engine would otherwise add."""
    monkeypatch.setattr(chat_module, "LLMRails", engine)
    config = _colang_2_config()

    def chat():
        with (
            patch.object(chat_module.RailsConfig, "from_path", return_value=config),
            patch.object(chat_module, "get_or_create_event_loop") as mock_get_loop,
        ):
            mock_get_loop.return_value.run_until_complete.side_effect = lambda coroutine: coroutine.close()
            chat_module.run_chat(config_path="config")

    assert _removals(_record_warnings(chat)) == []
    assert "Colang 2.x chat, including its debugger commands, is deprecated" in capsys.readouterr().err


def _colang_1_chat_notice(config: RailsConfig, capsys) -> str:
    """Run the Colang 1.0 chat for *config* without starting the loop, and return what it printed on stderr."""
    with (
        patch.object(chat_module.RailsConfig, "from_path", return_value=config),
        patch("asyncio.run") as mock_asyncio_run,
    ):
        mock_asyncio_run.side_effect = lambda coroutine: coroutine.close()
        chat_module.run_chat(config_path="config")
        mock_asyncio_run.assert_called_once()
    return capsys.readouterr().err


def _chat_config(config_kind: str, tmp_path) -> RailsConfig:
    """Build the Colang 1.0 config of *config_kind* that the chat tests run."""
    builders = {
        "iorails-compatible": lambda: RailsConfig.from_content(config=IORAILS_CONFIG),
        "dialog-flows": lambda: RailsConfig.from_content(colang_content=DIALOG_FLOWS_CONTENT, config=IORAILS_CONFIG),
        "config-py": lambda: _config_dir(tmp_path, config_py=True),
        "llmrails-only": lambda: RailsConfig.from_content(config=LLMRAILS_ONLY_CONFIG),
    }
    return builders[config_kind]()


class _ChatStopped(Exception):
    """Raised from the first prompt, so that a chat stops right after it builds its engine."""


@pytest.mark.parametrize("config_kind", ["iorails-compatible", "dialog-flows", "config-py", "llmrails-only"])
def test_colang_1_chat_prints_no_notice(tmp_path, capsys, config_kind):
    """Colang 1.0 chat on its default engine prints no notice of its own, whatever the config holds."""
    assert "is deprecated" not in _colang_1_chat_notice(_chat_config(config_kind, tmp_path), capsys)


@pytest.mark.parametrize(
    ("config_kind", "expected_diagnostics"),
    [
        ("iorails-compatible", []),
        ("dialog-flows", [DIALOG_FLOWS_DIAGNOSTIC]),
        ("config-py", [CONFIG_PY_DIAGNOSTIC]),
        ("llmrails-only", []),
    ],
    ids=["iorails-compatible", "dialog-flows", "config-py", "llmrails-only"],
)
def test_colang_1_chat_under_the_iorails_alias_shows_the_guardrails_diagnostics(
    monkeypatch, tmp_path, capsys, config_kind, expected_diagnostics
):
    """With LLMRails aliased to Guardrails, chat prints nothing of its own and Guardrails raises the matching diagnostics."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(chat_module, "LLMRails", Guardrails)
    config = _chat_config(config_kind, tmp_path)

    def chat_until_the_first_prompt():
        with (
            patch.object(chat_module.RailsConfig, "from_path", return_value=config),
            patch("builtins.input", side_effect=_ChatStopped),
            pytest.raises(_ChatStopped),
        ):
            chat_module.run_chat(config_path="config")

    caught = _record_warnings(chat_until_the_first_prompt)

    assert "is deprecated" not in capsys.readouterr().err
    assert _diagnostic_prefixes(caught) == expected_diagnostics
    assert _removals(caught) == []
