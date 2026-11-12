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

"""Where LLMRailsRemovalWarning fires, what it says, and which line it is attributed to."""

import sys
import warnings
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version
from typer.testing import CliRunner

import nemoguardrails
from nemoguardrails import Guardrails, LLMRails, LLMRailsRemovalWarning, RailsConfig, set_default_framework
from nemoguardrails.cli import app
from nemoguardrails.deprecation import (
    DEPRECATED_SINCE,
    MIGRATION_GUIDE_URL,
    REMOVAL_MONTH,
    REMOVAL_RELEASE,
    llmrails_deprecation_headers,
)
from nemoguardrails.rails.llm.llmrails import _wrap_legacy_llm
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
LLMRAILS_ONLY_CONFIG = {"models": [], "rails": {"input": {"flows": ["jailbreak detection heuristics"]}}}

# `nemoguardrails.cli.chat` is shadowed by the `chat` command function on the package.
chat_module = sys.modules["nemoguardrails.cli.chat"]
runner = CliRunner()


def _removal_warnings(action) -> list[warnings.WarningMessage]:
    """Run *action* with every warning shown and return the LLMRailsRemovalWarnings it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        action()
    return [w for w in caught if issubclass(w.category, LLMRailsRemovalWarning)]


def _colang_1_config() -> RailsConfig:
    return RailsConfig.from_content(config=COLANG_1_CONFIG)


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


class _UserSubclassOfLLMRails(LLMRails):
    def __init__(self, config: RailsConfig):
        super().__init__(config, llm=FakeLLMModel(responses=[]))


def test_removal_warning_is_shown_to_application_users_by_default():
    """LLMRailsRemovalWarning is a FutureWarning, which Python's default filters show."""
    assert issubclass(LLMRailsRemovalWarning, FutureWarning)


def test_message_names_both_releases_the_month_and_the_migration_guide():
    """The LLMRails warning names the deprecating release, the removal release and month, and the migration guide."""
    (warning,) = _removal_warnings(lambda: LLMRails(_colang_1_config(), llm=FakeLLMModel(responses=[])))

    message = str(warning.message)
    assert message.startswith("LLMRails is deprecated")
    assert DEPRECATED_SINCE in message
    assert f"{REMOVAL_RELEASE} ({REMOVAL_MONTH})" in message
    assert MIGRATION_GUIDE_URL in message


def test_llmrails_warning_points_at_the_constructing_line():
    """Constructing LLMRails warns once, attributed to the caller's file."""
    caught = _removal_warnings(lambda: LLMRails(_colang_1_config(), llm=FakeLLMModel(responses=[])))

    assert [w.filename for w in caught] == [__file__]


def test_llmrails_subclass_defined_by_a_user_still_warns():
    """A user's LLMRails subclass warns, because its super().__init__ call comes from outside the package."""
    caught = _removal_warnings(lambda: _UserSubclassOfLLMRails(_colang_1_config()))

    assert [w.filename for w in caught] == [__file__]


def test_guardrails_on_iorails_emits_no_removal_warning(monkeypatch):
    """Guardrails with a config IORails runs emits no LLMRailsRemovalWarning."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(config=IORAILS_CONFIG)

    assert _removal_warnings(lambda: Guardrails(config)) == []


def test_guardrails_with_llm_warns_once_about_the_llm_argument():
    """Passing llm to Guardrails warns once, in engine-neutral words, and not again for the LLMRails it builds."""
    caught = _removal_warnings(lambda: Guardrails(_colang_1_config(), llm=FakeLLMModel(responses=[])))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == ["Passing an llm argument"]
    assert caught[0].filename == __file__


def test_guardrails_use_iorails_false_warns_once():
    """Guardrails(use_iorails=False) warns once about the argument, not again for the LLMRails it builds."""
    caught = _removal_warnings(lambda: Guardrails(_colang_1_config(), use_iorails=False))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == ["Guardrails(use_iorails=False)"]
    assert caught[0].filename == __file__


def test_guardrails_with_llm_and_use_iorails_false_warns_once():
    """Passing both llm and use_iorails=False warns once, about use_iorails=False."""
    caught = _removal_warnings(
        lambda: Guardrails(_colang_1_config(), llm=FakeLLMModel(responses=[]), use_iorails=False)
    )

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == ["Guardrails(use_iorails=False)"]


def test_guardrails_that_must_run_iorails_raises_without_a_removal_warning():
    """require_iorails=True with an llm raises before any LLMRailsRemovalWarning, since no LLMRails is built."""

    def construct():
        with pytest.raises(ValueError, match="IORails cannot be used"):
            Guardrails(_colang_1_config(), llm=FakeLLMModel(responses=[]), require_iorails=True)

    assert _removal_warnings(construct) == []


@DIALOG_CONTENTS
def test_guardrails_on_iorails_with_dialog_flows_warns_that_they_are_not_run(monkeypatch, colang_content):
    """A Colang 1.0 config with dialog flows, with or without define user, runs on IORails and warns once."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=colang_content, config=IORAILS_CONFIG)
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _removal_warnings(construct)

    assert rails is not None and rails.use_iorails_engine
    assert [w.filename for w in caught] == [__file__]
    assert str(caught[0].message).startswith("Using Colang dialog flows (define user, define flow) is deprecated")


def test_guardrails_on_iorails_with_config_py_warns_that_it_is_not_imported(tmp_path, monkeypatch):
    """A config directory with config.py still runs on IORails, and warns once that IORails ignores the file."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=True)
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _removal_warnings(construct)

    assert rails is not None and rails.use_iorails_engine
    assert [w.filename for w in caught] == [__file__]
    assert str(caught[0].message).startswith("Using config.py is deprecated")


def test_guardrails_with_dialog_flows_and_config_py_warns_once_about_the_flows(tmp_path, monkeypatch):
    """A config with both dialog flows and config.py emits one warning, the dialog-flow one."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=True, colang_content=FLOW_ONLY_CONTENT)

    caught = _removal_warnings(lambda: Guardrails(config))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == [
        "Using Colang dialog flows (define user, define flow)"
    ]


def test_guardrails_on_a_config_directory_without_config_py_emits_no_removal_warning(tmp_path, monkeypatch):
    """An IORails config loaded from a directory without config.py emits no LLMRailsRemovalWarning."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = _config_dir(tmp_path, config_py=False)

    assert _removal_warnings(lambda: Guardrails(config)) == []


def test_guardrails_warns_when_a_flow_redefines_a_library_rail_iorails_runs(monkeypatch):
    """A user flow that redefines a library rail stays on IORails, which runs its own rail, so it warns like a dialog flow."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(colang_content=SELF_CHECK_INPUT_OVERRIDE, config=SELF_CHECK_INPUT_CONFIG)
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _removal_warnings(construct)

    assert rails is not None and rails.use_iorails_engine
    assert [str(w.message).split(" is deprecated")[0] for w in caught] == [
        "Using Colang dialog flows (define user, define flow)"
    ]


def test_custom_rail_flows_fall_back_with_the_reason_instead_of_the_dialog_flow_warning(monkeypatch):
    """A rail flow IORails does not know sends the config to LLMRails with the fallback warning, never the dialog-flow one."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    config = RailsConfig.from_content(
        colang_content=CUSTOM_OUTPUT_RAIL,
        config={**IORAILS_CONFIG, "rails": {"output": {"flows": ["check blocked terms"]}}},
    )
    rails = None

    def construct():
        nonlocal rails
        rails = Guardrails(config)

    caught = _removal_warnings(construct)

    assert rails is not None and not rails.use_iorails_engine
    assert len(caught) == 1
    assert str(caught[0].message).startswith("Running this config on the LLMRails engine ('check blocked terms'")


def test_guardrails_fallback_warns_once_with_the_reason():
    """A config that falls back to LLMRails warns once, naming why IORails can't run it."""
    config = RailsConfig.from_content(colang_content=COLANG_2_CONTENT, yaml_content=COLANG_2_YAML)

    caught = _removal_warnings(lambda: Guardrails(config))

    assert len(caught) == 1
    assert "IORails supports Colang 1.0 only" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_testchat_warns_once_at_the_constructing_line():
    """TestChat warns once, attributed to the test that built it, and its internal LLMRails stays silent."""
    caught = _removal_warnings(lambda: TestChat(_colang_1_config(), llm_completions=[]))

    assert [w.filename for w in caught] == [__file__]
    assert str(caught[0].message).startswith("TestChat is deprecated")


def test_runnable_rails_warns_once():
    """RunnableRails warns once, and its internal LLMRails stays silent."""
    pytest.importorskip("langchain_core")
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

    caught = _removal_warnings(lambda: RunnableRails(_colang_1_config()))

    assert [w.filename for w in caught] == [__file__]
    assert str(caught[0].message).startswith("RunnableRails is deprecated")


def test_guardrails_middleware_warns_with_the_subclass_name():
    """A GuardrailsMiddleware subclass warns once under its own name."""
    pytest.importorskip("langchain.agents.middleware")
    from nemoguardrails.integrations.langchain.middleware import InputRailsMiddleware

    caught = _removal_warnings(lambda: InputRailsMiddleware(config_yaml="models: []"))

    assert [w.filename for w in caught] == [__file__]
    assert str(caught[0].message).startswith("InputRailsMiddleware is deprecated")


def test_langchain_framework_warns():
    """Creating the LangChain model framework warns, attributed to the caller."""
    from nemoguardrails.integrations.langchain.llm_adapter import LangChainFramework

    caught = _removal_warnings(LangChainFramework)

    assert [w.filename for w in caught] == [__file__]


@pytest.mark.parametrize(
    ("construct", "subject"),
    [
        (lambda config: LLMRails(config), "LLMRails"),
        (lambda config: Guardrails(config, use_iorails=False), "Guardrails(use_iorails=False)"),
    ],
    ids=["llmrails", "guardrails-on-llmrails"],
)
def test_construction_on_the_langchain_framework_warns_once(langchain_framework, monkeypatch, construct, subject):
    """A construction that builds the LangChain model framework emits only its own removal warning."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = RailsConfig.from_content(config={"models": [{"type": "main", "engine": "openai", "model": "gpt-4o-mini"}]})

    caught = _removal_warnings(lambda: construct(config))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == [subject]
    assert caught[0].filename == __file__


def test_selecting_the_langchain_framework_warns_at_the_caller(langchain_framework):
    """set_default_framework("langchain") warns once, attributed to the caller's line."""
    caught = _removal_warnings(lambda: set_default_framework("langchain"))

    assert [str(w.message).split(" is deprecated")[0] for w in caught] == ["The LangChain model framework"]
    assert caught[0].filename == __file__


def test_selecting_the_default_framework_does_not_warn(langchain_framework):
    """set_default_framework("default") emits no removal warning."""
    assert _removal_warnings(lambda: set_default_framework("default")) == []


def test_langchain_adapter_built_by_a_user_warns():
    """A LangChainLLMAdapter built outside the package warns."""
    from nemoguardrails.integrations.langchain.llm_adapter import LangChainLLMAdapter

    caught = _removal_warnings(lambda: LangChainLLMAdapter(object()))

    assert [w.filename for w in caught] == [__file__]


def test_langchain_adapter_built_inside_the_package_is_silent():
    """A LangChainLLMAdapter the package builds for a raw LangChain LLM emits no removal warning."""
    pytest.importorskip("langchain_core")

    assert _removal_warnings(lambda: _wrap_legacy_llm(object())) == []


def test_deprecation_headers_use_the_first_day_of_each_month():
    """The server headers carry the deprecation and sunset months as first-of-month timestamps."""
    headers = llmrails_deprecation_headers()

    assert headers["Deprecation"] == f"@{int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp())}"
    assert headers["Sunset"] == "Fri, 01 Jan 2027 00:00:00 GMT"
    assert headers["Link"] == f'<{MIGRATION_GUIDE_URL}>; rel="deprecation"; type="text/html"'


def test_removal_release_has_not_shipped_yet():
    """The package version is below the removal release; reaching it means the deprecated code must be removed."""
    assert Version(nemoguardrails.__version__) < Version(REMOVAL_RELEASE), (
        f"NeMo Guardrails {nemoguardrails.__version__} has reached {REMOVAL_RELEASE}: remove LLMRails, the Colang "
        "runtimes, TestChat, and the LangChain integration, as announced in nemoguardrails/deprecation.py."
    )


@patch("nemoguardrails.cli.migrate")
def test_convert_command_prints_the_removal_notice(mock_migrate):
    """The convert command prints the removal notice and still runs."""
    result = runner.invoke(app, ["convert", "flows.co"])

    assert result.exit_code == 0
    assert "The convert command is deprecated" in result.output
    mock_migrate.assert_called_once()


@patch("uvicorn.run")
def test_actions_server_command_prints_the_removal_notice(mock_uvicorn):
    """The actions-server command prints the removal notice and still starts the server."""
    result = runner.invoke(app, ["actions-server"])

    assert result.exit_code == 0
    assert "The actions-server command is deprecated" in result.output
    mock_uvicorn.assert_called_once()


def test_eval_rail_topical_command_prints_the_removal_notice():
    """The eval rail topical command prints the removal notice before it validates its arguments."""
    result = runner.invoke(app, ["eval", "rail", "topical"])

    assert "The eval rail topical command is deprecated" in result.output


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

    assert "Colang 2.x chat, including its debugger commands, is deprecated" in capsys.readouterr().err


def _colang_1_chat_notice(config: RailsConfig, capsys, **run_chat_kwargs) -> str:
    """Run the Colang 1.0 chat for *config* without starting the loop, and return what it printed on stderr."""
    with (
        patch.object(chat_module.RailsConfig, "from_path", return_value=config),
        patch("asyncio.run") as mock_asyncio_run,
    ):
        mock_asyncio_run.side_effect = lambda coroutine: coroutine.close()
        chat_module.run_chat(config_path="config", **run_chat_kwargs)
        mock_asyncio_run.assert_called_once()
    return capsys.readouterr().err


def test_colang_1_chat_on_an_iorails_config_prints_no_removal_notice(capsys):
    """Chatting with a Colang 1.0 config IORails runs in full prints no removal notice."""
    notice = _colang_1_chat_notice(RailsConfig.from_content(config=IORAILS_CONFIG), capsys)

    assert "is deprecated" not in notice


@DIALOG_CONTENTS
def test_colang_1_chat_with_dialog_flows_prints_the_dialog_flows_notice(capsys, colang_content):
    """Chatting with a Colang 1.0 config that has dialog flows, with or without define user, prints the notice."""
    config = RailsConfig.from_content(colang_content=colang_content, config=IORAILS_CONFIG)

    notice = _colang_1_chat_notice(config, capsys)

    assert "Using Colang dialog flows (define user, define flow) is deprecated" in notice


def test_colang_1_chat_starts_even_when_the_iorails_check_fails(capsys):
    """A failing IORails support check skips the notice instead of stopping the chat."""
    config = RailsConfig.from_content(config=LLMRAILS_ONLY_CONFIG)

    with patch.object(chat_module.IORails, "unsupported_reason", side_effect=RuntimeError("action import failed")):
        notice = _colang_1_chat_notice(config, capsys)

    assert "is deprecated" not in notice


def test_remote_colang_1_chat_skips_the_local_engine_notice(capsys):
    """Chatting through a server URL prints no local-engine notice; the server decides which engine runs."""
    config = RailsConfig.from_content(config=LLMRAILS_ONLY_CONFIG)

    with patch.object(chat_module.IORails, "unsupported_reason") as unsupported_reason:
        notice = _colang_1_chat_notice(config, capsys, server_url="http://localhost:8000")

    unsupported_reason.assert_not_called()
    assert "is deprecated" not in notice


def test_colang_1_chat_with_config_py_prints_the_config_py_notice(tmp_path, capsys):
    """Chatting with a Colang 1.0 config directory that has config.py prints that IORails does not import it."""
    notice = _colang_1_chat_notice(_config_dir(tmp_path, config_py=True), capsys)

    assert "Using config.py is deprecated" in notice


def test_colang_1_chat_on_an_llmrails_only_config_prints_the_fallback_notice(capsys):
    """Chatting with a Colang 1.0 config only LLMRails can run prints why, and what to change."""
    notice = _colang_1_chat_notice(RailsConfig.from_content(config=LLMRAILS_ONLY_CONFIG), capsys)

    assert "Running this config on the LLMRails engine (" in notice
    assert "Use only rails that IORails supports" in notice
