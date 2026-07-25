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
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails.eval.models import (
    EvalConfig,
    EvalOutput,
    InteractionLog,
    InteractionOutput,
    Policy,
    Span,
)

# The eval UI modules import `plotly.express` at module load time, so a stub must
# be present before the imports below (a fixture runs too late: these imports
# happen during collection). We only inject the stub when plotly cannot really be
# imported, so an installed plotly is never shadowed, and we remove our own stubs
# after this module's tests so they do not leak into other files on the worker.
_injected_plotly_modules = {}
try:
    import plotly.express  # noqa: F401
except ImportError:
    for _module_name in ("plotly", "plotly.express"):
        if _module_name not in sys.modules:
            _injected_plotly_modules[_module_name] = types.ModuleType(_module_name)
            sys.modules[_module_name] = _injected_plotly_modules[_module_name]


@pytest.fixture(scope="module", autouse=True)
def _cleanup_plotly_stub():
    yield
    for _module_name, stub_module in _injected_plotly_modules.items():
        if sys.modules.get(_module_name) is stub_module:
            del sys.modules[_module_name]
    # The eval UI modules below were imported while the plotly stub was active, so
    # drop them too; otherwise they stay cached bound to the stubbed plotly.express
    # and a later import in this worker would not rebind the real dependency.
    if _injected_plotly_modules:
        for _ui_module_name in ("nemoguardrails.eval.ui.common", "nemoguardrails.eval.ui.chart_utils"):
            sys.modules.pop(_ui_module_name, None)


ui_common = importlib.import_module("nemoguardrails.eval.ui.common")
ui_utils = importlib.import_module("nemoguardrails.eval.ui.utils")
chart_utils = importlib.import_module("nemoguardrails.eval.ui.chart_utils")
streamlit_utils = importlib.import_module("nemoguardrails.eval.ui.streamlit_utils")
readme_page = importlib.import_module("nemoguardrails.eval.ui.README")

_get_compliance_df = ui_common._get_compliance_df
_get_resource_usage_and_latencies_df = ui_common._get_resource_usage_and_latencies_df
EvalData = ui_utils.EvalData
collect_interaction_metrics = ui_utils.collect_interaction_metrics
collect_interaction_metrics_with_expected_latencies = ui_utils.collect_interaction_metrics_with_expected_latencies


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Sidebar(_Context):
    def expander(self, *args, **kwargs):
        return _Context()


class _FakeStreamlit:
    def __init__(self, checkbox_values=None, button_values=None):
        self.checkbox_values = list(checkbox_values or [])
        self.button_values = list(button_values or [])
        self.session_state = SimpleNamespace(use_expected_latencies=False)
        self.sidebar = _Sidebar()
        self.calls = []

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def checkbox(self, *args, **kwargs):
        self._record("checkbox", *args, **kwargs)
        return self.checkbox_values.pop(0) if self.checkbox_values else True

    def button(self, *args, **kwargs):
        self._record("button", *args, **kwargs)
        return self.button_values.pop(0) if self.button_values else False

    def expander(self, *args, **kwargs):
        self._record("expander", *args, **kwargs)
        return _Context()

    def rerun(self):
        self._record("rerun")

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self._record(name, *args, **kwargs)

        return method


def _eval_data():
    eval_config = EvalConfig(
        policies=[Policy(id="p1", description="one"), Policy(id="p2", description="two")],
        interactions=[
            {
                "id": "1",
                "inputs": ["a"],
                "expected_output": [],
                "tags": ["keep"],
            },
            {
                "id": "2",
                "inputs": ["b"],
                "expected_output": [],
                "tags": ["drop"],
            },
        ],
        expected_latencies={
            "llm_call_main_fixed_latency": 1.0,
            "llm_call_main_prompt_token_latency": 0.5,
            "llm_call_main_completion_token_latency": 0.25,
        },
    )
    output = EvalOutput(
        results=[
            InteractionOutput(
                id="1",
                input="a",
                compliance={"p1": True, "p2": False},
                resource_usage={"llm_call_main_total": 1, "tokens_total": 10},
                latencies={"llm_call_main_seconds_avg": 2.0},
            ),
            InteractionOutput(
                id="2",
                input="b",
                compliance={"p1": False, "p2": "n/a"},
                resource_usage={"llm_call_main_total": 1, "tokens_total": 5},
                latencies={"llm_call_main_seconds_avg": 4.0},
            ),
        ],
        logs=[
            InteractionLog(
                id="1",
                trace=[
                    Span(
                        span_id="interaction",
                        name="interaction",
                        start_time=0,
                        end_time=2,
                        duration=2,
                        metrics={
                            "interaction_seconds_avg": 2.0,
                            "interaction_seconds_total": 2.0,
                        },
                    ),
                    Span(
                        span_id="llm",
                        parent_id="interaction",
                        name="LLM: main",
                        start_time=0,
                        end_time=2,
                        duration=2,
                        metrics={
                            "llm_call_main_total": 1,
                            "llm_call_main_seconds_avg": 2.0,
                            "llm_call_main_seconds_total": 2.0,
                            "llm_call_main_prompt_tokens_total": 4,
                            "llm_call_main_completion_tokens_total": 2,
                            "llm_call_main_tokens_total": 6,
                        },
                    ),
                ],
            ),
            InteractionLog(id="2"),
        ],
    )
    return EvalData(
        eval_config_path="config",
        eval_config=eval_config,
        output_paths=["run-a"],
        eval_outputs={"run-a": output},
    )


def test_collect_interaction_metrics_sums_resource_usage_and_averages_latency():
    metrics = collect_interaction_metrics(_eval_data().eval_outputs["run-a"].results)

    assert metrics["llm_call_main_total"] == 2
    assert metrics["tokens_total"] == 15
    assert metrics["llm_call_main_seconds_avg"] == 3.0


def test_collect_interaction_metrics_with_expected_latencies_recomputes_llm_spans():
    eval_data = _eval_data()

    metrics = collect_interaction_metrics_with_expected_latencies(
        [eval_data.eval_outputs["run-a"].results[0]],
        eval_data.eval_outputs["run-a"].logs,
        eval_data.eval_config.expected_latencies,
    )

    assert metrics["llm_call_main_seconds_avg"] == pytest.approx(3.5)
    assert metrics["interaction_seconds_avg"] == pytest.approx(3.5)
    assert metrics["tokens_total"] == 10


def test_get_compliance_df_builds_policy_rows():
    df = _get_compliance_df(["run-a"], ["p1", "p2"], _eval_data())

    rows = df.to_dict("records")
    assert rows == [
        {
            "Guardrail Config": "run-a",
            "Policy": "p1",
            "Compliance Rate": 50.0,
            "Violations Count": 1,
            "Interactions Count": 2,
        },
        {
            "Guardrail Config": "run-a",
            "Policy": "p2",
            "Compliance Rate": 0.0,
            "Violations Count": 1,
            "Interactions Count": 1,
        },
    ]


def test_get_resource_usage_and_latencies_df_splits_metric_tables():
    eval_data = _eval_data()

    resource_df, latency_df = _get_resource_usage_and_latencies_df(
        ["run-a"],
        eval_data,
        eval_data.eval_config,
        use_expected_latencies=False,
    )

    assert resource_df.to_dict("records") == [
        {"Metric": "llm_call_main_total", "run-a": 2},
        {"Metric": "tokens_total", "run-a": 15},
    ]
    assert latency_df.to_dict("records") == [{"Metric": "llm_call_main_seconds_avg", "run-a": 3.0}]


def test_eval_data_update_methods_delegate_to_update_dict_at_path():
    eval_data = _eval_data()
    eval_data.selected_output_path = "run-a"
    calls = []

    def fake_update(path, data):
        calls.append((path, data))

    from nemoguardrails.eval.ui import utils as ui_utils

    original = ui_utils.update_dict_at_path
    ui_utils.update_dict_at_path = fake_update
    try:
        eval_data.update_results()
        eval_data.update_results_and_logs("run-a")
        eval_data.update_config_latencies()
    finally:
        ui_utils.update_dict_at_path = original

    assert calls[0][0] == "run-a"
    assert "results" in calls[0][1]
    assert calls[1][0] == "run-a"
    assert set(calls[1][1]) == {"results", "logs"}
    assert calls[2] == ("config", {"expected_latencies": eval_data.eval_config.expected_latencies})


def test_chart_utils_render_charts_and_optional_tables(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_bar = MagicMock(return_value="figure")
    monkeypatch.setattr(chart_utils, "st", fake_st)
    monkeypatch.setattr(chart_utils.px, "bar", fake_bar, raising=False)

    import pandas as pd

    chart_utils.plot_as_series(pd.DataFrame({"Config": ["a"], "Value": [1]}), include_table=True)
    chart_utils.plot_bar_series(pd.DataFrame({"Config": ["a"], "Metric": ["m"], "Value": [1]}), include_table=True)
    chart_utils.plot_matrix_series(
        pd.DataFrame({"Metric": ["m"], "run-a": [1]}),
        var_name="Guardrail Config",
        value_name="Value",
        include_table=True,
    )

    assert fake_bar.call_count == 3
    assert [call[0] for call in fake_st.calls].count("plotly_chart") == 3
    assert [call[0] for call in fake_st.calls].count("dataframe") == 3


def test_streamlit_utils_get_span_colors_is_stable():
    output = EvalOutput(
        logs=[
            InteractionLog(
                id="1",
                trace=[
                    Span(span_id="1", name="interaction", start_time=0, end_time=1, duration=1),
                    Span(span_id="2", name="rail", start_time=0, end_time=1, duration=1),
                ],
            ),
            InteractionLog(
                id="2",
                trace=[Span(span_id="3", name="interaction", start_time=0, end_time=1, duration=1)],
            ),
        ]
    )

    colors = streamlit_utils.get_span_colors(output)

    assert set(colors) == {"interaction", "rail"}
    assert all(value.startswith("#") and len(value) == 7 for value in colors.values())


def test_streamlit_utils_load_eval_data_uses_discovered_paths(tmp_path, monkeypatch):
    eval_config = EvalConfig(policies=[Policy(id="p1", description="one")], interactions=[])
    eval_output = EvalOutput(results=[InteractionOutput(id="1/0", input="a")], logs=[InteractionLog(id="1/0")])
    hidden_output = EvalOutput(results=[InteractionOutput(id="2/0", input="b")], logs=[InteractionLog(id="2/0")])
    output_dir = tmp_path / "run-a"
    hidden_dir = tmp_path / ".hidden"
    output_dir.mkdir()
    hidden_dir.mkdir()

    with (
        patch("sys.argv", ["streamlit", "--eval-config-path", str(tmp_path / "config")]),
        patch.object(streamlit_utils.EvalConfig, "from_path", return_value=eval_config) as mock_config,
        patch.object(streamlit_utils.EvalOutput, "from_path", side_effect=[eval_output, hidden_output]) as mock_output,
        patch.object(streamlit_utils, "get_output_paths", return_value=[str(output_dir), str(hidden_dir)]),
    ):
        monkeypatch.chdir(tmp_path)
        streamlit_utils.load_eval_data.clear()
        data = streamlit_utils.load_eval_data()

    mock_config.assert_called_once_with(str((tmp_path / "config").resolve()))
    assert mock_output.call_count == 1
    assert data.output_paths == [str(output_dir), str(hidden_dir)]
    assert data.eval_outputs == {"run-a": eval_output}


def test_eval_readme_page_renders_markdown(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(readme_page, "st", fake_st)

    readme_page.main()

    markdown_calls = [call for call in fake_st.calls if call[0] == "markdown"]
    assert len(markdown_calls) == 1
    assert markdown_calls[0][2]["unsafe_allow_html"] is True


def test_render_sidebar_filters_and_reload(monkeypatch):
    fake_st = _FakeStreamlit(
        checkbox_values=[True, True, False, True, False, False, True],
        button_values=[True],
    )
    clear = MagicMock()
    monkeypatch.setattr(ui_common, "st", fake_st)
    monkeypatch.setattr(ui_common.load_eval_data, "clear", clear)

    output_names, policy_options, tags = ui_common._render_sidebar(["run-a", "run-b"], ["p1", "p2"], ["keep", "drop"])

    assert output_names == ["run-a"]
    assert policy_options == ["p1"]
    assert tags == ["drop"]
    clear.assert_called_once()
    assert any(call[0] == "rerun" for call in fake_st.calls)


def test_render_compliance_data_full_and_short(monkeypatch):
    fake_st = _FakeStreamlit()
    plot_calls = []
    monkeypatch.setattr(ui_common, "st", fake_st)
    monkeypatch.setattr(ui_common, "plot_as_series", lambda *args, **kwargs: plot_calls.append(("series", kwargs)))
    monkeypatch.setattr(ui_common, "plot_bar_series", lambda *args, **kwargs: plot_calls.append(("bar", kwargs)))

    ui_common._render_compliance_data(["run-a"], ["p1", "p2"], _eval_data(), short=False)
    ui_common._render_compliance_data(["run-a"], ["p1"], _eval_data(), short=True)

    assert any(call[0] == "info" for call in fake_st.calls)
    assert [call[0] for call in plot_calls].count("series") == 2
    assert [call[0] for call in plot_calls].count("bar") == 3


def test_render_resource_usage_and_latencies_full_and_short(monkeypatch):
    fake_st = _FakeStreamlit(checkbox_values=[True, True])
    fake_st.session_state.use_expected_latencies = False
    plot_calls = []
    monkeypatch.setattr(ui_common, "st", fake_st)
    monkeypatch.setattr(ui_common, "plot_as_series", lambda *args, **kwargs: plot_calls.append(("series", kwargs)))
    monkeypatch.setattr(ui_common, "plot_bar_series", lambda *args, **kwargs: plot_calls.append(("bar", kwargs)))
    monkeypatch.setattr(ui_common, "plot_matrix_series", lambda *args, **kwargs: plot_calls.append(("matrix", kwargs)))

    eval_data = _eval_data()
    for result in eval_data.eval_outputs["run-a"].results:
        result.resource_usage.update(
            {
                "llm_call_aux_total": 1,
                "llm_call_main_prompt_tokens_total": 4,
                "llm_call_main_completion_tokens_total": 2,
                "llm_call_main_tokens_total": 6,
            }
        )
        result.latencies.update(
            {
                "interaction_seconds_total": 2.0,
                "interaction_seconds_avg": 1.0,
                "llm_call_main_seconds_total": 1.0,
                "llm_call_main_seconds_avg": 1.0,
                "action_self_check_seconds_total": 1.0,
                "action_self_check_seconds_avg": 1.0,
            }
        )

    ui_common._render_resource_usage_and_latencies(["run-a"], eval_data, eval_data.eval_config, short=False)
    ui_common._render_resource_usage_and_latencies(["run-a"], eval_data, eval_data.eval_config, short=True)

    assert any(call[0] == "dataframe" for call in fake_st.calls)
    assert [call[0] for call in plot_calls].count("matrix") >= 5
    assert [call[0] for call in plot_calls].count("series") >= 3


def test_render_summary_filters_by_selected_tags(monkeypatch):
    eval_data = _eval_data()
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(ui_common, "st", fake_st)
    monkeypatch.setattr(ui_common, "load_eval_data", MagicMock(return_value=eval_data))
    monkeypatch.setattr(
        ui_common,
        "_render_sidebar",
        MagicMock(return_value=(["run-a"], ["p1"], ["keep"])),
    )
    render_compliance = MagicMock()
    render_resources = MagicMock()
    monkeypatch.setattr(ui_common, "_render_compliance_data", render_compliance)
    monkeypatch.setattr(ui_common, "_render_resource_usage_and_latencies", render_resources)

    ui_common.render_summary(short=True)

    filtered_eval_data = render_compliance.call_args.args[2]
    assert [result.id for result in filtered_eval_data.eval_outputs["run-a"].results] == ["1"]
    render_resources.assert_called_once()
