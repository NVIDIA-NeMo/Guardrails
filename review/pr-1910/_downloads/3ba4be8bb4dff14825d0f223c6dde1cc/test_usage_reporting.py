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

import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails import telemetry
from nemoguardrails.telemetry import (
    GuardrailsUsageEvent,
    _build_nvidia_payload,
    _collect_usage_data,
    _detect_builtin_features,
    _get_heartbeat_interval_s,
    _is_usage_stats_enabled,
    _rotate_audit_file,
    _send_report,
    _write_audit_file,
    report_usage,
    set_deployment_type,
)
from scripts import kibana_verify_export, telemetry_smoke


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    telemetry._session_uuid = None
    telemetry._heartbeat_started = False
    telemetry._deployment_type_override = None
    telemetry._lock = telemetry.threading.Lock()
    telemetry._CONFIG_DIR = None
    telemetry._AUDIT_FILE = None
    telemetry._DO_NOT_TRACK_FILE = None
    yield
    telemetry._session_uuid = None
    telemetry._heartbeat_started = False
    telemetry._deployment_type_override = None
    telemetry._lock = telemetry.threading.Lock()
    telemetry._CONFIG_DIR = None
    telemetry._AUDIT_FILE = None
    telemetry._DO_NOT_TRACK_FILE = None


def _send_event_calls(mock_thread):
    return [
        call
        for call in mock_thread.call_args_list
        if getattr(call.kwargs.get("target"), "__name__", "") == "_send_one_event"
    ]


def _heartbeat_calls(mock_thread):
    return [
        call
        for call in mock_thread.call_args_list
        if getattr(call.kwargs.get("target"), "__name__", "") == "_heartbeat_loop"
    ]


@contextmanager
def _without_pytest_module():
    pytest_module = sys.modules.pop("pytest", None)
    try:
        yield
    finally:
        if pytest_module is not None:
            sys.modules["pytest"] = pytest_module


@pytest.fixture()
def audit_dir(tmp_path):
    config_dir = tmp_path / ".config" / "nemoguardrails"
    audit_file = config_dir / "usage_stats.json"
    with patch.object(telemetry, "_CONFIG_DIR", config_dir), patch.object(telemetry, "_AUDIT_FILE", audit_file):
        yield config_dir, audit_file


@pytest.fixture()
def mock_config():
    config = MagicMock()
    config.colang_version = "2.x"

    model1 = MagicMock()
    model1.engine = "openai"
    model2 = MagicMock()
    model2.engine = "nvidia_ai_endpoints"
    config.models = [model1, model2]

    config.rails.input.flows = ["check_jailbreak"]
    config.rails.output.flows = ["check_output"]
    config.rails.retrieval.flows = []
    config.rails.tool_output.flows = []
    config.rails.tool_input.flows = []
    config.rails.dialog.single_call.enabled = False
    config.rails.output.streaming.enabled = True

    config.tracing.enabled = True
    config.docs = [MagicMock()]

    return config


class TestOptOut:
    def test_enabled_by_default(self):
        with (
            _without_pytest_module(),
            patch.dict(os.environ, {}, clear=True),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", Path("/nonexistent/path")),
        ):
            assert _is_usage_stats_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE"])
    def test_disabled_by_nemo_env_var(self, value):
        with _without_pytest_module(), patch.dict(os.environ, {"NEMO_GUARDRAILS_NO_USAGE_STATS": value}):
            assert _is_usage_stats_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE"])
    def test_disabled_by_do_not_track(self, value):
        with _without_pytest_module(), patch.dict(os.environ, {"DO_NOT_TRACK": value}):
            assert _is_usage_stats_enabled() is False

    def test_disabled_by_file(self, tmp_path):
        do_not_track = tmp_path / "do_not_track"
        do_not_track.touch()
        with (
            _without_pytest_module(),
            patch.dict(os.environ, {}, clear=True),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", do_not_track),
        ):
            assert _is_usage_stats_enabled() is False

    def test_path_helpers_fallback_when_home_is_unavailable(self, tmp_path):
        with (
            patch("nemoguardrails.telemetry.Path.home", side_effect=RuntimeError("home unavailable")),
            patch.dict(os.environ, {"HOME": str(tmp_path)}, clear=True),
        ):
            assert telemetry._get_config_dir() == tmp_path / ".config" / "nemoguardrails"
            assert telemetry._get_audit_file() == tmp_path / ".config" / "nemoguardrails" / "usage_stats.json"
            assert telemetry._get_do_not_track_file() == tmp_path / ".config" / "nemoguardrails" / "do_not_track"

    def test_not_disabled_when_env_var_is_zero(self):
        with (
            _without_pytest_module(),
            patch.dict(os.environ, {"NEMO_GUARDRAILS_NO_USAGE_STATS": "0"}, clear=True),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", Path("/nonexistent/path")),
        ):
            assert _is_usage_stats_enabled() is True

    def test_disabled_in_ci(self):
        with _without_pytest_module(), patch.dict(os.environ, {"CI": "true"}, clear=True):
            assert _is_usage_stats_enabled() is False

    def test_disabled_when_ci_is_one(self):
        with _without_pytest_module(), patch.dict(os.environ, {"CI": "1"}, clear=True):
            assert _is_usage_stats_enabled() is False

    def test_disabled_under_pytest(self):
        with _without_pytest_module(), patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "test_x"}, clear=True):
            assert _is_usage_stats_enabled() is False

    def test_disabled_when_pytest_module_loaded(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", Path("/nonexistent/path")),
        ):
            assert _is_usage_stats_enabled() is False

    def test_heartbeat_interval_env_var(self):
        with patch.dict(os.environ, {"NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S": "0.5"}, clear=True):
            assert _get_heartbeat_interval_s() == 0.5

    @pytest.mark.parametrize("value", ["", "bad", "0", "-1"])
    def test_invalid_heartbeat_interval_env_var_uses_default(self, value):
        with patch.dict(os.environ, {"NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S": value}, clear=True):
            assert _get_heartbeat_interval_s() == 600


class TestDataCollection:
    def test_collect_without_config(self):
        data = _collect_usage_data(None, "api")
        assert data.deployment_type == "api"
        assert data.event == "startup"
        assert data.python_version != ""
        assert data.platform != ""
        assert data.os_name != ""
        assert data.colang_version == "unknown"
        assert data.llm_providers == []
        assert data.num_rails_configured == 0

    def test_collect_with_config(self, mock_config):
        data = _collect_usage_data(mock_config, "library")
        assert data.deployment_type == "library"
        assert data.colang_version == "2.x"
        assert data.llm_providers == ["nvidia_ai_endpoints", "openai"]
        assert data.num_rails_configured == 2
        assert "input" in data.rail_types_in_use
        assert "output" in data.rail_types_in_use
        assert data.tracing_enabled is True
        assert data.has_knowledge_base is True
        assert data.streaming_configured is True

    def test_collect_with_rails_config_default_colang_version(self):
        from nemoguardrails.rails.llm.config import RailsConfig

        data = _collect_usage_data(RailsConfig(models=[]), "library")

        assert data.colang_version == "1.0"

    def test_engine_names_not_model_names(self, mock_config):
        data = _collect_usage_data(mock_config, "library")
        assert "openai" in data.llm_providers
        assert "nvidia_ai_endpoints" in data.llm_providers

    def test_rail_types_detected(self, mock_config):
        mock_config.rails.input.flows = []
        mock_config.rails.output.flows = ["some_flow"]
        data = _collect_usage_data(mock_config, "library")
        assert data.rail_types_in_use == ["output"]
        assert data.num_rails_configured == 1

    def test_session_id_is_unique(self):
        d1 = _collect_usage_data(None, "library")
        d2 = _collect_usage_data(None, "library")
        assert d1.session_id != d2.session_id

    def test_all_fields_serializable(self):
        data = _collect_usage_data(None, "library")
        payload = data.model_dump()
        for key, value in payload.items():
            if isinstance(value, list):
                for item in value:
                    assert isinstance(item, (str, int, float, bool)), (
                        f"Field {key} list contains non-primitive {type(item)}"
                    )
            else:
                assert isinstance(value, (str, int, float, bool)), f"Field {key} has unexpected type {type(value)}"

    def test_camel_case_aliases(self):
        data = _collect_usage_data(None, "library")
        payload = data.model_dump(by_alias=True)
        assert "sessionId" in payload
        assert "nemoSource" in payload
        assert "nemoguardrailsVersion" in payload
        assert "llmProviders" in payload
        assert "numRailsConfigured" in payload
        assert "railTypesInUse" in payload
        assert "deploymentType" in payload
        assert "hasKnowledgeBase" in payload
        assert "builtinFeatures" in payload
        assert "numCustomFlows" in payload
        assert "railsEngine" in payload

    def test_nemo_source_field_default_is_guardrails(self):
        data = _collect_usage_data(None, "library")
        assert data.nemo_source == "guardrails"

    def test_session_id_unprefixed_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            data = _collect_usage_data(None, "library")
        assert "-" in data.session_id
        assert not data.session_id.startswith("smoke-")


class TestRailsEngine:
    def test_default_is_undefined(self):
        data = _collect_usage_data(None, "api")
        assert data.rails_engine == "undefined"

    def test_set_via_report_usage(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library", rails_engine="LLMRails")
            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 1
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.rails_engine == "LLMRails"

    def test_iorails_engine(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library", rails_engine="IORails")
            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 1
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.rails_engine == "IORails"

    def test_direct_iorails_reports_iorails_engine(self):
        from nemoguardrails.guardrails.iorails import IORails
        from nemoguardrails.rails.llm.config import RailsConfig
        from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

        config = RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)
        with patch("nemoguardrails.telemetry.report_usage") as mock_report:
            IORails(config)

        mock_report.assert_called_once_with(
            config,
            deployment_type="library",
            rails_engine=telemetry.RailsEngineEnum.IORAILS.value,
        )

    def test_guardrails_routed_to_iorails_reports_once(self):
        from nemoguardrails.guardrails.guardrails import Guardrails
        from nemoguardrails.rails.llm.config import RailsConfig
        from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

        config = RailsConfig.from_content(config=CONTENT_SAFETY_CONFIG)
        with patch("nemoguardrails.telemetry.report_usage") as mock_report:
            Guardrails(config)

        mock_report.assert_called_once_with(
            config,
            deployment_type="library",
            rails_engine=telemetry.RailsEngineEnum.IORAILS.value,
        )


class TestAuditFile:
    def test_write_creates_directory(self, tmp_path):
        config_dir = tmp_path / "new" / "nested" / "dir"
        audit_file = config_dir / "usage_stats.json"
        with patch.object(telemetry, "_CONFIG_DIR", config_dir), patch.object(telemetry, "_AUDIT_FILE", audit_file):
            _write_audit_file({"test": "data"})
            assert audit_file.exists()
            lines = audit_file.read_text().strip().split("\n")
            assert len(lines) == 1
            assert json.loads(lines[0]) == {"test": "data"}

    def test_write_appends_jsonl(self, audit_dir):
        _, audit_file = audit_dir
        _write_audit_file({"event": "first"})
        _write_audit_file({"event": "second"})
        lines = audit_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event"] == "first"
        assert json.loads(lines[1])["event"] == "second"

    def test_rotation_at_cap(self, audit_dir):
        config_dir, audit_file = audit_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        audit_file.write_text("x" * (telemetry._AUDIT_FILE_MAX_BYTES + 1))
        _write_audit_file({"event": "after_rotation"})
        backup = audit_file.with_suffix(".json.1")
        assert backup.exists()
        assert audit_file.exists()
        content = audit_file.read_text().strip()
        assert json.loads(content)["event"] == "after_rotation"

    def test_readonly_dir_silent_fail(self, tmp_path):
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o444)
        config_dir = readonly / "nemoguardrails"
        audit_file = config_dir / "usage_stats.json"
        try:
            with patch.object(telemetry, "_CONFIG_DIR", config_dir), patch.object(telemetry, "_AUDIT_FILE", audit_file):
                _write_audit_file({"test": "data"})
        finally:
            readonly.chmod(0o755)


class TestTransport:
    def test_send_report_nvidia_envelope(self):
        event = GuardrailsUsageEvent(session_id="test-uuid", timestamp=1700000000.0)
        with patch("nemoguardrails.telemetry.urllib.request.urlopen") as mock_urlopen:
            _send_report(event, "https://example.com/stats", "0.21.0", "test-session")
            mock_urlopen.assert_called_once()
            call_args = mock_urlopen.call_args
            req = call_args[0][0]
            assert req.full_url == "https://example.com/stats"
            assert req.method == "POST"
            envelope = json.loads(req.data)
            assert envelope["clientId"] == "184482118588404"
            assert envelope["clientVer"] == "0.21.0"
            assert envelope["sessionId"] == "test-session"
            assert envelope["eventProtocol"] == "1.6"
            assert envelope["eventSchemaVer"] == "1.7"
            assert len(envelope["events"]) == 1
            ev = envelope["events"][0]
            assert ev["name"] == "guardrails_usage_event"
            assert ev["parameters"]["nemoSource"] == "guardrails"
            assert ev["parameters"]["sessionId"] == "test-uuid"

    def test_build_payload_rejects_mismatched_timestamps(self):
        events = [
            GuardrailsUsageEvent(session_id="one"),
            GuardrailsUsageEvent(session_id="two"),
        ]
        with pytest.raises(ValueError, match="timestamps length must match events length"):
            _build_nvidia_payload(events, "0.21.0", "test-session", timestamps=[1700000000.0])

    def test_send_report_failure_silent(self):
        event = GuardrailsUsageEvent(session_id="test")
        with patch(
            "nemoguardrails.telemetry.urllib.request.urlopen",
            side_effect=Exception("connection refused"),
        ):
            _send_report(event, "https://example.com/stats", "0.21.0", "s")

    def test_custom_server_url(self):
        with (
            patch.dict(
                os.environ,
                {"NEMO_GUARDRAILS_USAGE_STATS_SERVER": "https://custom.server/v1"},
            ),
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library")
            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 1
            assert send_calls[0].kwargs["args"][1] == "https://custom.server/v1"


class TestIntegration:
    def test_report_spawns_daemon_threads(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            report_usage(None, deployment_type="library")

            send_calls = _send_event_calls(mock_thread)
            heartbeat_calls = _heartbeat_calls(mock_thread)
            assert len(send_calls) == 1
            assert len(heartbeat_calls) == 1
            for call in send_calls + heartbeat_calls:
                assert call.kwargs["daemon"] is True
            assert mock_instance.start.call_count == 2

    def test_report_swallows_thread_start_failure(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value.start.side_effect = RuntimeError("thread limit reached")

            report_usage(None, deployment_type="library")

            assert mock_thread.return_value.start.called
            assert telemetry._heartbeat_started is False

    def test_send_thread_start_failure_does_not_block_heartbeat_start(self):
        created_targets = []

        def thread_factory(*args, **kwargs):
            target = kwargs["target"]
            created_targets.append(target.__name__)
            thread = MagicMock()
            if target.__name__ == "_send_one_event":
                thread.start.side_effect = RuntimeError("send thread limit reached")
            return thread

        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread", side_effect=thread_factory),
        ):
            report_usage(None, deployment_type="library")

        assert created_targets == ["_send_one_event", "_heartbeat_loop"]
        assert telemetry._heartbeat_started is True

    def test_heartbeat_thread_start_failure_retries_on_next_report(self):
        heartbeat_attempts = 0

        def thread_factory(*args, **kwargs):
            nonlocal heartbeat_attempts
            target = kwargs["target"]
            thread = MagicMock()
            if target.__name__ == "_heartbeat_loop":
                heartbeat_attempts += 1
                if heartbeat_attempts == 1:
                    thread.start.side_effect = RuntimeError("heartbeat thread limit reached")
            return thread

        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread", side_effect=thread_factory),
        ):
            report_usage(None, deployment_type="library")
            assert telemetry._heartbeat_started is False

            report_usage(None, deployment_type="library")
            assert telemetry._heartbeat_started is True

        assert heartbeat_attempts == 2

    def test_each_call_emits_event_with_shared_session_id(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library")
            report_usage(None, deployment_type="api")

            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 2
            first_event = send_calls[0].kwargs["args"][0]
            second_event = send_calls[1].kwargs["args"][0]
            assert first_event.session_id == second_event.session_id

    def test_heartbeat_thread_started_only_once(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library")
            report_usage(None, deployment_type="library")
            report_usage(None, deployment_type="library")

            assert len(_send_event_calls(mock_thread)) == 3
            assert len(_heartbeat_calls(mock_thread)) == 1

    def test_set_deployment_type_overrides_argument(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            set_deployment_type("api")
            report_usage(None, deployment_type="library")

            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 1
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.deployment_type == "api"

    def test_set_deployment_type_invalid_value_ignored(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            set_deployment_type("not_a_real_value")
            assert telemetry._deployment_type_override is None
            report_usage(None, deployment_type="library")

            send_calls = _send_event_calls(mock_thread)
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.deployment_type == "library"

    def test_server_lifespan_sets_api_deployment_type_before_config_init(self, tmp_path):
        from nemoguardrails.server.api import GuardrailsApp, lifespan

        (tmp_path / "config.py").write_text(
            "def init(app):\n"
            "    from nemoguardrails.telemetry import report_usage\n"
            "    report_usage(None, deployment_type='library')\n"
        )
        server_app = GuardrailsApp(lifespan=lifespan)
        server_app.rails_config_path = str(tmp_path)

        async def run_lifespan():
            async with lifespan(server_app):
                pass

        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            asyncio.run(run_lifespan())

        send_calls = _send_event_calls(mock_thread)
        assert len(send_calls) == 1
        usage_data = send_calls[0].kwargs["args"][0]
        assert usage_data.deployment_type == "api"

    def test_report_skipped_when_disabled(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=False),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, deployment_type="library")
            mock_thread.assert_not_called()

    def test_report_skipped_under_pytest_even_with_staging_env(self, tmp_path):
        audit_file = tmp_path / "usage_stats.json"
        with (
            patch.dict(
                os.environ,
                {"NEMO_GUARDRAILS_USAGE_STATS_SERVER": "https://staging.example/v1.1/events/json"},
                clear=True,
            ),
            patch.object(telemetry, "_AUDIT_FILE", audit_file),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", tmp_path / "do_not_track"),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, deployment_type="library")

        mock_thread.assert_not_called()
        assert not audit_file.exists()

    def test_report_swallows_post_collection_errors(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch.object(telemetry, "_get_usage_stats_server_url", side_effect=RuntimeError("env unavailable")),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, deployment_type="library")

        mock_thread.assert_not_called()

    def test_report_swallows_usage_stats_enabled_errors(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", side_effect=RuntimeError("opt-out check failed")),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, deployment_type="library")

        mock_thread.assert_not_called()

    @pytest.mark.parametrize(
        "env",
        [
            {"CI": "true"},
            {"CI": "1"},
            {"PYTEST_CURRENT_TEST": "test_telemetry.py::test_x"},
        ],
    )
    def test_report_skipped_by_test_environment_signals(self, tmp_path, env):
        audit_file = tmp_path / "usage_stats.json"
        with (
            _without_pytest_module(),
            patch.dict(os.environ, env, clear=True),
            patch.object(telemetry, "_AUDIT_FILE", audit_file),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", tmp_path / "do_not_track"),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, deployment_type="library")

        mock_thread.assert_not_called()
        assert not audit_file.exists()

    def test_send_one_event_writes_and_posts(self):
        data = GuardrailsUsageEvent(
            session_id="test-uuid-123",
            python_version="3.13.7",
            platform="test-platform",
        )
        payloads = []

        with (
            patch.object(telemetry, "_write_audit_file", side_effect=lambda d: payloads.append(d)),
            patch.object(telemetry, "_send_report") as mock_send,
        ):
            telemetry._send_one_event(data, "https://example.com", "0.21.0", "test-uuid-123")

        assert len(payloads) == 1
        assert payloads[0]["event"] == "startup"
        assert payloads[0]["sessionId"] == "test-uuid-123"
        assert payloads[0]["pythonVersion"] == "3.13.7"
        mock_send.assert_called_once_with(data, "https://example.com", "0.21.0", "test-uuid-123")

    def test_heartbeat_loop_reuses_startup_metadata(self):
        startup_event = GuardrailsUsageEvent(
            session_id="startup-session",
            timestamp=111.0,
            python_version="3.13.7",
            platform="test-platform",
            os_name="Darwin",
            deployment_type=telemetry.DeploymentTypeEnum.API,
            rails_engine=telemetry.RailsEngineEnum.LLMRAILS,
            llm_providers=["openai"],
            num_rails_configured=1,
            rail_types_in_use=["input"],
            builtin_features=["self_check"],
        )
        startup_payload = startup_event.model_dump(by_alias=True)
        payloads = []

        def mock_write(payload):
            payloads.append(payload)

        def mock_sleep(seconds):
            if len(payloads) >= 2:
                raise SystemExit()

        with (
            patch.object(telemetry, "_write_audit_file", side_effect=mock_write),
            patch.object(telemetry, "_send_report"),
            patch("nemoguardrails.telemetry.time.time", side_effect=[222.0, 333.0]),
            patch("nemoguardrails.telemetry.time.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(SystemExit):
                telemetry._heartbeat_loop(startup_event, "test-uuid-123", "0.21.0")

        assert len(payloads) >= 2
        for payload in payloads:
            assert payload["event"] == "heartbeat"
            assert payload["sessionId"] == "test-uuid-123"
            assert payload["pythonVersion"] == "3.13.7"
            changed_fields = {
                key for key, startup_value in startup_payload.items() if payload.get(key) != startup_value
            }
            assert changed_fields == {"event", "sessionId", "timestamp"}

    def test_heartbeat_loop_sleep_adds_jitter_without_shortening_interval(self):
        startup_event = GuardrailsUsageEvent(session_id="startup-session", timestamp=111.0)
        sleeps = []

        def mock_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) >= 2:
                raise SystemExit()

        with (
            patch.object(telemetry, "_HEARTBEAT_INTERVAL_S", 100.0),
            patch("nemoguardrails.telemetry.random.uniform", return_value=7.5) as mock_uniform,
            patch.object(telemetry, "_write_audit_file"),
            patch.object(telemetry, "_send_report"),
            patch("nemoguardrails.telemetry.time.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(SystemExit):
                telemetry._heartbeat_loop(startup_event, "test-uuid-123", "0.21.0")

        assert sleeps == [107.5, 107.5]
        mock_uniform.assert_called_with(0, 10.0)

    def test_heartbeat_loop_uses_current_server_url_each_tick(self):
        sent_urls = []

        def mock_send(event, server_url, client_version, session_id):
            sent_urls.append(server_url)
            if len(sent_urls) == 1:
                os.environ["NEMO_GUARDRAILS_USAGE_STATS_SERVER"] = "https://second.example/events"

        def mock_sleep(seconds):
            if len(sent_urls) >= 2:
                raise SystemExit()

        with (
            patch.dict(
                os.environ,
                {"NEMO_GUARDRAILS_USAGE_STATS_SERVER": "https://first.example/events"},
                clear=True,
            ),
            patch.object(telemetry, "_write_audit_file"),
            patch.object(telemetry, "_send_report", side_effect=mock_send),
            patch("nemoguardrails.telemetry.time.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(SystemExit):
                telemetry._heartbeat_loop(
                    GuardrailsUsageEvent(session_id="startup-session"),
                    "test-uuid-123",
                    "0.21.0",
                )

        assert sent_urls == ["https://first.example/events", "https://second.example/events"]

    def test_heartbeat_loop_survives_iteration_errors(self):
        iterations = [0]

        def flaky_write(payload):
            iterations[0] += 1
            if iterations[0] == 1:
                raise RuntimeError("transient disk failure")

        def mock_sleep(seconds):
            if iterations[0] >= 3:
                raise SystemExit()

        with (
            patch.object(telemetry, "_write_audit_file", side_effect=flaky_write),
            patch.object(telemetry, "_send_report"),
            patch("nemoguardrails.telemetry.time.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(SystemExit):
                telemetry._heartbeat_loop(
                    GuardrailsUsageEvent(session_id="startup-session"),
                    "test-uuid-123",
                    "0.21.0",
                )

        assert iterations[0] >= 2

    def test_reset_for_fork_clears_session_state(self):
        telemetry._session_uuid = "old-session-uuid"
        telemetry._heartbeat_started = True
        telemetry._deployment_type_override = telemetry.DeploymentTypeEnum.API

        telemetry._reset_for_fork()

        assert telemetry._session_uuid is None
        assert telemetry._heartbeat_started is False
        assert telemetry._deployment_type_override == telemetry.DeploymentTypeEnum.API

    def test_reset_for_fork_does_not_acquire_lock(self):
        class ExplodingLock:
            def __enter__(self):
                raise AssertionError("fork hook must not acquire locks")

            def __exit__(self, exc_type, exc, tb):
                return False

        telemetry._session_uuid = "old-session-uuid"
        telemetry._heartbeat_started = True
        telemetry._deployment_type_override = telemetry.DeploymentTypeEnum.API

        old_lock = ExplodingLock()
        telemetry._lock = old_lock
        telemetry._reset_for_fork()

        assert telemetry._session_uuid is None
        assert telemetry._heartbeat_started is False
        assert telemetry._deployment_type_override == telemetry.DeploymentTypeEnum.API
        assert telemetry._lock is not old_lock

    def test_reset_for_fork_replaces_poisoned_lock_and_next_report_completes(self):
        old_lock = telemetry.threading.Lock()
        old_lock.acquire()
        telemetry._lock = old_lock
        telemetry._session_uuid = "old-session-uuid"
        telemetry._heartbeat_started = True
        telemetry._deployment_type_override = telemetry.DeploymentTypeEnum.API

        try:
            telemetry._reset_for_fork()

            assert telemetry._lock is not old_lock
            assert telemetry._session_uuid is None
            assert telemetry._heartbeat_started is False
            assert telemetry._deployment_type_override == telemetry.DeploymentTypeEnum.API

            with (
                patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
                patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
            ):
                mock_thread.return_value = MagicMock()
                report_usage(None, deployment_type="library")

            send_calls = _send_event_calls(mock_thread)
            assert len(send_calls) == 1
            usage_data = send_calls[0].kwargs["args"][0]
            assert usage_data.deployment_type == "api"
            assert telemetry._heartbeat_started is True
        finally:
            old_lock.release()

    def test_reset_for_fork_lets_next_call_start_fresh(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, deployment_type="library")
            parent_session = telemetry._session_uuid
            assert parent_session is not None
            assert telemetry._heartbeat_started is True

            telemetry._reset_for_fork()

            report_usage(None, deployment_type="library")
            child_session = telemetry._session_uuid
            assert child_session is not None
            assert child_session != parent_session
            assert len(_heartbeat_calls(mock_thread)) == 2


class TestRotation:
    def test_rotate_creates_backup(self, audit_dir):
        config_dir, audit_file = audit_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        audit_file.write_text("original content")
        with patch.object(telemetry, "_AUDIT_FILE", audit_file):
            _rotate_audit_file()
        backup = audit_file.with_suffix(".json.1")
        assert backup.exists()
        assert backup.read_text() == "original content"
        assert not audit_file.exists()

    def test_rotate_overwrites_old_backup(self, audit_dir):
        config_dir, audit_file = audit_dir
        config_dir.mkdir(parents=True, exist_ok=True)
        backup = audit_file.with_suffix(".json.1")
        backup.write_text("old backup")
        audit_file.write_text("current")
        with patch.object(telemetry, "_AUDIT_FILE", audit_file):
            _rotate_audit_file()
        assert backup.read_text() == "current"


class TestBuiltinFeatures:
    def test_detects_configured_features(self):
        from nemoguardrails.rails.llm.config import JailbreakDetectionConfig, Rails, RailsConfigData

        config_data = RailsConfigData(
            jailbreak_detection=JailbreakDetectionConfig(nim_base_url="https://ai.api.nvidia.com"),
        )
        config = MagicMock()
        config.rails = Rails(config=config_data)
        result = _detect_builtin_features(config)
        assert "jailbreak_detection" in result

    def test_no_features_when_all_default(self):
        from nemoguardrails.rails.llm.config import Rails, RailsConfigData

        config = MagicMock()
        config.rails = Rails(config=RailsConfigData())
        result = _detect_builtin_features(config)
        assert result == []

    def test_multiple_features_detected(self):
        from nemoguardrails.rails.llm.config import (
            JailbreakDetectionConfig,
            Rails,
            RailsConfigData,
            SensitiveDataDetection,
            SensitiveDataDetectionOptions,
        )

        config_data = RailsConfigData(
            jailbreak_detection=JailbreakDetectionConfig(nim_base_url="https://example.com"),
            sensitive_data_detection=SensitiveDataDetection(
                input=SensitiveDataDetectionOptions(entities=["PERSON", "EMAIL"]),
            ),
        )
        config = MagicMock()
        config.rails = Rails(config=config_data)
        result = _detect_builtin_features(config)
        assert "jailbreak_detection" in result
        assert "sensitive_data_detection" in result

    def test_config_feature_ids_are_normalized(self):
        from nemoguardrails.rails.llm.config import (
            FactCheckingRailConfig,
            PatronusEvaluateApiParams,
            PatronusEvaluateConfig,
            PatronusRailConfig,
            Rails,
            RailsConfigData,
            RegexDetection,
            RegexDetectionOptions,
        )

        config_data = RailsConfigData(
            fact_checking=FactCheckingRailConfig(parameters={"endpoint": "http://example.com"}),
            patronus=PatronusRailConfig(
                output=PatronusEvaluateConfig(
                    evaluate_config=PatronusEvaluateApiParams(params={"criteria": "patronus:hallucination"})
                )
            ),
            regex_detection=RegexDetection(input=RegexDetectionOptions(patterns=["secret"])),
        )
        config = MagicMock()
        config.rails = Rails(config=config_data)

        result = _detect_builtin_features(config)

        assert "factchecking" in result
        assert "patronusai" in result
        assert "regex" in result
        assert "fact_checking" not in result
        assert "patronus" not in result
        assert "regex_detection" not in result

    def test_detects_features_from_exact_flow_names(self):
        from nemoguardrails.rails.llm.config import Rails

        config = MagicMock()
        config.rails = Rails()
        config.rails.input.flows = [
            "content safety check input $model=content_safety",
            "topic safety check input $model=topic_control",
            "jailbreak detection model",
        ]
        config.rails.output.flows = ["content safety check output $model=content_safety"]
        result = _detect_builtin_features(config)
        assert "content_safety" in result
        assert "topic_safety" in result
        assert "jailbreak_detection" in result

    def test_ignores_unknown_flow_names(self):
        from nemoguardrails.rails.llm.config import Rails

        config = MagicMock()
        config.rails = Rails()
        config.rails.input.flows = [
            "my custom content safety wrapper",
            "check user input for bad words",
        ]
        config.rails.output.flows = []
        result = _detect_builtin_features(config)
        assert result == []

    def test_combined_config_and_flow_detection(self):
        from nemoguardrails.rails.llm.config import JailbreakDetectionConfig, Rails, RailsConfigData

        config_data = RailsConfigData(
            jailbreak_detection=JailbreakDetectionConfig(nim_base_url="https://example.com"),
        )
        config = MagicMock()
        config.rails = Rails(config=config_data)
        config.rails.input.flows = ["self check input"]
        result = _detect_builtin_features(config)
        assert "jailbreak_detection" in result
        assert "self_check" in result

    def test_no_rails_config(self):
        config = MagicMock()
        config.rails = None
        assert _detect_builtin_features(config) == []

    def test_custom_flows_counted(self):
        config = MagicMock()
        config.colang_version = "2.x"
        config.models = []
        config.rails = None
        config.tracing = None
        config.docs = None
        config.flows = [
            {"id": "greeting", "is_system_flow": False},
            {"id": "farewell"},
            {"id": "self check input", "is_system_flow": True},
            {"id": "generate user intent", "is_system_flow": True},
        ]
        data = _collect_usage_data(config, "library")
        assert data.num_custom_flows == 2

    def test_v2_library_flows_not_counted_as_custom(self, tmp_path):
        from nemoguardrails.colang.v2_x.lang.colang_ast import Flow

        config = MagicMock()
        config.colang_version = "2.x"
        config.models = []
        config.rails = None
        config.tracing = None
        config.docs = None
        config.flows = [
            Flow(
                name="_user_said",
                file_info={"name": str(telemetry._COLANG_V2_LIBRARY_DIR / "core.co")},
            ),
            Flow(
                name="main",
                file_info={"name": str(tmp_path / "main.co")},
            ),
        ]

        data = _collect_usage_data(config, "library")

        assert data.num_custom_flows == 1

    def test_real_v2_tutorial_counts_only_user_flows(self):
        from nemoguardrails.rails.llm.config import RailsConfig

        config_path = Path(__file__).parents[2] / "examples" / "v2_x" / "tutorial" / "hello_world_1"
        config = RailsConfig.from_path(str(config_path))

        data = _collect_usage_data(config, "library")

        assert len(config.flows) == 46
        assert data.num_custom_flows == 1

    def test_v2_smoke_fixture_counts_only_user_flows(self):
        from nemoguardrails.rails.llm.config import RailsConfig

        config_path = Path(__file__).parent / "smoke_fixtures" / "v2_custom_flow"
        config = RailsConfig.from_path(str(config_path))

        data = _collect_usage_data(config, "library")

        assert len(config.flows) == 46
        assert data.num_custom_flows == 1

    def test_feature_alias_smoke_fixture_collects_documented_ids(self):
        from nemoguardrails.rails.llm.config import RailsConfig

        config_path = Path(__file__).parent / "smoke_fixtures" / "feature_aliases"
        config = RailsConfig.from_path(str(config_path))

        data = _collect_usage_data(config, "library")

        assert data.builtin_features == ["factchecking", "patronusai", "regex"]

    def test_included_in_usage_data(self):
        from nemoguardrails.rails.llm.config import JailbreakDetectionConfig, Rails, RailsConfigData

        config_data = RailsConfigData(
            jailbreak_detection=JailbreakDetectionConfig(nim_base_url="https://example.com"),
        )
        config = MagicMock()
        config.rails = Rails(config=config_data)
        config.rails.input.flows = []
        config.rails.output.flows = []
        config.rails.retrieval.flows = []
        config.rails.tool_output.flows = []
        config.rails.tool_input.flows = []
        config.rails.dialog.single_call.enabled = False
        config.rails.output.streaming.enabled = False
        config.tracing.enabled = False
        config.docs = None
        config.colang_version = "2.x"
        config.models = []

        data = _collect_usage_data(config, "library")
        assert "jailbreak_detection" in data.builtin_features


class TestUpstreamSchemaConformance:
    """Validate emitted events against the vendored upstream schema.

    Catches drift between our local Pydantic model and the canonical
    wire contract published by the shared nemo-telemetry repo at
    schemas/anonymous_events.json. A snapshot of that file lives at
    schemas/anonymous_events.snapshot.json. Refresh it manually when
    the upstream file changes (e.g. the upstream MR is merged).
    """

    @pytest.fixture
    def validator(self, pytestconfig):
        import jsonschema

        snapshot_path = pytestconfig.rootpath / "schemas" / "anonymous_events.snapshot.json"
        snapshot = json.loads(snapshot_path.read_text())
        return jsonschema.Draft7Validator(snapshot)

    def test_default_payload_validates(self, validator):
        event = GuardrailsUsageEvent(session_id="s")
        validator.validate(event.model_dump(by_alias=True))

    def test_heartbeat_payload_validates(self, validator):
        from nemoguardrails.telemetry import EventTypeEnum

        event = GuardrailsUsageEvent(session_id="s", event=EventTypeEnum.HEARTBEAT)
        validator.validate(event.model_dump(by_alias=True))

    def test_fully_populated_payload_validates(self, validator, mock_config):
        from nemoguardrails.telemetry import RailsEngineEnum

        event = _collect_usage_data(mock_config, "library")
        event.rails_engine = RailsEngineEnum.LLMRAILS
        validator.validate(event.model_dump(by_alias=True))


class TestTelemetrySmokeDriver:
    @staticmethod
    def _scenarios_by_name():
        scenarios = telemetry_smoke._build_scenarios(
            library_config="cfg1",
            rich_config="rich",
            feature_alias_config="feature_aliases",
            v2_config="v2_custom_flow",
            iorails_config="iorails",
            server_config_root="root",
        )
        return {scenario["name"]: scenario for scenario in scenarios}

    @staticmethod
    def _startup_event(**overrides):
        event = {
            "nemoSource": "guardrails",
            "event": "startup",
            "sessionId": "2b8e9879-80be-42bb-ad3f-81db8ec28e15",
            "nemoguardrailsVersion": "0.21.0",
            "pythonVersion": "3.13.7",
            "platform": "test-platform",
            "osName": "Darwin",
            "timestamp": 1700000000.0,
            "deploymentType": "library",
            "railsEngine": "LLMRails",
            "llmProviders": ["openai"],
            "colangVersion": "1.0",
            "numRailsConfigured": 0,
            "railTypesInUse": [],
            "builtinFeatures": [],
            "tracingEnabled": False,
            "hasKnowledgeBase": False,
            "streamingConfigured": False,
            "numCustomFlows": 0,
        }
        event.update(overrides)
        return event

    def test_smoke_scenarios_cover_feature_aliases_and_v2_custom_flows(self):
        scenarios = self._scenarios_by_name()

        assert "library_feature_aliases" in scenarios
        assert "library_v2_custom_flows" in scenarios
        assert "library_abc_v2" not in scenarios

    def test_positive_scenarios_wait_for_daemon_sends(self):
        scenarios = self._scenarios_by_name()

        assert telemetry_smoke.DEFAULT_NETWORK_SETTLE_S == 20.0
        assert "settle_after_audit_s" not in scenarios["library_llmrails"]
        assert "settle_after_audit_s" not in scenarios["server_single_config"]

    def test_kibana_filter_uses_exact_session_ids(self):
        assert telemetry_smoke._format_kibana_filter(["id1", "id2"]) == 'client.sessionId : ("id1" or "id2")'

    def test_common_assertions_require_non_empty_session_id_only(self):
        scenario = self._scenarios_by_name()["library_feature_aliases"]
        event = self._startup_event(sessionId="", builtinFeatures=["factchecking", "patronusai", "regex"])

        error = telemetry_smoke._validate_startup_events(
            scenario["name"],
            [event],
            assertion_sets=scenario["startup_assertions"],
        )

        assert error is not None
        assert "sessionId" in error

    def test_feature_alias_smoke_assertions_require_documented_ids(self):
        scenario = self._scenarios_by_name()["library_feature_aliases"]
        event = self._startup_event(builtinFeatures=["factchecking", "patronusai", "regex"])

        assert (
            telemetry_smoke._validate_startup_events(
                scenario["name"],
                [event],
                assertion_sets=scenario["startup_assertions"],
            )
            is None
        )

        bad_event = dict(event)
        bad_event["builtinFeatures"] = ["fact_checking", "patronus", "regex_detection"]
        error = telemetry_smoke._validate_startup_events(
            scenario["name"],
            [bad_event],
            assertion_sets=scenario["startup_assertions"],
        )

        assert error is not None
        assert "builtinFeatures" in error

    def test_v2_smoke_assertions_reject_bundled_library_flow_count(self):
        scenario = self._scenarios_by_name()["library_v2_custom_flows"]
        event = self._startup_event(colangVersion="2.x", numCustomFlows=1)

        assert (
            telemetry_smoke._validate_startup_events(
                scenario["name"],
                [event],
                assertion_sets=scenario["startup_assertions"],
            )
            is None
        )

        bad_event = dict(event)
        bad_event["numCustomFlows"] = 46
        error = telemetry_smoke._validate_startup_events(
            scenario["name"],
            [bad_event],
            assertion_sets=scenario["startup_assertions"],
        )

        assert error is not None
        assert "numCustomFlows" in error

    @pytest.mark.parametrize(
        ("runner_name", "scenario"),
        [
            (
                "_run_subprocess",
                {
                    "script": "pass",
                    "expected_count": 1,
                },
            ),
            (
                "_run_server",
                {
                    "server_config_root": "root",
                    "config_ids_to_hit": ["cfg1"],
                    "expected_count": 1,
                },
            ),
            (
                "_run_server_multi_worker",
                {
                    "server_config_root": "root",
                    "config_ids_to_hit": ["cfg1"],
                    "expected_count": 3,
                    "worker_count": 3,
                },
            ),
            (
                "_run_cli",
                {
                    "config_path": "cfg1",
                    "expected_count": 1,
                },
            ),
        ],
    )
    def test_process_runner_spawn_failures_return_structured_results(self, tmp_path, runner_name, scenario):
        audit_file = tmp_path / "run" / "scenario" / "usage_stats.json"
        runner = getattr(telemetry_smoke, runner_name)

        with (
            patch.object(telemetry_smoke, "_free_port", return_value=12345),
            patch.object(telemetry_smoke.subprocess, "Popen", side_effect=FileNotFoundError("missing executable")),
        ):
            result = runner(scenario, {}, audit_file)

        assert result["returncode"] == -1
        assert result["duration_s"] >= 0
        assert result["stderr_tail"] == ["FileNotFoundError('missing executable')"]
        if runner_name in {"_run_server", "_run_server_multi_worker"}:
            assert result["server_post_results"] == []
        if runner_name == "_run_server_multi_worker":
            assert result["worker_count"] == 3

    def test_terminate_normalizes_windows_termination_code_only_after_terminating(self):
        class NaturallyFailedProcess:
            returncode = 1
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True

        class TerminatedProcess:
            returncode = None
            terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True

            def wait(self, timeout):
                self.returncode = 1
                return self.returncode

        naturally_failed = NaturallyFailedProcess()
        assert telemetry_smoke._terminate(naturally_failed) == 1
        assert naturally_failed.terminated is False

        terminated = TerminatedProcess()
        assert telemetry_smoke._terminate(terminated) == 0
        assert terminated.terminated is True


class TestKibanaVerifyExport:
    def test_empty_manifest_raises_value_error(self):
        with pytest.raises(ValueError, match="manifest has no results to verify"):
            kibana_verify_export._verify({"results": []}, [])

    def test_main_returns_2_for_empty_manifest(self, tmp_path, capsys):
        manifest_path = tmp_path / "manifest.json"
        export_path = tmp_path / "kibana.json"
        manifest_path.write_text(json.dumps({"results": []}))
        export_path.write_text(json.dumps([]))

        with patch.object(
            sys,
            "argv",
            [
                "kibana_verify_export.py",
                "--manifest",
                str(manifest_path),
                "--export",
                str(export_path),
            ],
        ):
            assert kibana_verify_export.main() == 2

        captured = capsys.readouterr()
        assert "manifest has no results to verify" in captured.err

    def test_verifier_matches_exact_client_session_ids(self):
        manifest = {
            "results": [
                {
                    "name": "library_llmrails",
                    "startup_session_ids": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                    "expected_event_count": 1,
                    "verdict": "PASS",
                }
            ]
        }
        docs = [
            {
                "fields": {
                    "eventName": ["guardrails_usage_event"],
                    "client.sessionId": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                }
            }
        ]

        passed, failed = kibana_verify_export._verify(manifest, docs)

        assert passed == 1
        assert failed == 0

    def test_verifier_rejects_prefix_only_session_match(self):
        manifest = {
            "results": [
                {
                    "name": "library_llmrails",
                    "startup_session_ids": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                    "expected_event_count": 1,
                    "verdict": "PASS",
                }
            ]
        }
        docs = [
            {
                "fields": {
                    "eventName": ["guardrails_usage_event"],
                    "client.sessionId": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15-extra"],
                }
            }
        ]

        passed, failed = kibana_verify_export._verify(manifest, docs)

        assert passed == 0
        assert failed == 1

    def test_verifier_ignores_parameters_session_id(self):
        manifest = {
            "results": [
                {
                    "name": "library_llmrails",
                    "startup_session_ids": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                    "expected_event_count": 1,
                    "verdict": "PASS",
                }
            ]
        }
        docs = [
            {
                "fields": {
                    "eventName": ["guardrails_usage_event"],
                    "parameters.sessionId": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                }
            }
        ]

        passed, failed = kibana_verify_export._verify(manifest, docs)

        assert passed == 0
        assert failed == 1

    def test_failed_smoke_scenarios_fail_verification(self):
        manifest = {
            "results": [
                {
                    "name": "library_llmrails",
                    "startup_session_ids": ["2b8e9879-80be-42bb-ad3f-81db8ec28e15"],
                    "expected_event_count": 1,
                    "verdict": "FAIL",
                }
            ]
        }

        passed, failed = kibana_verify_export._verify(manifest, [])

        assert passed == 0
        assert failed == 1
