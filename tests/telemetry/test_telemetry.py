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

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemoguardrails import telemetry
from nemoguardrails.telemetry import (
    GuardrailsUsageEvent,
    _collect_usage_data,
    _detect_builtin_features,
    _is_usage_stats_enabled,
    _rotate_audit_file,
    _send_report,
    _write_audit_file,
    report_usage,
)


@pytest.fixture(autouse=True)
def reset_reported():
    telemetry._reported = False
    yield
    telemetry._reported = False


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
            patch.dict(os.environ, {}, clear=True),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", Path("/nonexistent/path")),
        ):
            assert _is_usage_stats_enabled() is True

    def test_disabled_by_nemo_env_var(self):
        with patch.dict(os.environ, {"NEMO_GUARDRAILS_NO_USAGE_STATS": "1"}):
            assert _is_usage_stats_enabled() is False

    def test_disabled_by_do_not_track(self):
        with patch.dict(os.environ, {"DO_NOT_TRACK": "1"}):
            assert _is_usage_stats_enabled() is False

    def test_disabled_by_file(self, tmp_path):
        do_not_track = tmp_path / "do_not_track"
        do_not_track.touch()
        with patch.dict(os.environ, {}, clear=True), patch.object(telemetry, "_DO_NOT_TRACK_FILE", do_not_track):
            assert _is_usage_stats_enabled() is False

    def test_not_disabled_when_env_var_is_zero(self):
        with (
            patch.dict(os.environ, {"NEMO_GUARDRAILS_NO_USAGE_STATS": "0"}),
            patch.object(telemetry, "_DO_NOT_TRACK_FILE", Path("/nonexistent/path")),
        ):
            assert _is_usage_stats_enabled() is True


class TestDataCollection:
    def test_collect_without_config(self):
        data = _collect_usage_data(None, "server")
        assert data.context == "server"
        assert data.event == "startup"
        assert data.python_version != ""
        assert data.platform != ""
        assert data.os_name != ""
        assert data.colang_version == "unknown"
        assert data.llm_providers == []
        assert data.num_rails_configured == 0

    def test_collect_with_config(self, mock_config):
        data = _collect_usage_data(mock_config, "embedded")
        assert data.context == "embedded"
        assert data.colang_version == "2.x"
        assert data.llm_providers == ["nvidia_ai_endpoints", "openai"]
        assert data.num_rails_configured == 2
        assert "input" in data.rail_types_in_use
        assert "output" in data.rail_types_in_use
        assert data.tracing_enabled is True
        assert data.has_knowledge_base is True
        assert data.streaming_configured is True

    def test_engine_names_not_model_names(self, mock_config):
        data = _collect_usage_data(mock_config, "embedded")
        assert "openai" in data.llm_providers
        assert "nvidia_ai_endpoints" in data.llm_providers

    def test_rail_types_detected(self, mock_config):
        mock_config.rails.input.flows = []
        mock_config.rails.output.flows = ["some_flow"]
        data = _collect_usage_data(mock_config, "embedded")
        assert data.rail_types_in_use == ["output"]
        assert data.num_rails_configured == 1

    def test_uuid_is_unique(self):
        d1 = _collect_usage_data(None, "embedded")
        telemetry._reported = False
        d2 = _collect_usage_data(None, "embedded")
        assert d1.uuid != d2.uuid

    def test_all_fields_serializable(self):
        data = _collect_usage_data(None, "embedded")
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
        data = _collect_usage_data(None, "embedded")
        payload = data.model_dump(by_alias=True)
        assert "nemoguardrailsVersion" in payload
        assert "llmProviders" in payload
        assert "numRailsConfigured" in payload
        assert "railTypesInUse" in payload
        assert "hasKnowledgeBase" in payload
        assert "builtinFeatures" in payload
        assert "numCustomFlows" in payload
        assert "railsEngine" in payload


class TestRailsEngine:
    def test_default_is_undefined(self):
        data = _collect_usage_data(None, "server")
        assert data.rails_engine == "undefined"

    def test_set_via_report_usage(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, context="embedded", rails_engine="LLMRails")
            call_args = mock_thread.call_args
            usage_data = call_args[1]["args"][0]
            assert usage_data.rails_engine == "LLMRails"

    def test_iorails_engine(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, context="embedded", rails_engine="IORails")
            call_args = mock_thread.call_args
            usage_data = call_args[1]["args"][0]
            assert usage_data.rails_engine == "IORails"


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
        event = GuardrailsUsageEvent(uuid="test-uuid", timestamp=1700000000.0)
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
            assert envelope["eventSchemaVer"] == "1.0"
            assert len(envelope["events"]) == 1
            ev = envelope["events"][0]
            assert ev["name"] == "guardrails_usage_event"
            assert ev["parameters"]["nemoSource"] == "guardrails"
            assert ev["parameters"]["uuid"] == "test-uuid"

    def test_send_report_failure_silent(self):
        event = GuardrailsUsageEvent(uuid="test")
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
            report_usage(None, context="embedded")
            call_args = mock_thread.call_args
            assert call_args[1]["args"][1] == "https://custom.server/v1"


class TestIntegration:
    def test_report_spawns_daemon_thread(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            report_usage(None, context="embedded")
            mock_thread.assert_called_once()
            assert mock_thread.call_args[1]["daemon"] is True
            mock_instance.start.assert_called_once()

    def test_report_idempotent(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=True),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            mock_thread.return_value = MagicMock()
            report_usage(None, context="embedded")
            report_usage(None, context="server")
            assert mock_thread.call_count == 1

    def test_report_skipped_when_disabled(self):
        with (
            patch.object(telemetry, "_is_usage_stats_enabled", return_value=False),
            patch("nemoguardrails.telemetry.threading.Thread") as mock_thread,
        ):
            report_usage(None, context="embedded")
            mock_thread.assert_not_called()

    def test_heartbeat_payload(self):
        data = GuardrailsUsageEvent(
            uuid="test-uuid-123",
            python_version="3.13.7",
            platform="test-platform",
        )
        payloads = []

        def mock_write(d):
            payloads.append(d)

        def mock_send(event, url, ver, sid):
            pass

        def mock_sleep(seconds):
            if len(payloads) >= 2:
                raise SystemExit()

        with (
            patch.object(telemetry, "_write_audit_file", side_effect=mock_write),
            patch.object(telemetry, "_send_report", side_effect=mock_send),
            patch("nemoguardrails.telemetry.time.sleep", side_effect=mock_sleep),
        ):
            with pytest.raises(SystemExit):
                telemetry._usage_reporter_thread(data, "https://example.com")

        assert len(payloads) >= 2
        startup_payload = payloads[0]
        assert startup_payload["event"] == "startup"
        assert startup_payload["uuid"] == "test-uuid-123"
        assert startup_payload["pythonVersion"] == "3.13.7"

        heartbeat_payload = payloads[1]
        assert heartbeat_payload["event"] == "heartbeat"
        assert heartbeat_payload["uuid"] == "test-uuid-123"
        assert heartbeat_payload["pythonVersion"] == "unknown"


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
        data = _collect_usage_data(config, "embedded")
        assert data.num_custom_flows == 2

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

        data = _collect_usage_data(config, "embedded")
        assert "jailbreak_detection" in data.builtin_features
