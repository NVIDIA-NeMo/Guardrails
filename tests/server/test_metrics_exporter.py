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

"""Server-side Prometheus export of the non-streaming admission-queue metrics.

The library records metrics through the OpenTelemetry API only, so these tests
cover the one place the server owns an SDK: ``nemoguardrails.server.metrics``.
They check the env-var contract, the view that narrows export to the
``guardrails.nonstream.*`` instruments, the scrape endpoint lifecycle, and the
FastAPI lifespan hook.  No test talks to a model provider.
"""

import asyncio
import copy
import sys
import urllib.request
from unittest.mock import patch

import opentelemetry.metrics._internal as otel_metrics_internal
import pytest
from fastapi.testclient import TestClient
from opentelemetry import metrics as otel_metrics
from opentelemetry.metrics import Observation
from prometheus_client import CollectorRegistry, generate_latest

from nemoguardrails import RailsConfig
from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.server import api
from nemoguardrails.server import metrics as server_metrics
from nemoguardrails.server.metrics import (
    ENV_EXPORTER,
    ENV_HOST,
    ENV_PORT,
    ENV_SERVICE_NAME,
    MetricsExporter,
    MetricsExporterConfigError,
    MetricsExporterSettings,
    build_meter_provider,
    get_active_metrics_exporter,
    shutdown_metrics_exporter,
    start_metrics_exporter,
)
from nemoguardrails.tracing import constants as tracing_constants
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

LOOPBACK_EPHEMERAL = {ENV_EXPORTER: "prometheus", ENV_HOST: "127.0.0.1", ENV_PORT: "0"}


@pytest.fixture(autouse=True)
def reset_global_metrics_state():
    """Give every test a process that has never installed a MeterProvider.

    The OpenTelemetry API only lets ``set_meter_provider`` succeed once per
    process, and the exporter refuses to start when a provider is already in
    place.  Resetting the API's set-once guard and the library's cached meter
    before and after each test keeps the tests order-independent.
    """

    def _reset():
        """Forget every process-wide singleton the exporter or the OTel API keeps."""
        shutdown_metrics_exporter()
        server_metrics._installed_provider = None
        otel_metrics_internal._METER_PROVIDER = None
        otel_metrics_internal._METER_PROVIDER_SET_ONCE = otel_metrics_internal.Once()
        # Proxy meters created while no provider was installed replay their
        # deferred instruments onto the next real provider.  Earlier tests
        # that built IORails without a global provider would otherwise land
        # their stale gauges on ours and shadow the ones under test.
        otel_metrics_internal._PROXY_METER_PROVIDER._meters.clear()
        otel_metrics_internal._PROXY_METER_PROVIDER._real_meter_provider = None
        telemetry._meter = None
        telemetry._request_instruments = None
        tracing_constants._llm_instruments = None

    _reset()
    yield
    _reset()


def _scrape(url: str) -> str:
    """GET a scrape URL and return the exposition text."""
    with urllib.request.urlopen(url, timeout=5) as response:
        assert response.status == 200
        return response.read().decode()


def _record_every_instrument_family(provider):
    """Emit one value on each family the engine records: the three admission-queue
    instruments plus a request counter and a GenAI histogram that must be dropped."""
    meter = provider.get_meter("nemo-guardrails", version="0.0.0-test")
    meter.create_observable_gauge("guardrails.nonstream.queued", callbacks=[lambda options: [Observation(3)]], unit="1")
    meter.create_observable_gauge("guardrails.nonstream.active", callbacks=[lambda options: [Observation(2)]], unit="1")
    meter.create_counter("guardrails.nonstream.rejections", unit="1").add(5)
    meter.create_counter("guardrails.requests", unit="1").add(9)
    meter.create_histogram("gen_ai.client.token.usage", unit="{token}").record(42)


class TestSettingsFromEnv:
    def test_defaults_to_no_exporter(self):
        """No env vars means no exporter and the documented defaults."""
        settings = MetricsExporterSettings.from_env({})
        assert settings.exporter is MetricsExporter.NONE
        assert settings.enabled is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 9464
        assert settings.service_name == "nemoguardrails-server"

    def test_reads_all_variables(self):
        """Every documented env var is honoured, case-insensitively for the exporter name."""
        settings = MetricsExporterSettings.from_env(
            {ENV_EXPORTER: "Prometheus", ENV_HOST: "127.0.0.1", ENV_PORT: "9999", ENV_SERVICE_NAME: "gr-prod"}
        )
        assert settings.exporter is MetricsExporter.PROMETHEUS
        assert settings.enabled is True
        assert settings.host == "127.0.0.1"
        assert settings.port == 9999
        assert settings.service_name == "gr-prod"

    def test_otel_service_name_is_the_fallback(self):
        """OTEL_SERVICE_NAME is used unless the NEMO-specific variable overrides it."""
        assert MetricsExporterSettings.from_env({"OTEL_SERVICE_NAME": "from-otel"}).service_name == "from-otel"
        both = {"OTEL_SERVICE_NAME": "from-otel", ENV_SERVICE_NAME: "from-nemo"}
        assert MetricsExporterSettings.from_env(both).service_name == "from-nemo"

    def test_rejects_unknown_exporter(self):
        """Unsupported exporter names fail with the list of supported values."""
        with pytest.raises(MetricsExporterConfigError, match="Unsupported .*otlp.*Supported values: none, prometheus"):
            MetricsExporterSettings.from_env({ENV_EXPORTER: "otlp"})

    @pytest.mark.parametrize("raw_port", ["abc", "-1", "70000"])
    def test_rejects_bad_port(self, raw_port):
        """Non-integer and out-of-range ports fail with an actionable message."""
        with pytest.raises(MetricsExporterConfigError, match=ENV_PORT):
            MetricsExporterSettings.from_env({ENV_EXPORTER: "prometheus", ENV_PORT: raw_port})


class TestExportView:
    def test_only_nonstream_admission_metrics_are_exported(self):
        """The export view keeps the three admission-queue instruments and drops everything else."""
        registry = CollectorRegistry()
        provider = build_meter_provider(MetricsExporterSettings(service_name="scoped"), registry)
        try:
            _record_every_instrument_family(provider)
            output = generate_latest(registry).decode()
        finally:
            provider.shutdown()

        assert "guardrails_nonstream_queued{" in output
        assert "guardrails_nonstream_active{" in output
        assert "guardrails_nonstream_rejections_total{" in output
        assert "} 3.0" in output and "} 2.0" in output and "} 5.0" in output
        assert 'service_name="scoped"' in output

        assert "guardrails_requests" not in output
        assert "gen_ai_client_token_usage" not in output

    def test_prometheus_type_lines(self):
        """Gauges export as Prometheus gauges and the counter gains the _total suffix."""
        registry = CollectorRegistry()
        provider = build_meter_provider(MetricsExporterSettings(), registry)
        try:
            _record_every_instrument_family(provider)
            output = generate_latest(registry).decode()
        finally:
            provider.shutdown()

        assert "# TYPE guardrails_nonstream_queued gauge" in output
        assert "# TYPE guardrails_nonstream_active gauge" in output
        assert "# TYPE guardrails_nonstream_rejections_total counter" in output


class TestStartAndShutdown:
    def test_disabled_by_default(self):
        """Without configuration nothing starts and the global provider stays untouched."""
        with patch.dict("os.environ", {}, clear=True):
            assert start_metrics_exporter() is None
        assert get_active_metrics_exporter() is None
        assert type(otel_metrics.get_meter_provider()).__module__.startswith("opentelemetry.metrics")

    def test_serves_scrape_endpoint_and_installs_global_provider(self):
        """A started exporter serves /metrics and the library meter records into it."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            exporter = start_metrics_exporter()
        assert exporter is not None
        assert exporter.port != 0
        assert exporter.url == f"http://127.0.0.1:{exporter.port}/metrics"

        # The library's meter now resolves to the server-owned provider.
        meter = telemetry.get_meter()
        assert meter is not None
        meter.create_counter("guardrails.nonstream.rejections", unit="1").add(1)

        output = _scrape(exporter.url)
        assert 'service_name="nemoguardrails-server"' in output
        assert "guardrails_nonstream_rejections_total{" in output

    def test_start_is_idempotent(self):
        """Calling start while running returns the same exporter."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            first = start_metrics_exporter()
            assert start_metrics_exporter() is first
            assert get_active_metrics_exporter() is first

    def test_shutdown_closes_listener_and_is_repeatable(self):
        """Shutdown closes the listener and a second shutdown is a no-op."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            exporter = start_metrics_exporter()
        url = exporter.url

        shutdown_metrics_exporter()
        shutdown_metrics_exporter()

        assert get_active_metrics_exporter() is None
        with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
            urllib.request.urlopen(url, timeout=1)

    def test_refuses_when_a_meter_provider_is_already_installed(self):
        """A foreign global MeterProvider makes start fail instead of silently exporting nothing."""
        from opentelemetry.sdk.metrics import MeterProvider

        otel_metrics.set_meter_provider(MeterProvider())
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with pytest.raises(MetricsExporterConfigError, match="already configured"):
                start_metrics_exporter()
        assert get_active_metrics_exporter() is None

    def test_port_in_use_is_reported_with_the_override_hint(self):
        """A bind failure fails with a message pointing at --metrics-port.

        The failure is simulated rather than provoked with a second listener:
        the WSGI server enables address reuse, and on Windows that lets a
        second socket bind a port that is already listening, so a real
        double bind is not a portable way to trigger EADDRINUSE.
        """
        import errno

        import prometheus_client

        def _refuse_bind(*args, **kwargs):
            raise OSError(errno.EADDRINUSE, "Address already in use")

        with patch.object(prometheus_client, "start_http_server", _refuse_bind):
            with patch.dict("os.environ", {**LOOPBACK_EPHEMERAL, ENV_PORT: "9464"}):
                with pytest.raises(MetricsExporterConfigError, match=r"127\.0\.0\.1:9464.*--metrics-port"):
                    start_metrics_exporter()
        assert get_active_metrics_exporter() is None
        # The provider stays installed, so a retry on a free port succeeds.
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            assert start_metrics_exporter() is not None

    def test_restart_after_shutdown_reuses_the_process_wide_provider(self):
        """A second lifespan in the same process must be able to export again."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            first = start_metrics_exporter()
            telemetry.get_meter().create_counter("guardrails.nonstream.rejections", unit="1").add(3)
            shutdown_metrics_exporter()
            second = start_metrics_exporter()

        assert second is not first
        assert second.provider is first.provider
        assert otel_metrics.get_meter_provider() is first.provider
        # State recorded before the restart is still served, and new
        # recordings keep landing on the same instruments.
        telemetry.get_meter().create_counter("guardrails.nonstream.rejections", unit="1").add(4)
        output = _scrape(second.url)
        rejections = next(
            line for line in output.splitlines() if line.startswith("guardrails_nonstream_rejections_total{")
        )
        assert rejections.endswith(" 7.0")

    def test_restart_with_a_different_service_name_keeps_the_original(self, caplog):
        """The provider cannot be replaced, so a changed service.name on restart is logged and ignored."""
        with patch.dict("os.environ", {**LOOPBACK_EPHEMERAL, ENV_SERVICE_NAME: "first"}):
            start_metrics_exporter()
            shutdown_metrics_exporter()
        with caplog.at_level("WARNING", logger="nemoguardrails.server.metrics"):
            with patch.dict("os.environ", {**LOOPBACK_EPHEMERAL, ENV_SERVICE_NAME: "second"}):
                exporter = start_metrics_exporter()
        telemetry.get_meter().create_counter("guardrails.nonstream.rejections", unit="1").add(1)
        assert 'service_name="first"' in _scrape(exporter.url)
        assert "keeping the original" in caplog.text

    def test_missing_optional_dependency_gives_install_hint(self):
        """A missing optional dependency fails with the server extra install hint."""
        with patch.dict(sys.modules, {"opentelemetry.exporter.prometheus": None}):
            with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
                with pytest.raises(MetricsExporterConfigError, match=r"nemoguardrails\[server\]"):
                    start_metrics_exporter()
        assert get_active_metrics_exporter() is None


class TestServerLifespan:
    def test_lifespan_starts_and_stops_the_exporter(self, tmp_path):
        """The FastAPI lifespan starts the exporter on startup and stops it on shutdown."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with TestClient(api.app):
                exporter = get_active_metrics_exporter()
                assert exporter is not None
                # The scrape output is empty until an instrument exists, so
                # record through the library's meter the way the engine does.
                telemetry.get_meter().create_counter("guardrails.nonstream.rejections", unit="1").add(1)
                assert "guardrails_nonstream_rejections_total{" in _scrape(exporter.url)
            assert get_active_metrics_exporter() is None

    def test_lifespan_releases_exporter_when_startup_fails(self, tmp_path):
        """A startup failure after the exporter started still releases the listener."""
        (tmp_path / "challenges.json").write_text("{not json")
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with pytest.raises(Exception):
                with TestClient(api.app):
                    pass  # pragma: no cover - startup raises before the body runs
        assert get_active_metrics_exporter() is None

    def test_lifespan_can_run_twice_in_one_process(self, tmp_path):
        """Back-to-back lifespans in one process both export metrics."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with TestClient(api.app):
                first_port = get_active_metrics_exporter().port
            with TestClient(api.app):
                second = get_active_metrics_exporter()
                assert second is not None
                telemetry.get_meter().create_counter("guardrails.nonstream.rejections", unit="1").add(1)
                assert "guardrails_nonstream_rejections_total{" in _scrape(second.url)
            assert get_active_metrics_exporter() is None
        assert first_port != 0

    def test_lifespan_without_exporter_leaves_metrics_untouched(self, tmp_path):
        """Without configuration the lifespan does not start an exporter."""
        with patch.dict("os.environ", {}, clear=True), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with TestClient(api.app):
                assert get_active_metrics_exporter() is None


class TestIORailsEndToEnd:
    """The engine's real gauges reach the scrape endpoint through the server-owned provider."""

    @pytest.mark.asyncio
    async def test_admission_queue_gauges_track_a_live_request(self):
        """A real IORails request shows up as active=1 on the scrape endpoint while it runs."""
        with patch.dict("os.environ", {**LOOPBACK_EPHEMERAL, "NVIDIA_API_KEY": "test-key"}):
            exporter = start_metrics_exporter()
            config = copy.deepcopy(CONTENT_SAFETY_CONFIG)
            config["metrics"] = {"enabled": True}
            iorails = IORails(RailsConfig.from_content(config=config))

        gate = asyncio.Event()

        async def _blocked_generate(messages, req_id, request_span=None, **kwargs):
            """Stand-in pipeline that parks the request until the test releases the gate."""
            await gate.wait()
            return {"role": "assistant", "content": "done"}

        async with iorails:
            idle = _scrape(exporter.url)
            assert "guardrails_nonstream_queued{" in idle
            assert "guardrails_nonstream_active{" in idle

            with patch.object(iorails, "_do_generate", _blocked_generate):
                task = asyncio.create_task(iorails.generate_async(messages=[{"role": "user", "content": "hi"}]))
                # Let the worker pick the item up before observing.
                for _ in range(50):
                    await asyncio.sleep(0.01)
                    if iorails._generate_async_queue.num_busy_workers() == 1:
                        break
                busy = await asyncio.to_thread(_scrape, exporter.url)
                gate.set()
                await task

        active_line = next(line for line in busy.splitlines() if line.startswith("guardrails_nonstream_active{"))
        assert active_line.endswith(" 1.0")
        assert "guardrails_requests_active" not in busy
        assert "gen_ai_client" not in busy
