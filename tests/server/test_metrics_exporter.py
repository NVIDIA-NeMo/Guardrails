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
FastAPI lifespans of the main server and the hooks server.  No test talks to a
model provider.
"""

import asyncio
import copy
import logging
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
from nemoguardrails import telemetry as usage_telemetry
from nemoguardrails.guardrails import telemetry
from nemoguardrails.guardrails.iorails import IORails
from nemoguardrails.server import api
from nemoguardrails.server import metrics as server_metrics
from nemoguardrails.server.hooks import app as hooks_app
from nemoguardrails.server.hooks.app import create_hooks_app, metrics_exporter_settings
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
    running_metrics_exporter,
    shutdown_metrics_exporter,
    start_metrics_exporter,
)
from nemoguardrails.tracing import constants as tracing_constants
from tests.guardrails.test_data import CONTENT_SAFETY_CONFIG

LOOPBACK_EPHEMERAL = {ENV_EXPORTER: "prometheus", ENV_HOST: "127.0.0.1", ENV_PORT: "0"}
# The same exporter as LOOPBACK_EPHEMERAL, for starting one as another app would, without the variables.
LOOPBACK_EPHEMERAL_SETTINGS = MetricsExporterSettings(MetricsExporter.PROMETHEUS, "127.0.0.1", 0)


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

    def test_defaults_fill_the_variables_left_unset(self):
        """The caller's defaults take the place of the module's for every variable that is unset."""
        defaults = MetricsExporterSettings(host="127.0.0.1", port=9999, service_name="from-defaults")

        assert MetricsExporterSettings.from_env({}, defaults=defaults) == defaults

    def test_variables_override_the_defaults(self):
        """A set variable wins over the caller's defaults, and OTEL_SERVICE_NAME over the default service name."""
        defaults = MetricsExporterSettings(host="127.0.0.1", port=9999, service_name="from-defaults")
        environ = {ENV_EXPORTER: "prometheus", ENV_HOST: "0.0.0.0", ENV_PORT: "9500", "OTEL_SERVICE_NAME": "from-otel"}

        settings = MetricsExporterSettings.from_env(environ, defaults=defaults)

        assert settings == MetricsExporterSettings(MetricsExporter.PROMETHEUS, "0.0.0.0", 9500, "from-otel")

    def test_rejects_unknown_exporter(self):
        """Unsupported exporter names fail with the list of supported values."""
        with pytest.raises(MetricsExporterConfigError, match="Unsupported .*otlp.*Supported values: none, prometheus"):
            MetricsExporterSettings.from_env({ENV_EXPORTER: "otlp"})

    @pytest.mark.parametrize("raw_port", ["abc", "-1", "70000"])
    def test_rejects_bad_port(self, raw_port):
        """Non-integer and out-of-range ports fail with an actionable message."""
        with pytest.raises(MetricsExporterConfigError, match=ENV_PORT):
            MetricsExporterSettings.from_env({ENV_EXPORTER: "prometheus", ENV_PORT: raw_port})


class TestDisplayHost:
    @pytest.mark.parametrize(
        ("host", "shown"),
        [
            ("0.0.0.0", "localhost"),
            ("::", "localhost"),
            ("", "localhost"),
            ("127.0.0.1", "127.0.0.1"),
            ("::1", "[::1]"),
            ("metrics.internal", "metrics.internal"),
        ],
    )
    def test_wildcards_become_localhost_and_ipv6_is_bracketed(self, host, shown):
        """The logged scrape URL names a host a client can actually dial."""
        assert server_metrics._display_host(host) == shown

    def test_url_uses_display_host(self):
        """A wildcard bind is reported as a localhost URL, not http://0.0.0.0."""
        with patch.dict("os.environ", {**LOOPBACK_EPHEMERAL, ENV_HOST: "0.0.0.0"}):
            exporter = start_metrics_exporter()
        assert exporter.url == f"http://localhost:{exporter.port}/metrics"
        assert "guardrails_nonstream" in _scrape(exporter.url) or _scrape(exporter.url) == ""


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

    def test_refuses_when_a_noop_meter_provider_was_set_first(self):
        """A pre-set NoOpMeterProvider looks like the unset state to the module-path check,
        so the post-install identity check must catch that ours did not take."""
        from opentelemetry.metrics import NoOpMeterProvider

        otel_metrics.set_meter_provider(NoOpMeterProvider())
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with pytest.raises(MetricsExporterConfigError, match="had no effect"):
                start_metrics_exporter()
        assert get_active_metrics_exporter() is None
        assert server_metrics._installed_provider is None

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


class TestRunningMetricsExporter:
    def test_stops_the_exporter_it_started(self):
        """An exporter started on entry runs inside the block and is stopped on exit."""
        with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS) as exporter:
            assert exporter is not None
            assert get_active_metrics_exporter() is exporter
        assert get_active_metrics_exporter() is None

    def test_leaves_an_exporter_that_was_already_running(self):
        """An exporter running on entry is yielded, even with export off in the settings, and still runs on exit."""
        running = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        with running_metrics_exporter(MetricsExporterSettings()) as exporter:
            assert exporter is running
        assert get_active_metrics_exporter() is running

    def test_warns_when_the_running_exporter_has_other_settings(self, caplog):
        """Asking for an exporter other than the running one warns that the running one is served instead."""
        running = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        other = MetricsExporterSettings(MetricsExporter.PROMETHEUS, "127.0.0.1", 0, "other-service")
        with caplog.at_level(logging.WARNING, logger=server_metrics.__name__):
            with running_metrics_exporter(other) as exporter:
                assert exporter is running
        assert f"already running on {running.url}" in caplog.text

    def test_reusing_an_exporter_with_the_same_settings_does_not_warn(self, caplog):
        """Asking for the running exporter's own settings, as a lifespan does under the CLI, logs no warning."""
        start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        with caplog.at_level(logging.WARNING, logger=server_metrics.__name__):
            with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
                pass
        assert "already running" not in caplog.text

    def test_leaves_an_exporter_started_inside_the_block(self):
        """With export off, an exporter another app starts inside the block still runs on exit."""
        with running_metrics_exporter(MetricsExporterSettings()) as exporter:
            assert exporter is None
            started_by_another_app = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        assert get_active_metrics_exporter() is started_by_another_app

    def test_leaves_an_exporter_that_replaced_the_one_it_started(self):
        """An exporter started after this one was stopped inside the block belongs to its starter, and still runs."""
        with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
            shutdown_metrics_exporter()
            replacement = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        assert get_active_metrics_exporter() is replacement

    def test_keeps_the_exporter_until_the_last_holder_exits(self):
        """When the block that started the exporter exits first, the exporter runs until the block reusing it exits."""
        starter = running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        reuser = running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        exporter = starter.__enter__()
        assert reuser.__enter__() is exporter

        starter.__exit__(None, None, None)
        assert get_active_metrics_exporter() is exporter

        reuser.__exit__(None, None, None)
        assert get_active_metrics_exporter() is None

    def test_stops_the_exporter_when_its_starter_exits_last(self):
        """When the block reusing the exporter exits first, the exporter stops as the block that started it exits."""
        with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS) as exporter:
            with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
                pass
            assert get_active_metrics_exporter() is exporter
        assert get_active_metrics_exporter() is None

    def test_a_block_without_an_exporter_does_not_hold_one(self):
        """A block that got no exporter, with export off, doesn't keep running one a later block started."""
        without_export = running_metrics_exporter(MetricsExporterSettings())
        assert without_export.__enter__() is None

        with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
            pass
        assert get_active_metrics_exporter() is None

        without_export.__exit__(None, None, None)
        assert get_active_metrics_exporter() is None

    def test_holders_leave_an_exporter_the_cli_started_running(self):
        """Blocks reusing an exporter started directly, as the CLI does, leave it running as the last of them exits."""
        started = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
            with running_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS):
                pass
        assert get_active_metrics_exporter() is started

    def test_resolves_the_settings_from_the_environment_when_given_none(self, caplog):
        """Given no settings, it asks for the environment's, and warns when the running exporter's differ."""
        running = start_metrics_exporter(MetricsExporterSettings(MetricsExporter.PROMETHEUS, "127.0.0.1", 0, "other"))
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with caplog.at_level(logging.WARNING, logger=server_metrics.__name__):
                with running_metrics_exporter() as exporter:
                    assert exporter is running
        assert f"already running on {running.url}" in caplog.text


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

    def test_lifespan_leaves_an_exporter_it_did_not_start(self, tmp_path):
        """An exporter already running, such as the CLI's or another app's, is served and still runs after shutdown."""
        running = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        with patch.dict("os.environ", {}, clear=True), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with TestClient(api.app):
                assert get_active_metrics_exporter() is running
        assert get_active_metrics_exporter() is running

    def test_lifespan_warns_when_the_running_exporter_has_other_settings(self, tmp_path, caplog):
        """With the hooks server's exporter running, the lifespan warns that it serves that one instead of its own."""
        hooks_exporter = start_metrics_exporter(metrics_exporter_settings(LOOPBACK_EPHEMERAL))
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            with caplog.at_level(logging.WARNING, logger=server_metrics.__name__):
                with TestClient(api.app):
                    assert get_active_metrics_exporter() is hooks_exporter
        assert f"already running on {hooks_exporter.url}" in caplog.text

    def test_lifespan_under_the_cli_does_not_warn(self, tmp_path, caplog):
        """With the exporter the CLI started from the same environment, the lifespan logs no warning."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            start_metrics_exporter()
            with caplog.at_level(logging.WARNING, logger=server_metrics.__name__):
                with TestClient(api.app):
                    pass
        assert "already running" not in caplog.text

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
        arrived = asyncio.Event()

        async def _blocked_generate(messages, req_id, request_span=None, **kwargs):
            """Stand-in pipeline that signals arrival, then parks until the test releases the gate.

            The queue worker increments its busy count before it awaits this
            coroutine, so ``arrived`` firing already implies
            ``num_busy_workers() == 1``.
            """
            arrived.set()
            await gate.wait()
            return {"role": "assistant", "content": "done"}

        async with iorails:
            idle = _scrape(exporter.url)
            assert "guardrails_nonstream_queued{" in idle
            assert "guardrails_nonstream_active{" in idle

            with patch.object(iorails, "_do_generate", _blocked_generate):
                task = asyncio.create_task(iorails.generate_async(messages=[{"role": "user", "content": "hi"}]))
                # The worker is executing the request once ``arrived`` fires; the
                # timeout only bounds a hang, it does not pace the handshake.
                await asyncio.wait_for(arrived.wait(), timeout=10)
                assert iorails._generate_async_queue.num_busy_workers() == 1
                busy = await asyncio.to_thread(_scrape, exporter.url)
                gate.set()
                await task

        active_line = next(line for line in busy.splitlines() if line.startswith("guardrails_nonstream_active{"))
        assert active_line.endswith(" 1.0")
        assert "guardrails_requests_active" not in busy
        assert "gen_ai_client" not in busy


# Only model-free rails, so no check reaches a model.
HOOKS_METRICS_CONFIG = {
    "models": [],
    "metrics": {"enabled": True},
    "rails": {
        "config": {"regex_detection": {"tool_call": {"Bash": {"patterns": [r"rm\s+-rf"]}}}},
        "tool_call": {"per_tool": {"Bash": ["regex check tool call $argument=command"]}},
    },
}

NO_METRICS_WARNING = (
    "The metrics exporter is running, but the config doesn't set `metrics.enabled: true`, so the engine "
    "records no metrics, and the scrape endpoint shows none."
)


def _hooks_app(config: dict):
    """A hooks server for *config*."""
    return create_hooks_app(RailsConfig.from_content(config=config))


def _bash_hook(command: str) -> dict:
    """A Claude Code `PreToolUse` hook for a Bash call running *command*."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": "toolu_01",
    }


@pytest.fixture
def hooks_server_env(monkeypatch):
    """No metrics variables but those a test sets, and the process's deployment type restored after the test."""
    for name in (ENV_EXPORTER, ENV_HOST, ENV_PORT, ENV_SERVICE_NAME, "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(usage_telemetry, "_deployment_type_override", None)


@pytest.mark.usefixtures("hooks_server_env")
class TestHooksServer:
    def test_defaults_keep_clear_of_the_main_server(self):
        """Unset variables give the hooks server no exporter, loopback port 9465, and a service name of its own."""
        settings = metrics_exporter_settings({})

        assert settings == MetricsExporterSettings(
            MetricsExporter.NONE, "127.0.0.1", 9465, "nemoguardrails-hooks-server"
        )

    def test_lifespan_exports_the_admission_queue_gauges_from_startup(self):
        """The lifespan starts the exporter, showing the engine's queue gauges as the hooks service, and stops it."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with TestClient(_hooks_app(HOOKS_METRICS_CONFIG)):
                exporter = get_active_metrics_exporter()
                assert exporter is not None
                output = _scrape(exporter.url)
            assert get_active_metrics_exporter() is None

        assert "guardrails_nonstream_queued{" in output
        assert "guardrails_nonstream_active{" in output
        assert 'service_name="nemoguardrails-hooks-server"' in output

    def test_scrape_after_checks_holds_only_the_admission_queue_series(self):
        """After an allow and a deny, the scrape has no request series and nothing of the tool call."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with TestClient(_hooks_app(HOOKS_METRICS_CONFIG)) as client:
                assert client.post("/v1/hooks/claude-code", json=_bash_hook("npm test")).json() == {}
                denied = client.post("/v1/hooks/claude-code", json=_bash_hook("rm -rf /tmp/secret-marker-7f3a"))
                assert denied.json()["hookSpecificOutput"]["permissionDecision"] == "deny"
                output = _scrape(get_active_metrics_exporter().url)

        assert "guardrails_nonstream_queued{" in output
        assert "guardrails_requests" not in output
        assert "Bash" not in output
        assert "secret-marker-7f3a" not in output

    def test_lifespan_serves_the_exporter_the_cli_started_and_leaves_it_to_the_cli(self):
        """The lifespan serves the exporter the CLI started before the app, and leaves it for the CLI to stop."""
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            started = start_metrics_exporter(metrics_exporter_settings())
            with TestClient(_hooks_app(HOOKS_METRICS_CONFIG)):
                assert get_active_metrics_exporter() is started
        assert get_active_metrics_exporter() is started

    def test_lifespan_leaves_another_apps_exporter_running_with_its_own_export_off(self):
        """With no metrics variables, the lifespan serves an exporter another app started, and leaves it running."""
        running = start_metrics_exporter(LOOPBACK_EPHEMERAL_SETTINGS)
        with TestClient(_hooks_app(HOOKS_METRICS_CONFIG)):
            assert get_active_metrics_exporter() is running
        assert get_active_metrics_exporter() is running

    @pytest.mark.parametrize("first_to_exit", ["main", "hooks"])
    def test_main_and_hooks_lifespans_in_one_process_share_the_exporter_until_both_exit(self, tmp_path, first_to_exit):
        """The main server's lifespan starts the exporter, the hooks app's reuses it, and it serves until both exit."""
        second_to_exit = "hooks" if first_to_exit == "main" else "main"
        clients = {"main": TestClient(api.app), "hooks": TestClient(_hooks_app(HOOKS_METRICS_CONFIG))}
        running = []
        with patch.dict("os.environ", LOOPBACK_EPHEMERAL), patch.object(api.app, "rails_config_path", str(tmp_path)):
            try:
                for name in ("main", "hooks"):
                    clients[name].__enter__()
                    running.append(name)
                exporter = get_active_metrics_exporter()

                clients[first_to_exit].__exit__(None, None, None)
                running.remove(first_to_exit)
                assert get_active_metrics_exporter() is exporter
                _scrape(exporter.url)

                clients[second_to_exit].__exit__(None, None, None)
                running.remove(second_to_exit)
                assert get_active_metrics_exporter() is None
            finally:
                for name in running:
                    clients[name].__exit__(None, None, None)

    def test_lifespan_releases_the_exporter_when_the_engine_fails_to_start(self, monkeypatch):
        """An engine that fails to start, with the exporter already running, fails startup and releases the exporter."""
        app = _hooks_app(HOOKS_METRICS_CONFIG)
        exporters_at_startup = []

        async def fail_to_start():
            """Note the exporter running when the engine starts, then fail."""
            exporters_at_startup.append(get_active_metrics_exporter())
            raise RuntimeError("Failed to start engine")

        monkeypatch.setattr(app.state.engine, "startup", fail_to_start)

        with patch.dict("os.environ", LOOPBACK_EPHEMERAL):
            with pytest.raises(RuntimeError, match="Failed to start engine"):
                with TestClient(app):
                    pass  # pragma: no cover - startup raises before the body runs

        [exporter_at_startup] = exporters_at_startup
        assert exporter_at_startup is not None
        assert get_active_metrics_exporter() is None

    def test_lifespan_without_variables_starts_no_exporter(self):
        """Without the variables, the hooks server exports no metrics."""
        with TestClient(_hooks_app(HOOKS_METRICS_CONFIG)):
            assert get_active_metrics_exporter() is None

    @pytest.mark.parametrize(
        "metrics_enabled, environ, warnings",
        [(False, LOOPBACK_EPHEMERAL, [NO_METRICS_WARNING]), (True, LOOPBACK_EPHEMERAL, []), (False, {}, [])],
        ids=["exporter-without-metrics", "exporter-with-metrics", "no-exporter"],
    )
    def test_exporter_for_a_config_without_metrics_is_warned_about(self, caplog, metrics_enabled, environ, warnings):
        """The server warns at startup when the exporter runs and the config doesn't turn metrics on, and only then."""
        config = {**HOOKS_METRICS_CONFIG, "metrics": {"enabled": metrics_enabled}}

        with patch.dict("os.environ", environ), caplog.at_level(logging.WARNING, logger=hooks_app.__name__):
            with TestClient(_hooks_app(config)):
                pass

        assert [record.getMessage() for record in caplog.records if record.name == hooks_app.__name__] == warnings
