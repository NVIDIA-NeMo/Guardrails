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

"""Opt-in metrics export for the Guardrails server.

The IORails engine records its metrics through the OpenTelemetry *API* only;
something has to install an SDK ``MeterProvider`` or every emission is a
silent no-op.  For library users that "something" is their application.  For
the packaged ``nemoguardrails server`` command it is this module: it builds a
``MeterProvider`` backed by a ``PrometheusMetricReader``, serves the scrape
endpoint on a dedicated port, and stops the listener with the server.

Scope is deliberately narrow.  Only the non-streaming admission-queue metrics
are exported (see :data:`EXPORTED_INSTRUMENT_PATTERNS`); everything else the
engine records is dropped by an SDK view so the scrape output stays small and
stable while the GenAI semantic conventions are still evolving.

Lifecycle: the OpenTelemetry API allows exactly one global ``MeterProvider``
per process and never lets it be replaced, so the provider is installed once
and kept for the life of the process (the SDK shuts it down at interpreter
exit).  ``start``/``shutdown`` only manage the HTTP listener, which is what
lets an embedded application or a test client cycle the server lifespan
repeatedly without losing metrics export.

All OpenTelemetry SDK and Prometheus imports are deferred into
:func:`_load_sdk` so that importing this module never requires the optional
dependencies; they are pulled in by the ``server`` extra.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.view import View
    from prometheus_client import CollectorRegistry

log = logging.getLogger(__name__)

ENV_EXPORTER = "NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER"
ENV_HOST = "NEMO_GUARDRAILS_SERVER_METRICS_HOST"
ENV_PORT = "NEMO_GUARDRAILS_SERVER_METRICS_PORT"
ENV_SERVICE_NAME = "NEMO_GUARDRAILS_SERVER_METRICS_SERVICE_NAME"

DEFAULT_HOST = "0.0.0.0"
# 9464 is the port the OpenTelemetry Prometheus exporters use by default.
DEFAULT_PORT = 9464
DEFAULT_SERVICE_NAME = "nemoguardrails-server"

# Instrument-name globs that survive the export view.  Anything the engine
# records that does not match is dropped before it reaches Prometheus.
EXPORTED_INSTRUMENT_PATTERNS = ("guardrails.nonstream.*",)

INSTALL_HINT = 'pip install "nemoguardrails[server]"'


class MetricsExporter(str, Enum):
    """Supported values for ``NEMO_GUARDRAILS_SERVER_METRICS_EXPORTER``."""

    NONE = "none"
    PROMETHEUS = "prometheus"


class MetricsExporterConfigError(ValueError):
    """The metrics exporter cannot start with the current configuration.

    Raised for invalid settings, missing optional dependencies, a port that
    cannot be bound, and a global ``MeterProvider`` that some other code
    already installed.  Callers turn it into a startup failure with the
    message shown to the operator verbatim.
    """


@dataclass(frozen=True)
class MetricsExporterSettings:
    """Resolved exporter settings; :meth:`from_env` is the only reader of the env vars."""

    exporter: MetricsExporter = MetricsExporter.NONE
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    service_name: str = DEFAULT_SERVICE_NAME

    @property
    def enabled(self) -> bool:
        """Whether an exporter is configured at all."""
        return self.exporter is not MetricsExporter.NONE

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "MetricsExporterSettings":
        """Resolve settings from environment variables, raising :class:`MetricsExporterConfigError` on invalid values."""
        env = os.environ if environ is None else environ

        raw_exporter = env.get(ENV_EXPORTER, MetricsExporter.NONE.value).strip().lower() or MetricsExporter.NONE.value
        try:
            exporter = MetricsExporter(raw_exporter)
        except ValueError:
            supported = ", ".join(e.value for e in MetricsExporter)
            raise MetricsExporterConfigError(
                f"Unsupported {ENV_EXPORTER}={raw_exporter!r}. Supported values: {supported}."
            ) from None

        raw_port = env.get(ENV_PORT, str(DEFAULT_PORT)).strip()
        try:
            port = int(raw_port)
        except ValueError:
            raise MetricsExporterConfigError(f"{ENV_PORT} must be an integer, got {raw_port!r}.") from None
        if not 0 <= port <= 65535:
            raise MetricsExporterConfigError(f"{ENV_PORT} must be between 0 and 65535, got {port}.")

        host = env.get(ENV_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        # OTEL_SERVICE_NAME is the SDK-standard knob; the NEMO_ variable wins
        # when both are set so server operators have one obvious override.
        service_name = (
            env.get(ENV_SERVICE_NAME, "").strip() or env.get("OTEL_SERVICE_NAME", "").strip() or DEFAULT_SERVICE_NAME
        )

        return cls(exporter=exporter, host=host, port=port, service_name=service_name)


def _load_sdk():
    """Import the optional SDK + exporter modules or raise an actionable error."""
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.view import DropAggregation, View
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import CollectorRegistry, start_http_server
    except ImportError as e:
        raise MetricsExporterConfigError(
            f"{ENV_EXPORTER}=prometheus requires the OpenTelemetry SDK and Prometheus exporter "
            f"(missing: {e.name}). Install them with: {INSTALL_HINT}"
        ) from e
    return PrometheusMetricReader, MeterProvider, DropAggregation, View, Resource, CollectorRegistry, start_http_server


def build_views() -> List["View"]:
    """Return the SDK views that restrict export to :data:`EXPORTED_INSTRUMENT_PATTERNS`.

    The SDK applies *only* the matching views to an instrument and falls back
    to its default aggregation when none match, so a catch-all drop view must
    be paired with an explicit keep view for every exported pattern.
    """
    _, _, DropAggregation, View, _, _, _ = _load_sdk()
    views = [View(instrument_name=pattern) for pattern in EXPORTED_INSTRUMENT_PATTERNS]
    views.append(View(instrument_name="*", aggregation=DropAggregation()))
    return views


def build_meter_provider(settings: MetricsExporterSettings, registry: "CollectorRegistry") -> "MeterProvider":
    """Build a ``MeterProvider`` that exports the scoped metrics into ``registry``."""
    PrometheusMetricReader, MeterProvider, _, _, Resource, _, _ = _load_sdk()
    reader = PrometheusMetricReader(registry=registry)
    resource = Resource.create({"service.name": settings.service_name})
    return MeterProvider(resource=resource, metric_readers=[reader], views=build_views())


def _is_meter_provider_configured(provider: object) -> bool:
    """True when something other than the API's own placeholder is installed.

    Before ``set_meter_provider`` runs, the API hands out its no-op or proxy
    provider, both defined inside the ``opentelemetry.metrics`` package.  Any
    real provider comes from an SDK package, so the defining module is the
    stable way to tell the two apart without importing private names.
    """
    return not type(provider).__module__.startswith("opentelemetry.metrics")


@dataclass(frozen=True)
class _InstalledProvider:
    """The process-wide ``MeterProvider`` this module installed, plus the
    registry its Prometheus reader writes into."""

    provider: "MeterProvider"
    registry: "CollectorRegistry"
    service_name: str


_installed_provider: Optional[_InstalledProvider] = None


def _install_meter_provider(settings: MetricsExporterSettings) -> _InstalledProvider:
    """Install the global ``MeterProvider`` on first use and return it thereafter.

    The OpenTelemetry API refuses to replace a provider once set, so this is
    a one-way door per process: the first exporter start decides the
    ``service.name`` resource and the export view, and later starts reuse
    them.  The SDK registers its own ``atexit`` hook to shut the provider
    down, so nothing here needs to.
    """
    global _installed_provider
    from opentelemetry import metrics as otel_metrics

    _, _, _, _, _, CollectorRegistry, _ = _load_sdk()

    if _installed_provider is not None:
        if _installed_provider.service_name != settings.service_name:
            log.warning(
                "Metrics exporter restarted with service.name=%s but the process-wide MeterProvider "
                "already carries service.name=%s; keeping the original.",
                settings.service_name,
                _installed_provider.service_name,
            )
        return _installed_provider

    if _is_meter_provider_configured(otel_metrics.get_meter_provider()):
        raise MetricsExporterConfigError(
            "A global OpenTelemetry MeterProvider is already configured, so the server cannot install "
            f"its own. Unset {ENV_EXPORTER} and expose that provider yourself, or remove the other "
            "MeterProvider setup (for example in a root config.py)."
        )

    registry = CollectorRegistry()
    provider = build_meter_provider(settings, registry)
    otel_metrics.set_meter_provider(provider)
    _installed_provider = _InstalledProvider(provider=provider, registry=registry, service_name=settings.service_name)
    return _installed_provider


class PrometheusMetricsExporter:
    """A running Prometheus scrape endpoint serving the installed provider's registry.

    Create it through :func:`start_metrics_exporter`; the constructor binds
    the listener so a half-built exporter is never handed out.  Only the HTTP
    listener belongs to this object: the ``MeterProvider`` behind it is
    process-wide (see :func:`_install_meter_provider`).
    """

    def __init__(self, settings: MetricsExporterSettings, installed: _InstalledProvider):
        """Bind the scrape listener for ``settings`` against ``installed``'s registry."""
        _, _, _, _, _, _, start_http_server = _load_sdk()

        self.settings = settings
        self.provider = installed.provider
        self.registry = installed.registry
        try:
            self._server, self._thread = start_http_server(settings.port, addr=settings.host, registry=self.registry)
        except OSError as e:
            raise MetricsExporterConfigError(
                f"Cannot bind the Prometheus metrics endpoint to {settings.host}:{settings.port}: {e}. "
                f"Choose another port with {ENV_PORT} or --metrics-port."
            ) from e

    @property
    def port(self) -> int:
        """The bound port; differs from the requested one only when port 0 was asked for."""
        return self._server.server_port

    @property
    def url(self) -> str:
        """Scrape URL of the running listener."""
        return f"http://{self.settings.host}:{self.port}/metrics"

    def shutdown(self) -> None:
        """Stop the listener and join its thread.  The provider keeps collecting
        so a later :func:`start_metrics_exporter` resumes export seamlessly."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


_active_exporter: Optional[PrometheusMetricsExporter] = None


def get_active_metrics_exporter() -> Optional[PrometheusMetricsExporter]:
    """Return the running exporter, or ``None`` when none is active."""
    return _active_exporter


def start_metrics_exporter(settings: Optional[MetricsExporterSettings] = None) -> Optional[PrometheusMetricsExporter]:
    """Start the configured exporter and return it.

    Returns ``None`` when no exporter is configured.  Repeated calls while an
    exporter is running return it unchanged, which lets both the CLI entry
    point and the FastAPI lifespan call it without coordinating: whichever
    runs first wins, and the provider is in place before the first
    ``IORails`` instance is built.  After :func:`shutdown_metrics_exporter`
    the next call starts a fresh listener against the same provider.
    """
    global _active_exporter
    if _active_exporter is not None:
        return _active_exporter

    if settings is None:
        settings = MetricsExporterSettings.from_env()
    if not settings.enabled:
        return None

    installed = _install_meter_provider(settings)
    _active_exporter = PrometheusMetricsExporter(settings, installed)
    log.info(
        "Prometheus metrics endpoint listening on %s (service.name=%s, exporting %s)",
        _active_exporter.url,
        installed.service_name,
        ", ".join(EXPORTED_INSTRUMENT_PATTERNS),
    )
    return _active_exporter


def shutdown_metrics_exporter() -> None:
    """Stop the running exporter's listener, if any.  Safe to call repeatedly."""
    global _active_exporter
    if _active_exporter is None:
        return
    exporter, _active_exporter = _active_exporter, None
    exporter.shutdown()
    log.info("Prometheus metrics endpoint stopped")
