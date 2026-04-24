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
import logging
import os
import platform
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

_USAGE_STATS_SERVER = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
_NVIDIA_CLIENT_ID = "184482118588404"
_NVIDIA_EVENT_PROTOCOL = "1.6"
_NVIDIA_EVENT_SYS_VER = "nemo-telemetry/1.0"
_HEARTBEAT_INTERVAL_S = 600
_AUDIT_FILE_MAX_BYTES = 10 * 1024 * 1024
_CONFIG_DIR = Path.home() / ".config" / "nemoguardrails"
_AUDIT_FILE = _CONFIG_DIR / "usage_stats.json"
_DO_NOT_TRACK_FILE = _CONFIG_DIR / "do_not_track"

_reported = False
_lock = threading.Lock()


class NemoSourceEnum(str, Enum):
    GUARDRAILS = "guardrails"


class ContextEnum(str, Enum):
    EMBEDDED = "embedded"
    SERVER = "server"
    UNDEFINED = "undefined"


class RailsEngineEnum(str, Enum):
    LLMRAILS = "LLMRails"
    IORAILS = "IORails"
    UNDEFINED = "undefined"


class EventTypeEnum(str, Enum):
    STARTUP = "startup"
    HEARTBEAT = "heartbeat"


class TelemetryEvent(BaseModel):
    """Abstract base for telemetry events.

    Subclasses must define ``_event_name`` as a ClassVar. The optional
    ``_schema_version`` ClassVar is used by the payload builder to set
    ``eventSchemaVer`` in the NVIDIA telemetry envelope.

    Attributes:
        _event_name: Unique name for this event type (e.g. "guardrails_usage_event").
        _schema_version: Schema version string, defaults to "1.0".

    Raises:
        TypeError: If a subclass fails to define ``_event_name``.
    """

    _event_name: ClassVar[str]
    _schema_version: ClassVar[str] = "1.0"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_event_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_event_name' class variable")


class GuardrailsUsageEvent(TelemetryEvent):
    """Instance-level usage census event for NeMo Guardrails.

    Emitted once at startup and periodically as heartbeats. Contains no
    user content, model names, or request-level data. All fields have
    sensible defaults so partial events (e.g. heartbeats) remain readable
    in dashboards.
    """

    _event_name: ClassVar[str] = "guardrails_usage_event"
    _schema_version: ClassVar[str] = "1.0"

    uuid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Random UUID4 per process. Not traceable to any user or machine.",
    )
    nemoguardrails_version: str = Field(
        default="unknown",
        alias="nemoguardrailsVersion",
        description='Installed package version (e.g. "0.21.0"). "unknown" if unavailable.',
    )
    python_version: str = Field(
        default="unknown",
        alias="pythonVersion",
        description='Python interpreter version (e.g. "3.13.7").',
    )
    platform: str = Field(
        default="unknown",
        description='OS and architecture string (e.g. "Linux-5.15.0-x86_64-with-glibc2.35").',
    )
    os_name: str = Field(
        default="unknown",
        alias="osName",
        description='Operating system name (e.g. "Darwin", "Linux", "Windows").',
    )
    colang_version: str = Field(
        default="unknown",
        alias="colangVersion",
        description='Colang version in use. Values: "1.0", "2.x", or "unknown" if no config.',
    )
    llm_providers: List[str] = Field(
        default_factory=list,
        alias="llmProviders",
        description='LLM engine names, sorted (e.g. ["nim", "openai"]). Engine identifiers, not model names.',
    )
    num_rails_configured: int = Field(
        default=0,
        alias="numRailsConfigured",
        description="Total count of configured rail flows across all rail types.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    rail_types_in_use: List[str] = Field(
        default_factory=list,
        alias="railTypesInUse",
        description="Active rail categories. Possible values: input, output, retrieval, tool_input, tool_output, dialog.",
    )
    tracing_enabled: bool = Field(
        default=False,
        alias="tracingEnabled",
        description="Whether the tracing subsystem is enabled.",
    )
    context: ContextEnum = Field(
        default=ContextEnum.UNDEFINED,
        description='How guardrails was started. "embedded" via LLMRails, "server" via FastAPI.',
    )
    rails_engine: RailsEngineEnum = Field(
        default=RailsEngineEnum.UNDEFINED,
        alias="railsEngine",
        description='Which rails engine class is in use. "LLMRails" or "IORails".',
    )
    has_knowledge_base: bool = Field(
        default=False,
        alias="hasKnowledgeBase",
        description="Whether a knowledge base (document set) is configured.",
    )
    streaming_configured: bool = Field(
        default=False,
        alias="streamingConfigured",
        description="Whether streaming output is enabled.",
    )
    builtin_features: List[str] = Field(
        default_factory=list,
        alias="builtinFeatures",
        description="Active built-in library features, sorted. Only our feature names, never user-defined.",
    )
    num_custom_flows: int = Field(
        default=0,
        alias="numCustomFlows",
        description="Count of user-defined Colang flows. Indicates dialog/topical rail usage without exposing names.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    timestamp: float = Field(
        default=0.0,
        description="Unix timestamp (seconds since epoch) when data was collected.",
    )
    event: EventTypeEnum = Field(
        default=EventTypeEnum.STARTUP,
        description="Event type. startup for initial report, heartbeat for periodic pings.",
    )

    model_config = {"populate_by_name": True, "validate_assignment": True, "use_enum_values": True}


def _is_usage_stats_enabled() -> bool:
    """Check whether usage reporting is enabled.

    Respects three opt-out mechanisms, any of which disables reporting:
    the ``NEMO_GUARDRAILS_NO_USAGE_STATS`` env var, the industry-standard
    ``DO_NOT_TRACK`` env var, or the presence of a
    ``~/.config/nemoguardrails/do_not_track`` file.

    Returns:
        True if reporting should proceed, False if any opt-out is active.
    """
    if os.environ.get("NEMO_GUARDRAILS_NO_USAGE_STATS", "0") == "1":
        return False
    if os.environ.get("DO_NOT_TRACK", "0") == "1":
        return False
    if _DO_NOT_TRACK_FILE.is_file():
        return False
    return True


_KNOWN_BUILTIN_FLOWS = {
    "activefence moderation on input": "activefence",
    "activefence moderation on input detailed": "activefence",
    "activefence moderation on output": "activefence",
    "ai defense inspect prompt": "ai_defense",
    "ai defense inspect response": "ai_defense",
    "alignscore check facts": "factchecking",
    "autoalign check input": "autoalign",
    "autoalign check output": "autoalign",
    "autoalign factcheck output": "autoalign",
    "autoalign groundedness output": "autoalign",
    "clavata check for": "clavata",
    "clavata check input": "clavata",
    "clavata check output": "clavata",
    "cleanlab trustworthiness": "cleanlab",
    "content safety check input": "content_safety",
    "content safety check output": "content_safety",
    "crowdstrike aidr guard input": "crowdstrike_aidr",
    "crowdstrike aidr guard output": "crowdstrike_aidr",
    "detect pii on input": "sensitive_data_detection",
    "detect pii on output": "sensitive_data_detection",
    "detect pii on retrieval": "sensitive_data_detection",
    "detect sensitive data on input": "sensitive_data_detection",
    "detect sensitive data on output": "sensitive_data_detection",
    "detect sensitive data on retrieval": "sensitive_data_detection",
    "fiddler bot faithfulness": "fiddler",
    "fiddler bot safety": "fiddler",
    "fiddler user safety": "fiddler",
    "gliner detect pii on input": "gliner",
    "gliner detect pii on output": "gliner",
    "gliner detect pii on retrieval": "gliner",
    "gliner mask pii on input": "gliner",
    "gliner mask pii on output": "gliner",
    "gliner mask pii on retrieval": "gliner",
    "guardrailsai check input": "guardrails_ai",
    "guardrailsai check output": "guardrails_ai",
    "hallucination warning": "hallucination",
    "injection detection": "injection_detection",
    "jailbreak detection heuristics": "jailbreak_detection",
    "jailbreak detection model": "jailbreak_detection",
    "llama guard check input": "llama_guard",
    "llama guard check output": "llama_guard",
    "mask pii on input": "sensitive_data_detection",
    "mask pii on output": "sensitive_data_detection",
    "mask pii on retrieval": "sensitive_data_detection",
    "mask sensitive data on input": "sensitive_data_detection",
    "mask sensitive data on output": "sensitive_data_detection",
    "mask sensitive data on retrieval": "sensitive_data_detection",
    "pangea ai guard input": "pangea",
    "pangea ai guard output": "pangea",
    "patronus api check output": "patronusai",
    "patronus lynx check output hallucination": "patronusai",
    "policyai moderation on input": "policyai",
    "policyai moderation on output": "policyai",
    "protect prompt": "prompt_security",
    "protect response": "prompt_security",
    "regex check input": "regex",
    "regex check output": "regex",
    "regex check retrieval": "regex",
    "self check facts": "self_check",
    "self check hallucination": "self_check",
    "self check input": "self_check",
    "self check output": "self_check",
    "topic safety check input": "topic_safety",
    "trend ai guard input": "trend_micro",
    "trend ai guard output": "trend_micro",
}


def _normalize_flow_name(flow_name: str) -> str:
    """Strip parameter and argument syntax from a Colang flow name.

    Matches the semantics of ``nemoguardrails.colang.v1_0.runtime.flows._normalize_flow_id``.
    For example, ``"content safety check input $model=main"`` normalizes to
    ``"content safety check input"``, and ``"flow_id(arg1, arg2)"`` to ``"flow_id"``.

    Args:
        flow_name: The raw flow name from a rails config, possibly with args.

    Returns:
        The flow name with parameter/argument suffixes stripped.
    """
    flow_name = flow_name.strip()
    if "(" in flow_name:
        flow_name = flow_name.split("(")[0]
    elif "$" in flow_name:
        flow_name = flow_name.split("$")[0]
    return flow_name.strip()


def _detect_builtin_features(config: "RailsConfig") -> List[str]:
    """Detect which built-in NeMo Guardrails library features are active.

    Uses two signals: (1) fields on ``RailsConfigData`` that differ from
    their defaults (explicit config), and (2) exact-match flow names
    against a known set of built-in library flows. Only our own feature
    names are ever reported, never user-defined flow names.

    Args:
        config: The ``RailsConfig`` instance to inspect.

    Returns:
        Sorted list of active built-in feature names (e.g.
        ``["content_safety", "jailbreak_detection"]``). Empty list if no
        built-in features are active or ``config.rails`` is missing.
    """
    features = set()

    rails = getattr(config, "rails", None)
    if rails is None:
        return []

    config_data = getattr(rails, "config", None)
    if config_data is not None:
        config_type = type(config_data)
        try:
            default = config_type()
            for field_name in getattr(config_type, "model_fields", {}):
                current_val = getattr(config_data, field_name, None)
                default_val = getattr(default, field_name, None)
                if current_val != default_val:
                    features.add(field_name)
        except Exception:
            pass

    all_flows = []
    for rail_group in ["input", "output", "retrieval", "tool_output", "tool_input"]:
        group = getattr(rails, rail_group, None)
        if group is not None:
            all_flows.extend(getattr(group, "flows", []))

    for flow_name in all_flows:
        normalized = _normalize_flow_name(flow_name)
        feature = _KNOWN_BUILTIN_FLOWS.get(normalized)
        if feature is not None:
            features.add(feature)

    return sorted(features)


def _collect_usage_data(config: Optional["RailsConfig"], context: str) -> GuardrailsUsageEvent:
    """Collect anonymous usage data into a ``GuardrailsUsageEvent``.

    Always populates system fields (version, platform, Python version).
    When ``config`` is provided, additionally populates config-derived
    fields: LLM provider names, rail types in use, built-in features,
    custom flow count, and feature flags. Never reads model names,
    prompts, or user content.

    Args:
        config: The ``RailsConfig`` to inspect, or ``None`` for a
            system-only event (e.g. from the server startup context).
        context: How guardrails was started, e.g. ``"embedded"`` or
            ``"server"``. Coerced to ``ContextEnum.UNDEFINED`` if falsy.

    Returns:
        A fully populated ``GuardrailsUsageEvent``.
    """
    data = GuardrailsUsageEvent()
    data.timestamp = time.time()
    data.context = context or ContextEnum.UNDEFINED
    data.event = EventTypeEnum.STARTUP
    data.rails_engine = RailsEngineEnum.UNDEFINED

    try:
        from importlib.metadata import version

        data.nemoguardrails_version = version("nemoguardrails")
    except Exception:
        data.nemoguardrails_version = "unknown"

    data.python_version = sys.version.split()[0]
    data.platform = platform.platform()
    data.os_name = platform.system()

    if config is not None:
        data.colang_version = getattr(config, "colang_version", ".1.0")

        engines = set()
        for model in getattr(config, "models", []):
            if hasattr(model, "engine") and model.engine:
                engines.add(model.engine)
        data.llm_providers = sorted(engines)

        rails = getattr(config, "rails", None)
        if rails is not None:
            rail_types = []
            flow_lists = {
                "input": getattr(getattr(rails, "input", None), "flows", []),
                "output": getattr(getattr(rails, "output", None), "flows", []),
                "retrieval": getattr(getattr(rails, "retrieval", None), "flows", []),
                "tool_output": getattr(getattr(rails, "tool_output", None), "flows", []),
                "tool_input": getattr(getattr(rails, "tool_input", None), "flows", []),
            }

            total_rails = 0
            for rail_type, flows in flow_lists.items():
                if flows:
                    rail_types.append(rail_type)
                    total_rails += len(flows)

            dialog = getattr(rails, "dialog", None)
            if dialog is not None:
                single_call = getattr(dialog, "single_call", None)
                if single_call is not None and getattr(single_call, "enabled", False):
                    rail_types.append("dialog")

            data.rail_types_in_use = rail_types
            data.num_rails_configured = total_rails

            output_rails = getattr(rails, "output", None)
            if output_rails is not None:
                streaming = getattr(output_rails, "streaming", None)
                if streaming is not None:
                    data.streaming_configured = getattr(streaming, "enabled", False)

        data.builtin_features = _detect_builtin_features(config)

        flows = getattr(config, "flows", [])
        data.num_custom_flows = sum(1 for f in flows if not f.get("is_system_flow", False))

        tracing = getattr(config, "tracing", None)
        if tracing is not None:
            data.tracing_enabled = getattr(tracing, "enabled", False)

        data.has_knowledge_base = bool(getattr(config, "docs", None))

    return data


def _rotate_audit_file() -> None:
    """Rotate the local audit file when it exceeds the size cap.

    The current ``usage_stats.json`` is renamed to ``usage_stats.json.1``,
    overwriting any previous backup. This bounds on-disk usage at
    approximately ``2 * _AUDIT_FILE_MAX_BYTES``. Errors are silently
    logged at DEBUG level.
    """
    backup = _AUDIT_FILE.with_suffix(".json.1")
    try:
        if backup.exists():
            backup.unlink()
        _AUDIT_FILE.rename(backup)
    except Exception:
        log.debug("Failed to rotate usage audit file", exc_info=True)


def _write_audit_file(data: Dict[str, Any]) -> None:
    """Append a payload to the local audit file as a JSON line.

    Creates the config directory if it does not exist. Rotates the
    audit file when it exceeds ``_AUDIT_FILE_MAX_BYTES``. All errors
    (permission denied, disk full, etc.) are silently logged at DEBUG
    level so telemetry never disrupts the main process.

    Args:
        data: Serialized event payload (already converted to a dict
            via ``model_dump(by_alias=True)``).
    """
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        if _AUDIT_FILE.exists() and _AUDIT_FILE.stat().st_size > _AUDIT_FILE_MAX_BYTES:
            _rotate_audit_file()

        with open(_AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")
    except Exception:
        log.debug("Failed to write usage audit file", exc_info=True)


def _get_iso_timestamp(ts: Optional[float] = None) -> str:
    """Format a Unix timestamp as an ISO 8601 UTC string with millisecond precision.

    Args:
        ts: Unix timestamp (seconds since epoch). If ``None``, uses
            the current UTC time.

    Returns:
        ISO 8601 formatted string ending with ``"Z"``, e.g.
        ``"2026-04-22T18:34:56.789Z"``.
    """
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _build_event(event: TelemetryEvent, ts: Optional[float] = None) -> Dict[str, Any]:
    """Wrap a ``TelemetryEvent`` in the inner event dict for the NVIDIA envelope.

    Always injects ``nemoSource="guardrails"`` into the parameters so
    the backend can route the event correctly.

    Args:
        event: The Pydantic event instance to serialize.
        ts: Optional Unix timestamp for the ``ts`` field; uses current
            time if ``None``.

    Returns:
        A dict with ``ts``, ``name``, and ``parameters`` keys, ready to
        be inserted into the ``events`` array of the envelope.
    """
    params = event.model_dump(by_alias=True)
    params["nemoSource"] = "guardrails"
    return {
        "ts": _get_iso_timestamp(ts),
        "name": event._event_name,
        "parameters": params,
    }


def _build_nvidia_payload(
    events: List[TelemetryEvent],
    client_version: str,
    session_id: str,
    timestamps: Optional[List[Optional[float]]] = None,
) -> Dict[str, Any]:
    """Build the outer NVIDIA telemetry envelope that wraps one or more events.

    All envelope fields other than ``clientVer``, ``sessionId``,
    ``eventSchemaVer``, ``sentTs``, ``cpuArchitecture``, and ``events``
    are hardcoded to ``"undefined"`` or ``"None"`` per the NVIDIA
    telemetry protocol spec. ``eventSchemaVer`` is read from the first
    event's ``_schema_version`` ClassVar.

    Args:
        events: Non-empty list of telemetry events to include.
        client_version: Version string of the calling product, set as
            ``clientVer`` (typically the package version).
        session_id: Session identifier set as ``sessionId`` in the envelope.
        timestamps: Optional per-event Unix timestamps. If ``None``,
            each event is timestamped with the current time.

    Returns:
        The complete envelope as a dict, ready to be JSON-serialized
        and POSTed to the telemetry endpoint.

    Raises:
        ValueError: If ``events`` is empty.
    """
    if not events:
        raise ValueError("at least one event is required to build a payload")
    if timestamps is None:
        timestamps = [None] * len(events)

    return {
        "browserType": "undefined",
        "clientId": _NVIDIA_CLIENT_ID,
        "clientType": "Native",
        "clientVariant": "Release",
        "clientVer": client_version,
        "cpuArchitecture": platform.uname().machine,
        "deviceGdprBehOptIn": "None",
        "deviceGdprFuncOptIn": "None",
        "deviceGdprTechOptIn": "None",
        "deviceId": "undefined",
        "deviceMake": "undefined",
        "deviceModel": "undefined",
        "deviceOS": "undefined",
        "deviceOSVersion": "undefined",
        "deviceType": "undefined",
        "eventProtocol": _NVIDIA_EVENT_PROTOCOL,
        "eventSchemaVer": events[0]._schema_version,
        "eventSysVer": _NVIDIA_EVENT_SYS_VER,
        "externalUserId": "undefined",
        "gdprBehOptIn": "None",
        "gdprFuncOptIn": "None",
        "gdprTechOptIn": "None",
        "idpId": "undefined",
        "integrationId": "undefined",
        "productName": "undefined",
        "productVersion": "undefined",
        "sentTs": _get_iso_timestamp(),
        "sessionId": session_id,
        "userId": "undefined",
        "events": [_build_event(event, ts) for event, ts in zip(events, timestamps)],
    }


def _send_report(event: TelemetryEvent, server_url: str, client_version: str, session_id: str) -> None:
    """POST a single telemetry event to the configured server.

    Fire-and-forget: a single attempt with a 5-second timeout, no
    retries, all exceptions silently logged at DEBUG level. Runs in a
    daemon thread so it never blocks the main process.

    Args:
        event: The telemetry event to send.
        server_url: Full HTTPS URL of the telemetry endpoint.
        client_version: Value to set as ``clientVer`` in the envelope.
        session_id: Value to set as ``sessionId`` in the envelope.
    """
    try:
        timestamp = getattr(event, "timestamp", None)
        envelope = _build_nvidia_payload([event], client_version, session_id, [timestamp])
        payload = json.dumps(envelope).encode("utf-8")
        req = urllib.request.Request(
            server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        log.debug("Failed to send usage report", exc_info=True)


def _usage_reporter_thread(data: GuardrailsUsageEvent, server_url: str) -> None:
    """Run the usage reporter loop in a daemon thread.

    Sends the initial startup payload (to both the audit file and the
    server), then loops forever sending a minimal heartbeat every
    ``_HEARTBEAT_INTERVAL_S`` seconds. The thread is a daemon so it
    dies automatically when the main process exits; no explicit
    shutdown or cleanup is required.

    Args:
        data: The fully populated startup event.
        server_url: Full HTTPS URL of the telemetry endpoint.
    """
    client_version = data.nemoguardrails_version
    session_id = data.uuid

    _write_audit_file(data.model_dump(by_alias=True))
    _send_report(data, server_url, client_version, session_id)

    while True:
        time.sleep(_HEARTBEAT_INTERVAL_S)
        heartbeat = GuardrailsUsageEvent(
            uuid=data.uuid,
            timestamp=time.time(),
            event=EventTypeEnum.HEARTBEAT,
        )
        _write_audit_file(heartbeat.model_dump(by_alias=True))
        _send_report(heartbeat, server_url, client_version, session_id)


def report_usage(
    config: Optional["RailsConfig"] = None,
    context: str = "embedded",
    rails_engine: str = "",
) -> None:
    """Public entrypoint to trigger anonymous usage reporting.

    Idempotent across a process: the first call spawns a daemon thread
    that sends the startup event and heartbeats; subsequent calls are
    no-ops. Respects the triple opt-out (env vars and file). All work
    happens off the calling thread so this function returns immediately.

    Args:
        config: Optional ``RailsConfig`` to introspect. When ``None``
            (e.g. from the server lifespan context), only system-level
            fields are populated.
        context: How guardrails was started (e.g. ``"embedded"``,
            ``"server"``).
        rails_engine: Which engine class is in use (e.g. ``"LLMRails"``,
            ``"IORails"``). Ignored if empty.
    """
    global _reported

    if not _is_usage_stats_enabled():
        return

    with _lock:
        if _reported:
            return
        _reported = True

    try:
        usage_data = _collect_usage_data(config, context)
        if rails_engine:
            usage_data.rails_engine = rails_engine
    except Exception:
        log.debug("Failed to collect usage data", exc_info=True)
        return

    server_url = os.environ.get("NEMO_GUARDRAILS_USAGE_STATS_SERVER", _USAGE_STATS_SERVER)

    t = threading.Thread(
        target=_usage_reporter_thread,
        args=(usage_data, server_url),
        daemon=True,
    )
    t.start()
