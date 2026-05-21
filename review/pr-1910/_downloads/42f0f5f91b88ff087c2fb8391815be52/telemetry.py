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
import random
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from nemoguardrails.colang.v1_0.runtime.flows import _normalize_flow_id

if TYPE_CHECKING:
    from nemoguardrails.rails.llm.config import RailsConfig

log = logging.getLogger(__name__)

_USAGE_STATS_SERVER = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
_NVIDIA_CLIENT_ID = "184482118588404"
_NVIDIA_EVENT_PROTOCOL = "1.6"
_NVIDIA_EVENT_SYS_VER = "nemo-telemetry/1.0"
_AUDIT_FILE_MAX_BYTES = 10 * 1024 * 1024
_CONFIG_DIR: Optional[Path] = None
_AUDIT_FILE: Optional[Path] = None
_DO_NOT_TRACK_FILE: Optional[Path] = None

_session_uuid: Optional[str] = None
_heartbeat_started = False
_deployment_type_override: Optional["DeploymentTypeEnum"] = None
_lock = threading.Lock()


def _safe_home_dir() -> Path:
    """Return the user's home directory without raising at import time."""
    try:
        return Path.home()
    except Exception:
        expanded = os.path.expanduser("~")
        if expanded and expanded != "~":
            return Path(expanded)
        home = os.environ.get("HOME")
        if home:
            return Path(home)
        return Path(tempfile.gettempdir())


def _get_config_dir() -> Path:
    """Return the telemetry config directory, honoring test overrides."""
    if _CONFIG_DIR is not None:
        return Path(_CONFIG_DIR)
    return _safe_home_dir() / ".config" / "nemoguardrails"


def _get_audit_file() -> Path:
    """Return the local audit JSONL path, honoring test overrides."""
    if _AUDIT_FILE is not None:
        return Path(_AUDIT_FILE)
    return _get_config_dir() / "usage_stats.json"


def _get_do_not_track_file() -> Path:
    """Return the do-not-track marker path, honoring test overrides."""
    if _DO_NOT_TRACK_FILE is not None:
        return Path(_DO_NOT_TRACK_FILE)
    return _get_config_dir() / "do_not_track"


def _get_usage_stats_server_url() -> str:
    """Return the telemetry server URL, honoring runtime env overrides."""
    return os.environ.get("NEMO_GUARDRAILS_USAGE_STATS_SERVER", _USAGE_STATS_SERVER)


def _get_heartbeat_interval_s() -> float:
    """Return the heartbeat interval, falling back safely on bad env values."""
    raw_value = os.environ.get("NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S", "600")
    try:
        interval = float(raw_value)
    except ValueError:
        log.debug(
            "Invalid NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S=%r; using default 600",
            raw_value,
        )
        return 600

    if interval <= 0:
        log.debug(
            "Non-positive NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S=%r; using default 600",
            raw_value,
        )
        return 600
    return interval


def _get_version(package_name: str) -> str:
    from importlib.metadata import version

    return version(package_name)


_HEARTBEAT_INTERVAL_S = _get_heartbeat_interval_s()


def _reset_for_fork() -> None:
    """Clear module state in a forked child so it starts a fresh session.

    Registered via ``os.register_at_fork(after_in_child=...)`` at module
    import time on POSIX platforms. Without this, workers spawned by
    ``gunicorn --preload`` (or any other fork-based runtime) would
    inherit the parent's session ID and its ``_heartbeat_started``
    flag, but lose the actual heartbeat thread (POSIX ``fork()`` only
    duplicates the calling thread). The result would be silent: every
    worker would report under the parent's session and never emit
    heartbeats. After this hook runs, the child's first ``report_usage``
    call generates a fresh session UUID and starts its own heartbeat.
    The deployment-type override is intentionally preserved so a forked
    API or CLI worker keeps the attribution claimed by its parent.
    """
    global _session_uuid, _heartbeat_started, _lock
    # This function is registered as an after-in-child fork hook. Do not
    # acquire the inherited lock here: if another thread held it at fork time,
    # the child would deadlock before fork() returns. Replace it instead.
    _lock = threading.Lock()
    _session_uuid = None
    _heartbeat_started = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_for_fork)


class NemoSourceEnum(str, Enum):
    """The NeMo product that created the event.

    Mirrors the shared nemo-telemetry definition so the guardrails event
    plugs into the same ``definitions/types`` block other NeMo products
    (DataDesigner, Safe Synthesizer, Agent Toolkit, etc.) already share.
    """

    INFERENCE = "inference"
    AUDITOR = "auditor"
    DATADESIGNER = "datadesigner"
    EVALUATOR = "evaluator"
    GUARDRAILS = "guardrails"
    SAFE_SYNTHESIZER = "safe-synthesizer"
    ANONYMIZER = "anonymizer"
    AGENT_TOOLKIT = "agent_toolkit"
    UNDEFINED = "undefined"


class DeploymentTypeEnum(str, Enum):
    """How the NeMo product was invoked.

    Mirrors the shared nemo-telemetry definition. ``library`` covers
    direct ``LLMRails`` / ``Guardrails`` use in user code, ``api`` is
    the FastAPI server, ``cli`` is the interactive chat command.
    ``sdk`` and ``nmp`` are inherited from the shared enum and are
    valid values guardrails does not produce itself.
    """

    LIBRARY = "library"
    API = "api"
    CLI = "cli"
    SDK = "sdk"
    NMP = "nmp"
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
        _schema_version: Schema version string used for ``eventSchemaVer`` in the telemetry envelope.

    Raises:
        TypeError: If a subclass fails to define ``_event_name``.
    """

    _event_name: ClassVar[str]
    _schema_version: ClassVar[str] = "1.7"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_event_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define '_event_name' class variable")


class GuardrailsUsageEvent(TelemetryEvent):
    """Usage event for NeMo Guardrails.

    Emitted at each ``LLMRails`` or ``IORails`` instantiation and as
    periodic heartbeats from a single daemon thread per process. The
    ``Guardrails`` wrapper emits through whichever runtime engine it
    constructs. All events from one process share a session ID. Contains
    no user content, model names, or request-level data.
    """

    _event_name: ClassVar[str] = "guardrails_usage_event"

    nemo_source: NemoSourceEnum = Field(
        default=NemoSourceEnum.GUARDRAILS,
        alias="nemoSource",
        description="The NeMo product that created the event. Always 'guardrails' for this event type.",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        alias="sessionId",
        description=(
            "Random UUID4 generated in memory at process start; acts as the session ID. "
            "Held only for the process lifetime, never persisted to disk. Not traceable "
            "to any user or machine."
        ),
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
    deployment_type: DeploymentTypeEnum = Field(
        default=DeploymentTypeEnum.UNDEFINED,
        alias="deploymentType",
        description=(
            "How guardrails was deployed. 'library' for direct LLMRails / Guardrails use "
            "in user code, 'api' when running the FastAPI server, 'cli' for the interactive "
            "chat command."
        ),
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

    Opt-out signals, any of which disables reporting:

    - ``NEMO_GUARDRAILS_NO_USAGE_STATS=1`` / ``true`` env var
      (product-specific).
    - ``DO_NOT_TRACK=1`` / ``true`` env var (industry-standard).
    - ``~/.config/nemoguardrails/do_not_track`` file present.
    - ``CI`` env var truthy (set by GitHub Actions, GitLab CI, CircleCI,
      Travis, Buildkite, etc.). Suppresses telemetry from automated test
      runs that are not real deployments. Honoring ``CI`` is the same
      convention used by Homebrew, npm, conda, and others.
    - ``PYTEST_CURRENT_TEST`` env var present, or ``pytest`` already
      loaded in the process. Catches the case of a developer running
      tests locally without ``CI=true``, including collection/import
      phases where ``PYTEST_CURRENT_TEST`` is not set, and suppresses
      any telemetry that would otherwise leak from a downstream user's
      test suite that happens to import nemoguardrails.

    The intent is that adoption metrics reflect real deployments only,
    not synthetic test/CI traffic.

    Returns:
        True if reporting should proceed, False if any opt-out is active.
    """
    if os.environ.get("NEMO_GUARDRAILS_NO_USAGE_STATS", "0").lower() in ("1", "true"):
        return False
    if os.environ.get("DO_NOT_TRACK", "0").lower() in ("1", "true"):
        return False
    if os.environ.get("CI", "").lower() in ("1", "true"):
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    try:
        if _get_do_not_track_file().is_file():
            return False
    except Exception:
        log.debug("Failed to check usage telemetry do-not-track file", exc_info=True)
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

_CONFIG_BUILTIN_FEATURE_ALIASES = {
    "fact_checking": "factchecking",
    "patronus": "patronusai",
    "regex_detection": "regex",
}

_COLANG_V2_LIBRARY_DIR = Path(__file__).resolve().parent / "colang" / "v2_x" / "library"


def _normalize_builtin_feature_id(field_name: str) -> str:
    """Return the documented feature id for a RailsConfigData field."""
    return _CONFIG_BUILTIN_FEATURE_ALIASES.get(field_name, field_name)


def _flow_file_name(flow: Any) -> Optional[str]:
    if isinstance(flow, dict):
        file_info = flow.get("file_info", {})
    else:
        file_info = getattr(flow, "file_info", {})

    if isinstance(file_info, dict):
        file_name = file_info.get("name")
        if file_name:
            return str(file_name)
    return None


def _is_v2_library_flow(flow: Any) -> bool:
    file_name = _flow_file_name(flow)
    if not file_name:
        return False

    try:
        Path(file_name).resolve().relative_to(_COLANG_V2_LIBRARY_DIR)
        return True
    except (OSError, ValueError):
        return False


def _is_custom_flow(flow: Any) -> bool:
    if isinstance(flow, dict):
        if flow.get("is_system_flow", False):
            return False
    elif getattr(flow, "is_system_flow", False):
        return False

    return not _is_v2_library_flow(flow)


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
                    features.add(_normalize_builtin_feature_id(field_name))
        except Exception:
            pass

    all_flows = []
    for rail_group in ["input", "output", "retrieval", "tool_output", "tool_input"]:
        group = getattr(rails, rail_group, None)
        if group is not None:
            all_flows.extend(getattr(group, "flows", []))

    for flow_name in all_flows:
        normalized = _normalize_flow_id(flow_name)
        feature = _KNOWN_BUILTIN_FLOWS.get(normalized)
        if feature is not None:
            features.add(feature)

    return sorted(features)


def _collect_usage_data(
    config: Optional["RailsConfig"],
    deployment_type: str,
) -> GuardrailsUsageEvent:
    """Collect anonymous usage data into a ``GuardrailsUsageEvent``.

    Always populates system fields (version, platform, Python version).
    When ``config`` is provided, additionally populates config-derived
    fields: LLM provider names, rail types in use, built-in features,
    custom flow count, and feature flags. Never reads model names,
    prompts, or user content.

    Args:
        config: The ``RailsConfig`` to inspect, or ``None`` for a
            system-only event (e.g. from the server startup context).
        deployment_type: How guardrails was deployed, e.g. ``"library"``,
            ``"api"``, or ``"cli"``. Coerced to
            ``DeploymentTypeEnum.UNDEFINED`` if falsy.

    Returns:
        A fully populated ``GuardrailsUsageEvent``.
    """
    data = GuardrailsUsageEvent()
    data.timestamp = time.time()
    try:
        data.deployment_type = DeploymentTypeEnum(deployment_type or DeploymentTypeEnum.UNDEFINED)
    except (TypeError, ValueError):
        data.deployment_type = DeploymentTypeEnum.UNDEFINED
    data.event = EventTypeEnum.STARTUP
    data.rails_engine = RailsEngineEnum.UNDEFINED

    try:
        data.nemoguardrails_version = _get_version("nemoguardrails")
    except Exception:
        data.nemoguardrails_version = "unknown"

    data.python_version = sys.version.split()[0]
    data.platform = platform.platform()
    data.os_name = platform.system()

    if config is not None:
        data.colang_version = config.colang_version

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
        data.num_custom_flows = sum(1 for f in flows if _is_custom_flow(f))

        tracing = getattr(config, "tracing", None)
        if tracing is not None:
            data.tracing_enabled = getattr(tracing, "enabled", False)

        data.has_knowledge_base = bool(getattr(config, "docs", None))

    return data


def _rotate_audit_file(audit_file: Optional[Path] = None) -> None:
    """Rotate the local audit file when it exceeds the size cap.

    The current ``usage_stats.json`` is renamed to ``usage_stats.json.1``,
    overwriting any previous backup. This bounds on-disk usage at
    approximately ``2 * _AUDIT_FILE_MAX_BYTES``. Errors are silently
    logged at DEBUG level.
    """
    audit_file = audit_file or _get_audit_file()
    backup = audit_file.with_suffix(".json.1")
    try:
        if backup.exists():
            backup.unlink()
        audit_file.rename(backup)
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
        audit_file = _get_audit_file()
        audit_file.parent.mkdir(parents=True, exist_ok=True)

        if audit_file.exists() and audit_file.stat().st_size > _AUDIT_FILE_MAX_BYTES:
            _rotate_audit_file(audit_file)

        with open(audit_file, "a", encoding="utf-8") as f:
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
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts is not None else datetime.now(tz=timezone.utc)
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
    event_timestamps: List[Optional[float]]
    if timestamps is None:
        event_timestamps = [None for _ in events]
    else:
        if len(timestamps) != len(events):
            raise ValueError("timestamps length must match events length")
        event_timestamps = timestamps

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
        "events": [_build_event(event, ts) for event, ts in zip(events, event_timestamps)],
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
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception:
        log.debug("Failed to send usage report", exc_info=True)


def _send_one_event(event: GuardrailsUsageEvent, server_url: str, client_version: str, session_id: str) -> None:
    """Write a single event to the audit file and POST it.

    Used as the target of a one-shot daemon thread spawned by
    ``report_usage``. Fire-and-forget: errors are swallowed at DEBUG
    level inside ``_write_audit_file`` and ``_send_report``.

    Args:
        event: The telemetry event to record and transmit.
        server_url: Full HTTPS URL of the telemetry endpoint.
        client_version: Value to set as ``clientVer`` in the envelope.
        session_id: Value to set as ``sessionId`` in the envelope.
    """
    _write_audit_file(event.model_dump(by_alias=True))
    _send_report(event, server_url, client_version, session_id)


def _heartbeat_loop(
    startup_event: GuardrailsUsageEvent,
    session_id: str,
    client_version: str,
) -> None:
    """Run the heartbeat loop forever in a daemon thread.

    Started exactly once per process (gated by ``_heartbeat_started``).
    Sleeps at least ``_HEARTBEAT_INTERVAL_S`` seconds, then emits a
    heartbeat event tied to the process's session ID. The heartbeat
    reuses a copy of the first startup event's metadata and changes
    only ``event``, ``timestamp``, and ``sessionId``. The thread is a
    daemon so it dies with the main process; no explicit shutdown is
    required.

    Args:
        startup_event: The first startup event emitted by this process.
        session_id: The process-stable session ID, mirrored into the
            heartbeat event's ``session_id`` field and the envelope.
        client_version: Value to set as ``clientVer`` in the envelope.
    """
    while True:
        jitter = random.uniform(0, min(60.0, _HEARTBEAT_INTERVAL_S * 0.1))
        time.sleep(_HEARTBEAT_INTERVAL_S + jitter)
        try:
            heartbeat = startup_event.model_copy(deep=True)
            heartbeat.timestamp = time.time()
            heartbeat.event = EventTypeEnum.HEARTBEAT
            heartbeat.session_id = session_id
            _write_audit_file(heartbeat.model_dump(by_alias=True))
            _send_report(heartbeat, _get_usage_stats_server_url(), client_version, session_id)
        except Exception:
            log.debug("Heartbeat iteration failed; loop continues", exc_info=True)


def set_deployment_type(deployment_type: str) -> None:
    """Set the deployment type that subsequent ``report_usage`` calls will use.

    Called by the FastAPI server lifespan (``"api"``) and the
    ``nemoguardrails chat`` CLI command (``"cli"``) before ``LLMRails``
    is constructed. Takes precedence over the ``deployment_type`` argument
    passed by ``LLMRails.__init__`` and ``Guardrails.__init__``, so the
    same library-internal call sites emit the correct value depending on
    how the process was invoked.

    Invalid values are silently ignored (logged at DEBUG level), so a
    bad caller never crashes the host process.

    Args:
        deployment_type: One of the ``DeploymentTypeEnum`` string values
            (e.g. ``"api"``, ``"cli"``, ``"library"``).
    """
    global _deployment_type_override
    try:
        new_value = DeploymentTypeEnum(deployment_type)
    except ValueError:
        log.debug("Invalid deployment_type %r passed to set_deployment_type", deployment_type)
        return
    with _lock:
        _deployment_type_override = new_value


def _start_daemon_thread(target: Any, args: tuple[Any, ...], failure_message: str) -> bool:
    """Start a daemon thread and report whether startup succeeded."""
    try:
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()
        return True
    except Exception:
        log.debug(failure_message, exc_info=True)
        return False


def report_usage(
    config: Optional["RailsConfig"] = None,
    deployment_type: str = "library",
    rails_engine: str = "",
) -> None:
    """Emit one anonymous usage event for the given config.

    Each call produces a single event tied to the process's session ID
    (lazily generated on first call, reused thereafter). The heartbeat
    daemon thread is started exactly once per process. All work happens
    off the calling thread so this function returns immediately.

    The effective deployment type is resolved as: the value previously
    passed to ``set_deployment_type`` if any, else the function argument.
    This lets the server lifespan or CLI command claim the deployment
    context once, and downstream ``LLMRails.__init__`` /
    ``Guardrails.__init__`` calls inherit it without coordination.

    Respects the triple opt-out (env vars and file).

    Args:
        config: ``RailsConfig`` to introspect. When ``None``, only
            system-level fields are populated.
        deployment_type: How guardrails was deployed (e.g. ``"library"``,
            ``"api"``, ``"cli"``). Overridden by ``set_deployment_type``.
        rails_engine: Which engine class is in use (e.g. ``"LLMRails"``,
            ``"IORails"``). Ignored if empty.
    """
    global _session_uuid, _heartbeat_started

    try:
        if not _is_usage_stats_enabled():
            return

        with _lock:
            effective_deployment_type: str = (
                _deployment_type_override.value if _deployment_type_override is not None else deployment_type
            )

        usage_data = _collect_usage_data(config, effective_deployment_type)
        if rails_engine:
            usage_data.rails_engine = RailsEngineEnum(rails_engine)

        server_url = _get_usage_stats_server_url()

        with _lock:
            if _session_uuid is None:
                _session_uuid = usage_data.session_id
            else:
                usage_data.session_id = _session_uuid
            session_id = _session_uuid

        client_version = usage_data.nemoguardrails_version

        _start_daemon_thread(
            _send_one_event,
            (usage_data, server_url, client_version, session_id),
            "Failed to start usage telemetry send thread",
        )

        with _lock:
            if not _heartbeat_started:
                if _start_daemon_thread(
                    _heartbeat_loop,
                    (usage_data.model_copy(deep=True), session_id, client_version),
                    "Failed to start usage telemetry heartbeat thread",
                ):
                    _heartbeat_started = True
    except Exception:
        log.debug("Usage reporting failed", exc_info=True)
