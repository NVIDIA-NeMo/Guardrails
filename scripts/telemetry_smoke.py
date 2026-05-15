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

"""Local staging smoke-test driver for NeMo Guardrails telemetry.

Drives several deployment scenarios that mirror real production paths:

  - library scenarios construct ``LLMRails`` in a Python subprocess
    (this is what library-mode users actually do).
  - server scenarios start either a real ``python -m nemoguardrails server``
    subprocess or a Uvicorn multi-worker subprocess, poll until it is
    ready, hit ``/v1/chat/completions`` with one or more config_ids to
    trigger ``_get_rails`` (which constructs ``LLMRails`` and emits
    telemetry), then SIGTERM the server.
  - the cli scenario spawns a real ``python -m nemoguardrails chat`` subprocess
    so the typer entry point, ``set_deployment_type('cli')`` call,
    and ``run_chat`` flow are all exercised. The driver holds stdin
    open until the audit file contains the expected startup event, then
    terminates the chat process.
  - opt-out scenarios spawn a minimal subprocess with the relevant
    suppression signal set, and assert the audit file stays empty.

Each scenario runs with an isolated ``HOME`` / ``XDG_CONFIG_HOME`` so
its audit file lands in its own per-scenario directory. The driver
strips ``CI``, ``GITHUB_ACTIONS``, and ``PYTEST_CURRENT_TEST`` from
each subprocess's env so the auto-disable in
``_is_usage_stats_enabled`` does not silently no-op the positive
scenarios. The driver itself refuses to start if any of those are
exported in the calling shell.

Pre-flight (do this once, by hand, before driving the script):

    unset CI GITHUB_ACTIONS PYTEST_CURRENT_TEST
    NEMO_GUARDRAILS_USAGE_STATS_SERVER=<staging-events-url>/v1.1/events/json \\
    poetry run python - <<'PY'
    import json
    import time
    from nemoguardrails import LLMRails, RailsConfig, telemetry

    LLMRails(RailsConfig.from_path("examples/bots/abc"))
    audit_file = telemetry._get_audit_file()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if audit_file.exists() and audit_file.read_text().strip():
            break
        time.sleep(0.1)
    else:
        raise SystemExit(f"pre-flight audit event not observed at {audit_file}")
    time.sleep(20)
    lines = [json.loads(line) for line in audit_file.read_text().splitlines() if line.strip()]
    startup_ids = sorted({line["sessionId"] for line in lines if line.get("event") == "startup"})
    print(f"pre-flight audit event observed at {audit_file}")
    print("kibana filter: client.sessionId : (" + " or ".join(f'"{value}"' for value in startup_ids) + ")")
    PY

Then wait briefly for indexing and search Kibana with the exact printed
``client.sessionId`` filter. If nothing lands, defer the full smoke run until SMS has the schema
registered.

Driver invocation:

    unset CI GITHUB_ACTIONS PYTEST_CURRENT_TEST
    poetry run python scripts/telemetry_smoke.py \\
        --staging-url "$NEMO_GUARDRAILS_SMOKE_STAGING_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "schemas" / "anonymous_events.snapshot.json"
DEFAULT_STAGING_URL = os.environ.get("NEMO_GUARDRAILS_SMOKE_STAGING_URL", "")
DEFAULT_SERVER_CONFIG_ROOT = REPO_ROOT / "tests" / "telemetry" / "smoke_fixtures"
DEFAULT_LIBRARY_CONFIG = REPO_ROOT / "tests" / "telemetry" / "smoke_fixtures" / "cfg1"
DEFAULT_RICH_CONFIG = REPO_ROOT / "tests" / "telemetry" / "smoke_fixtures" / "rich"
DEFAULT_FEATURE_ALIAS_CONFIG = REPO_ROOT / "tests" / "telemetry" / "smoke_fixtures" / "feature_aliases"
DEFAULT_V2_CONFIG = REPO_ROOT / "tests" / "telemetry" / "smoke_fixtures" / "v2_custom_flow"
DEFAULT_IORAILS_CONFIG = REPO_ROOT / "examples" / "configs" / "nemoguards"

ENV_VARS_TO_STRIP = ("CI", "GITHUB_ACTIONS", "PYTEST_CURRENT_TEST")
DEFAULT_AUDIT_TIMEOUT_S = 30.0
DEFAULT_AUDIT_POLL_S = 0.1
DEFAULT_NETWORK_SETTLE_S = 20.0


def _build_subprocess_env(
    parent_env: dict[str, str],
    *,
    staging_url: str,
    config_dir: Path,
    extra: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Build the env for a scenario subprocess.

    Strips CI / pytest signals so ``_is_usage_stats_enabled`` does not
    auto-disable the run. Sets the staging endpoint and an isolated
    XDG_CONFIG_HOME so each scenario's audit file lands in its own
    directory.
    """
    env = {key: value for key, value in parent_env.items() if key not in ENV_VARS_TO_STRIP}
    env["NEMO_GUARDRAILS_USAGE_STATS_SERVER"] = staging_url
    env["XDG_CONFIG_HOME"] = str(config_dir)
    env["HOME"] = str(config_dir)
    if extra:
        env.update(extra)
    return env


def _audit_file_for(config_dir: Path) -> Path:
    return config_dir / ".config" / "nemoguardrails" / "usage_stats.json"


def _read_audit_lines(audit_file: Path) -> list[dict[str, Any]]:
    if not audit_file.exists():
        return []
    return [json.loads(line) for line in audit_file.read_text().splitlines() if line.strip()]


def _wait_for_audit_events(
    audit_file: Path,
    expected_count: int,
    *,
    timeout: float = DEFAULT_AUDIT_TIMEOUT_S,
    poll: float = DEFAULT_AUDIT_POLL_S,
) -> list[dict[str, Any]]:
    """Poll the audit JSONL file until at least ``expected_count`` events exist."""
    deadline = time.monotonic() + timeout
    last_lines: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            last_lines = _read_audit_lines(audit_file)
        except (json.JSONDecodeError, OSError):
            # A writer may be in the middle of appending the last line.
            pass
        if len(last_lines) >= expected_count:
            return last_lines
        time.sleep(poll)

    try:
        return _read_audit_lines(audit_file)
    except (json.JSONDecodeError, OSError):
        return last_lines


def _network_settle_s(scenario: dict[str, Any]) -> float:
    return float(scenario.get("settle_after_audit_s", DEFAULT_NETWORK_SETTLE_S))


def _make_validator():
    import jsonschema

    snapshot = json.loads(SNAPSHOT_PATH.read_text())
    return jsonschema.Draft7Validator(snapshot)


def _validate_lines(lines: list[dict[str, Any]], validator) -> Optional[str]:
    """Return None if all lines validate; otherwise a short error string."""
    for index, line in enumerate(lines):
        errors = list(validator.iter_errors(line))
        if errors:
            return f"line {index} fails schema: {errors[0].message}"
    return None


@dataclass(frozen=True)
class FieldExpectation:
    description: str
    predicate: Callable[[Any], bool]


def _expect_exact(expected: Any) -> FieldExpectation:
    return FieldExpectation(repr(expected), lambda actual: actual == expected)


def _expect_one_of(values: list[str]) -> FieldExpectation:
    return FieldExpectation(f"one of {values!r}", lambda actual: actual in values)


def _expect_non_empty_string() -> FieldExpectation:
    return FieldExpectation("a non-empty string", lambda actual: isinstance(actual, str) and bool(actual))


def _expect_positive_number() -> FieldExpectation:
    return FieldExpectation(
        "a positive number",
        lambda actual: isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual > 0,
    )


def _expect_list_contains(items: list[str]) -> FieldExpectation:
    return FieldExpectation(
        f"a list containing {items!r}",
        lambda actual: isinstance(actual, list) and all(item in actual for item in items),
    )


def _expect_int_at_least(minimum: int) -> FieldExpectation:
    return FieldExpectation(
        f"an integer >= {minimum}",
        lambda actual: isinstance(actual, int) and not isinstance(actual, bool) and actual >= minimum,
    )


def _as_expectation(expected: Any) -> FieldExpectation:
    if isinstance(expected, FieldExpectation):
        return expected
    return _expect_exact(expected)


def _validate_field(
    scenario_name: str,
    field: str,
    actual: Any,
    expected: Any,
) -> Optional[str]:
    expectation = _as_expectation(expected)
    if not expectation.predicate(actual):
        return f"{scenario_name}.{field}: expected {expectation.description}, got {actual!r}"
    return None


def _common_event_assertions() -> dict[str, FieldExpectation]:
    return {
        "nemoSource": _expect_exact("guardrails"),
        "event": _expect_one_of(["startup", "heartbeat"]),
        "sessionId": _expect_non_empty_string(),
        "nemoguardrailsVersion": _expect_non_empty_string(),
        "pythonVersion": _expect_non_empty_string(),
        "platform": _expect_non_empty_string(),
        "osName": _expect_non_empty_string(),
        "timestamp": _expect_positive_number(),
    }


def _validate_common_event_fields(
    scenario_name: str,
    event: dict[str, Any],
) -> Optional[str]:
    for field, expectation in _common_event_assertions().items():
        error = _validate_field(scenario_name, field, event.get(field), expectation)
        if error:
            return error
    return None


def _validate_startup_events(
    scenario_name: str,
    startup_events: list[dict[str, Any]],
    *,
    assertion_sets: list[dict[str, Any]],
) -> Optional[str]:
    """Validate startup events with exact, predicate, and list assertions."""
    if len(startup_events) != len(assertion_sets):
        return f"{scenario_name}.event: expected {len(assertion_sets)} startup event(s), got {len(startup_events)}"

    for event in startup_events:
        common_error = _validate_common_event_fields(scenario_name, event)
        if common_error:
            return common_error

        startup_error = _validate_field(scenario_name, "event", event.get("event"), "startup")
        if startup_error:
            return startup_error

    unmatched_assertions = list(assertion_sets)
    for event in startup_events:
        matched_index: Optional[int] = None
        first_error: Optional[str] = None
        for index, assertions in enumerate(unmatched_assertions):
            errors = [
                _validate_field(scenario_name, field, event.get(field), expected)
                for field, expected in assertions.items()
            ]
            errors = [error for error in errors if error]
            if not errors:
                matched_index = index
                break
            if first_error is None:
                first_error = errors[0]

        if matched_index is None:
            return first_error or f"{scenario_name}.startup: no assertion set matched {event!r}"
        unmatched_assertions.pop(matched_index)

    return None


def _summarize_expectation(expected: Any) -> str:
    return _as_expectation(expected).description


def _summarize_assertions(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "common_event_fields": {
            field: _summarize_expectation(expectation) for field, expectation in _common_event_assertions().items()
        },
        "startup_events": [
            {field: _summarize_expectation(expected) for field, expected in assertions.items()}
            for assertions in scenario.get("startup_assertions", [])
        ],
        "distinct_startup_sessions": scenario.get("expects_distinct_sessions", 1),
    }


def _free_port() -> int:
    """Ask the OS for a free TCP port and release it.

    Small race window between release and reuse by the server, but
    acceptable for a smoke driver.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, *, timeout: float = 30.0, poll: float = 0.5) -> bool:
    """Poll ``/v1/rails/configs`` until 200 or timeout."""
    deadline = time.monotonic() + timeout
    url = base_url.rstrip("/") + "/v1/rails/configs"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError):
            pass
        time.sleep(poll)
    return False


def _post_chat(base_url: str, config_id: str, *, timeout: float = 10.0) -> tuple[int, str]:
    """POST to /v1/chat/completions. Returns (status, body preview).

    The long thread_id passes request validation but fails after
    ``_get_rails`` when no datastore is configured, so construction
    emits telemetry without requiring or making a real model call.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "smoke"}],
        "guardrails": {
            "config_ids": [config_id],
            "options": {},
            "thread_id": "smoke-thread-0001",
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(500).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_preview = exc.read(500).decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body_preview
    except urllib.error.URLError as exc:
        return -1, f"URLError: {exc.reason}"


def _terminate(proc: subprocess.Popen, *, term_timeout: float = 10.0) -> int:
    """SIGTERM then SIGKILL fallback. Returns the final exit code."""
    if proc.poll() is not None:
        return int(proc.returncode)
    proc.terminate()
    try:
        returncode = proc.wait(timeout=term_timeout)
        return 0 if returncode in {-15, 1} else returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return -1


def _run_subprocess(scenario: dict[str, Any], env: dict[str, str], audit_file: Path) -> dict[str, Any]:
    """Run a ``kind=subprocess`` scenario: a Python -c script.

    Used for library and opt-out scenarios where we directly construct
    rails from a config or verify no event is emitted (opt-out).
    """
    started_at = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", scenario["script"]],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        return {"returncode": -1, "duration_s": time.time() - started_at, "stderr_tail": [repr(exc)]}

    expected_count = int(scenario["expected_count"])
    audit_timeout = float(scenario.get("audit_timeout_s", DEFAULT_AUDIT_TIMEOUT_S))
    try:
        if expected_count > 0:
            _wait_for_audit_events(audit_file, expected_count, timeout=audit_timeout)
            settle = _network_settle_s(scenario)
            if settle:
                time.sleep(settle)
        else:
            time.sleep(float(scenario.get("settle_after_action_s", 1.0)))

        returncode = _terminate(proc)
    finally:
        if proc.poll() is None:
            returncode = _terminate(proc)

    stderr_tail: list[str] = []
    if proc.stderr is not None:
        try:
            stderr_tail = proc.stderr.read().splitlines()[-10:]
        except Exception:
            pass
    return {
        "returncode": returncode,
        "duration_s": time.time() - started_at,
        "stderr_tail": stderr_tail,
    }


def _startup_session_ids(lines: list[dict[str, Any]]) -> set[str]:
    return {line["sessionId"] for line in lines if line.get("event") == "startup" and line.get("sessionId")}


def _format_kibana_filter(session_ids: list[str]) -> str:
    if not session_ids:
        return "client.sessionId : (no startup session IDs collected)"
    quoted = [json.dumps(session_id) for session_id in session_ids]
    return "client.sessionId : (" + " or ".join(quoted) + ")"


def _wait_for_distinct_startup_sessions(
    audit_file: Path,
    expected_count: int,
    *,
    timeout: float = DEFAULT_AUDIT_TIMEOUT_S,
    poll: float = DEFAULT_AUDIT_POLL_S,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    last_lines: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            last_lines = _read_audit_lines(audit_file)
        except (json.JSONDecodeError, OSError):
            pass
        if len(_startup_session_ids(last_lines)) >= expected_count:
            return last_lines
        time.sleep(poll)

    try:
        return _read_audit_lines(audit_file)
    except (json.JSONDecodeError, OSError):
        return last_lines


def _run_server(scenario: dict[str, Any], env: dict[str, str], audit_file: Path) -> dict[str, Any]:
    """Run a ``kind=server`` scenario: spin up ``python -m nemoguardrails server``.

    Polls /v1/rails/configs until ready, POSTs to /v1/chat/completions
    for each config_id in scenario["config_ids_to_hit"], polls the
    audit file until the expected event count is observed, then SIGTERMs.
    """
    started_at = time.time()
    post_results: list[dict[str, Any]] = []
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "nemoguardrails",
                "server",
                "--config",
                scenario["server_config_root"],
                "--port",
                str(port),
                "--disable-chat-ui",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        return {
            "returncode": -1,
            "duration_s": time.time() - started_at,
            "stderr_tail": [repr(exc)],
            "server_post_results": post_results,
        }
    try:
        if not _wait_for_server(base_url, timeout=30):
            _terminate(proc)
            stderr_tail = proc.stderr.read().splitlines()[-10:] if proc.stderr else []
            return {
                "returncode": -1,
                "duration_s": time.time() - started_at,
                "stderr_tail": stderr_tail,
                "server_post_results": post_results,
                "early_failure": "server did not become ready within 30s",
            }
        audit_timeout = float(scenario.get("audit_timeout_s", DEFAULT_AUDIT_TIMEOUT_S))
        config_ids = scenario["config_ids_to_hit"]
        successful_posts = 0
        for index, config_id in enumerate(config_ids, start=1):
            status, body_preview = _post_chat(base_url, config_id)
            post_results.append(
                {
                    "config_id": config_id,
                    "status": status,
                    "body_preview": body_preview[:500],
                }
            )
            if status != -1:
                successful_posts += 1
                lines = _wait_for_audit_events(audit_file, successful_posts, timeout=audit_timeout)
                if len(lines) >= successful_posts and index < len(config_ids):
                    settle = _network_settle_s(scenario)
                    if settle:
                        time.sleep(settle)

        unreachable = [result for result in post_results if result["status"] == -1]
        if not unreachable:
            _wait_for_audit_events(
                audit_file,
                int(scenario["expected_count"]),
                timeout=audit_timeout,
            )
            settle = _network_settle_s(scenario)
            if settle:
                time.sleep(settle)
    finally:
        rc = _terminate(proc)
    # Drain stderr so any warnings show up on FAIL.
    stderr_tail: list[str] = []
    if proc.stderr is not None:
        try:
            stderr_tail = proc.stderr.read().splitlines()[-10:]
        except Exception:
            pass
    return {
        "returncode": rc,
        "duration_s": time.time() - started_at,
        "stderr_tail": stderr_tail,
        "server_post_results": post_results,
        "post_reachability_failure": any(result["status"] == -1 for result in post_results),
    }


def _write_uvicorn_smoke_app(module_dir: Path) -> str:
    """Create an importable FastAPI app module for Uvicorn worker smoke."""
    module_name = "telemetry_smoke_uvicorn_app"
    module_path = module_dir / f"{module_name}.py"
    module_path.write_text(
        """\
import os

os.environ["NEMO_GUARDRAILS_DISABLE_CHAT_UI"] = "true"

from nemoguardrails.server import api
from nemoguardrails.telemetry import set_deployment_type

set_deployment_type("api")
api.app.rails_config_path = os.environ["NEMO_GUARDRAILS_SMOKE_SERVER_CONFIG_ROOT"]
api.app.disable_chat_ui = True
app = api.app
"""
    )
    return f"{module_name}:app"


def _run_server_multi_worker(scenario: dict[str, Any], env: dict[str, str], audit_file: Path) -> dict[str, Any]:
    """Run a Uvicorn multi-worker server and require one event per worker."""
    started_at = time.time()
    post_results: list[dict[str, Any]] = []
    workers = int(scenario.get("worker_count", 3))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    module_dir = audit_file.parent.parent.parent
    app_import = _write_uvicorn_smoke_app(module_dir)
    proc_env = env.copy()
    proc_env["NEMO_GUARDRAILS_SMOKE_SERVER_CONFIG_ROOT"] = scenario["server_config_root"]
    proc_env["NEMO_GUARDRAILS_DISABLE_CHAT_UI"] = "true"
    pythonpath_parts = [str(module_dir), str(REPO_ROOT)]
    if proc_env.get("PYTHONPATH"):
        pythonpath_parts.append(proc_env["PYTHONPATH"])
    proc_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                app_import,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                str(workers),
                "--log-level",
                "info",
            ],
            env=proc_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        return {
            "returncode": -1,
            "duration_s": time.time() - started_at,
            "stderr_tail": [repr(exc)],
            "server_post_results": post_results,
            "worker_count": workers,
        }
    try:
        if not _wait_for_server(base_url, timeout=45):
            _terminate(proc)
            stderr_tail = proc.stderr.read().splitlines()[-10:] if proc.stderr else []
            return {
                "returncode": -1,
                "duration_s": time.time() - started_at,
                "stderr_tail": stderr_tail,
                "server_post_results": post_results,
                "worker_count": workers,
                "early_failure": "multi-worker server did not become ready within 45s",
            }

        config_id = scenario["config_ids_to_hit"][0]
        audit_timeout = float(scenario.get("audit_timeout_s", DEFAULT_AUDIT_TIMEOUT_S))
        expected_sessions = int(scenario.get("expects_distinct_sessions", workers))
        max_posts = int(scenario.get("max_worker_posts", workers * 30))
        deadline = time.monotonic() + audit_timeout
        lines: list[dict[str, Any]] = []
        for _ in range(max_posts):
            status, body_preview = _post_chat(base_url, config_id)
            post_results.append(
                {
                    "config_id": config_id,
                    "status": status,
                    "body_preview": body_preview[:500],
                }
            )
            lines = _wait_for_distinct_startup_sessions(audit_file, expected_sessions, timeout=0.5)
            if len(_startup_session_ids(lines)) >= expected_sessions:
                break
            if time.monotonic() >= deadline:
                break

        if not any(result["status"] == -1 for result in post_results):
            remaining = max(0.0, deadline - time.monotonic())
            if len(_startup_session_ids(lines)) < expected_sessions and remaining:
                lines = _wait_for_distinct_startup_sessions(audit_file, expected_sessions, timeout=remaining)
            if len(_startup_session_ids(lines)) >= expected_sessions:
                settle = _network_settle_s(scenario)
                if settle:
                    time.sleep(settle)
    finally:
        rc = _terminate(proc)

    stderr_tail: list[str] = []
    if proc.stderr is not None:
        try:
            stderr_tail = proc.stderr.read().splitlines()[-10:]
        except Exception:
            pass
    return {
        "returncode": rc,
        "duration_s": time.time() - started_at,
        "stderr_tail": stderr_tail,
        "server_post_results": post_results,
        "worker_count": workers,
        "post_reachability_failure": any(result["status"] == -1 for result in post_results),
    }


def _run_cli(scenario: dict[str, Any], env: dict[str, str], audit_file: Path) -> dict[str, Any]:
    """Run a ``kind=cli`` scenario: spawn ``python -m nemoguardrails chat``.

    Holds stdin open so ``LLMRails`` is constructed and the chat loop
    blocks on input. The driver polls the audit file, then terminates.
    """
    started_at = time.time()
    # Leave stdin open (PIPE but never written to) so the chat command's
    # input("> ") at cli/chat.py:76 blocks forever. LLMRails is
    # constructed at line 68 *before* that input call, so telemetry
    # fires during the wait. Then we SIGTERM. Closing stdin instead
    # would feed EOF to input(), raising EOFError and aborting before
    # the daemon thread can flush the audit file reliably.
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "nemoguardrails", "chat", "--config", scenario["config_path"]],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(REPO_ROOT),
        )
    except Exception as exc:
        return {"returncode": -1, "duration_s": time.time() - started_at, "stderr_tail": [repr(exc)]}
    try:
        _wait_for_audit_events(
            audit_file,
            int(scenario["expected_count"]),
            timeout=float(scenario.get("audit_timeout_s", DEFAULT_AUDIT_TIMEOUT_S)),
        )
        settle = _network_settle_s(scenario)
        if settle:
            time.sleep(settle)
    finally:
        rc = _terminate(proc)
    stderr_tail: list[str] = []
    if proc.stderr is not None:
        try:
            stderr_tail = proc.stderr.read().splitlines()[-10:]
        except Exception:
            pass
    return {
        "returncode": rc,
        "duration_s": time.time() - started_at,
        "stderr_tail": stderr_tail,
    }


_RUNNERS = {
    "subprocess": _run_subprocess,
    "server": _run_server,
    "server_multi_worker": _run_server_multi_worker,
    "cli": _run_cli,
}


def _script_construct_rails(config_path: str) -> str:
    """Construct rails from a config and keep the subprocess alive.

    Scenario-specific behavior is controlled by the config path and
    environment: opt-out scenarios suppress telemetry via env vars, and
    the IORails scenario sets NEMO_GUARDRAILS_IORAILS_ENGINE=true.
    """
    return f"""
import time
from nemoguardrails import LLMRails, RailsConfig
LLMRails(RailsConfig.from_path({config_path!r}))
while True:
    time.sleep(3600)
"""


def _scenario(
    *,
    name: str,
    kind: str,
    expected_count: int,
    script: Optional[str] = None,
    server_config_root: Optional[str] = None,
    config_ids_to_hit: Optional[list[str]] = None,
    worker_count: Optional[int] = None,
    config_path: Optional[str] = None,
    startup_assertions: Optional[list[dict[str, Any]]] = None,
    audit_timeout_s: Optional[float] = None,
    settle_after_audit_s: Optional[float] = None,
    settle_after_action_s: Optional[float] = None,
    extra_env: Optional[dict[str, str]] = None,
    expects_distinct_sessions: int = 1,
) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "expected_count": expected_count,
        "startup_assertions": startup_assertions or [],
        "extra_env": extra_env or {},
        "expects_distinct_sessions": expects_distinct_sessions,
    }
    if script is not None:
        scenario["script"] = script
    if server_config_root is not None:
        scenario["server_config_root"] = server_config_root
    if config_ids_to_hit is not None:
        scenario["config_ids_to_hit"] = config_ids_to_hit
    if worker_count is not None:
        scenario["worker_count"] = worker_count
    if config_path is not None:
        scenario["config_path"] = config_path
    if audit_timeout_s is not None:
        scenario["audit_timeout_s"] = audit_timeout_s
    if settle_after_audit_s is not None:
        scenario["settle_after_audit_s"] = settle_after_audit_s
    if settle_after_action_s is not None:
        scenario["settle_after_action_s"] = settle_after_action_s
    return scenario


def _config_startup_assertions(
    *,
    deployment_type: str,
    rails_engine: str,
    llm_providers: list[str],
    colang_version: str = "1.0",
    num_rails_configured: int,
    rail_types_in_use: list[str],
    builtin_features: Any,
    tracing_enabled: bool = False,
    has_knowledge_base: bool = False,
    streaming_configured: bool = False,
    num_custom_flows: Any = 0,
) -> dict[str, Any]:
    return {
        "deploymentType": deployment_type,
        "railsEngine": rails_engine,
        "llmProviders": llm_providers,
        "colangVersion": colang_version,
        "numRailsConfigured": num_rails_configured,
        "railTypesInUse": rail_types_in_use,
        "builtinFeatures": builtin_features,
        "tracingEnabled": tracing_enabled,
        "hasKnowledgeBase": has_knowledge_base,
        "streamingConfigured": streaming_configured,
        "numCustomFlows": num_custom_flows,
    }


def _cfg1_assertions(deployment_type: str, rails_engine: str = "LLMRails") -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type=deployment_type,
        rails_engine=rails_engine,
        llm_providers=["openai"],
        num_rails_configured=1,
        rail_types_in_use=["input"],
        builtin_features=_expect_list_contains(["self_check"]),
    )


def _cfg2_assertions(deployment_type: str, rails_engine: str = "LLMRails") -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type=deployment_type,
        rails_engine=rails_engine,
        llm_providers=["openai"],
        num_rails_configured=1,
        rail_types_in_use=["output"],
        builtin_features=_expect_list_contains(["self_check"]),
    )


def _cfg3_assertions(deployment_type: str, rails_engine: str = "LLMRails") -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type=deployment_type,
        rails_engine=rails_engine,
        llm_providers=["openai"],
        num_rails_configured=2,
        rail_types_in_use=["input", "output"],
        builtin_features=_expect_list_contains(["self_check"]),
    )


def _rich_assertions() -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type="library",
        rails_engine="LLMRails",
        llm_providers=["openai"],
        num_rails_configured=2,
        rail_types_in_use=["input", "output"],
        builtin_features=_expect_list_contains(["self_check"]),
        tracing_enabled=True,
        has_knowledge_base=True,
        streaming_configured=True,
        num_custom_flows=_expect_int_at_least(1),
    )


def _feature_alias_assertions() -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type="library",
        rails_engine="LLMRails",
        llm_providers=["openai"],
        num_rails_configured=0,
        rail_types_in_use=[],
        builtin_features=["factchecking", "patronusai", "regex"],
    )


def _v2_custom_flow_assertions() -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type="library",
        rails_engine="LLMRails",
        llm_providers=["openai"],
        colang_version="2.x",
        num_rails_configured=0,
        rail_types_in_use=[],
        builtin_features=[],
        num_custom_flows=1,
    )


def _iorails_assertions() -> dict[str, Any]:
    return _config_startup_assertions(
        deployment_type="library",
        rails_engine="IORails",
        llm_providers=["nim"],
        num_rails_configured=4,
        rail_types_in_use=["input", "output"],
        builtin_features=_expect_list_contains(["content_safety", "jailbreak_detection", "topic_safety"]),
    )


def _build_scenarios(
    *,
    library_config: str,
    rich_config: str,
    feature_alias_config: str,
    v2_config: str,
    iorails_config: str,
    server_config_root: str,
) -> list[dict[str, Any]]:
    return [
        _scenario(
            name="library_llmrails",
            kind="subprocess",
            script=_script_construct_rails(library_config),
            expected_count=1,
            startup_assertions=[_cfg1_assertions("library")],
        ),
        _scenario(
            name="library_rich_config",
            kind="subprocess",
            script=_script_construct_rails(rich_config),
            expected_count=1,
            startup_assertions=[_rich_assertions()],
        ),
        _scenario(
            name="library_feature_aliases",
            kind="subprocess",
            script=_script_construct_rails(feature_alias_config),
            expected_count=1,
            startup_assertions=[_feature_alias_assertions()],
        ),
        _scenario(
            name="library_v2_custom_flows",
            kind="subprocess",
            script=_script_construct_rails(v2_config),
            expected_count=1,
            startup_assertions=[_v2_custom_flow_assertions()],
        ),
        _scenario(
            name="library_iorails",
            kind="subprocess",
            script=_script_construct_rails(iorails_config),
            expected_count=1,
            startup_assertions=[_iorails_assertions()],
            extra_env={"NEMO_GUARDRAILS_IORAILS_ENGINE": "true"},
        ),
        _scenario(
            name="server_single_config",
            kind="server",
            server_config_root=server_config_root,
            config_ids_to_hit=["cfg1"],
            expected_count=1,
            startup_assertions=[_cfg1_assertions("api")],
        ),
        _scenario(
            name="server_multi_config",
            kind="server",
            server_config_root=server_config_root,
            config_ids_to_hit=["cfg1", "cfg2", "cfg3"],
            expected_count=3,
            startup_assertions=[_cfg1_assertions("api"), _cfg2_assertions("api"), _cfg3_assertions("api")],
        ),
        _scenario(
            name="server_multi_worker",
            kind="server_multi_worker",
            server_config_root=server_config_root,
            config_ids_to_hit=["cfg1"],
            worker_count=3,
            expected_count=3,
            startup_assertions=[_cfg1_assertions("api"), _cfg1_assertions("api"), _cfg1_assertions("api")],
            audit_timeout_s=45.0,
            expects_distinct_sessions=3,
        ),
        _scenario(
            name="cli_chat",
            kind="cli",
            config_path=library_config,
            expected_count=1,
            startup_assertions=[_cfg1_assertions("cli")],
        ),
        _scenario(
            name="heartbeat",
            kind="server",
            server_config_root=server_config_root,
            config_ids_to_hit=["cfg1"],
            expected_count=2,  # one startup + one heartbeat
            startup_assertions=[_cfg1_assertions("api")],
            audit_timeout_s=20.0,
            extra_env={"NEMO_GUARDRAILS_HEARTBEAT_INTERVAL_S": "8.0"},
        ),
        _scenario(
            name="opt_out_explicit",
            kind="subprocess",
            script=_script_construct_rails(library_config),
            expected_count=0,
            settle_after_action_s=1.0,
            extra_env={"NEMO_GUARDRAILS_NO_USAGE_STATS": "1"},
        ),
        _scenario(
            name="opt_out_ci",
            kind="subprocess",
            script=_script_construct_rails(library_config),
            expected_count=0,
            settle_after_action_s=1.0,
            extra_env={"CI": "true"},
        ),
        _scenario(
            name="opt_out_pytest",
            kind="subprocess",
            script=_script_construct_rails(library_config),
            expected_count=0,
            settle_after_action_s=1.0,
            extra_env={"PYTEST_CURRENT_TEST": "smoke_x"},
        ),
    ]


def _sanity_check_driver_env() -> None:
    leaked = [name for name in ENV_VARS_TO_STRIP if os.environ.get(name)]
    if leaked:
        sys.stderr.write(
            "refusing to run: the driver inherited " + ", ".join(leaked) + " in its environment.\n"
            "  unset " + " ".join(leaked) + "\n"
            "and re-invoke. The driver strips these from each subprocess, but if they "
            "are exported in the calling shell some scenarios may behave unexpectedly.\n"
        )
        sys.exit(2)


def _run_scenario(
    scenario: dict[str, Any],
    *,
    run_id: str,
    staging_url: str,
    run_dir: Path,
    validator,
) -> dict[str, Any]:
    config_dir = run_dir / scenario["name"]
    if config_dir.exists():
        shutil.rmtree(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    env = _build_subprocess_env(
        parent_env=os.environ.copy(),
        staging_url=staging_url,
        config_dir=config_dir,
        extra=scenario["extra_env"],
    )

    runner = _RUNNERS[scenario["kind"]]
    audit_file = _audit_file_for(config_dir)
    run_outcome = runner(scenario, env, audit_file)
    try:
        lines = _read_audit_lines(audit_file)
    except json.JSONDecodeError as exc:
        lines = []
        audit_error = f"audit file is not valid JSONL: {exc}"
    else:
        audit_error = ""
    startup_session_ids = sorted(_startup_session_ids(lines))

    result: dict[str, Any] = {
        "name": scenario["name"],
        "kind": scenario["kind"],
        "startup_session_ids": startup_session_ids,
        "audit_file": str(audit_file),
        "subprocess_returncode": run_outcome["returncode"],
        "subprocess_duration_s": round(run_outcome["duration_s"], 2),
        "stderr_tail": run_outcome.get("stderr_tail", []),
        "server_post_results": run_outcome.get("server_post_results", []),
        "expected_event_count": scenario["expected_count"],
        "actual_event_count": len(lines),
        "expected_assertions": _summarize_assertions(scenario),
    }
    if "worker_count" in run_outcome:
        result["worker_count"] = run_outcome["worker_count"]

    if audit_error:
        result["verdict"] = "FAIL"
        result["reason"] = audit_error
        return result

    early = run_outcome.get("early_failure")
    if early:
        result["verdict"] = "FAIL"
        result["reason"] = early
        return result

    if run_outcome.get("post_reachability_failure"):
        result["verdict"] = "FAIL"
        result["reason"] = "server POST could not reach the local server"
        return result

    if run_outcome["returncode"] != 0:
        result["verdict"] = "FAIL"
        result["reason"] = f"subprocess exited with code {run_outcome['returncode']}"
        return result

    if scenario["expected_count"] == 0:
        result["verdict"] = "PASS" if not lines else "FAIL"
        if lines:
            result["reason"] = f"expected 0 events but found {len(lines)}"
        return result

    if len(lines) < scenario["expected_count"]:
        result["verdict"] = "FAIL"
        result["reason"] = f"expected {scenario['expected_count']} events, got {len(lines)}"
        return result

    schema_error = _validate_lines(lines, validator)
    if schema_error:
        result["verdict"] = "FAIL"
        result["reason"] = schema_error
        return result

    for line in lines:
        common_error = _validate_common_event_fields(scenario["name"], line)
        if common_error:
            result["verdict"] = "FAIL"
            result["reason"] = common_error
            return result

    startup_lines = [line for line in lines if line.get("event") == "startup"]
    startup_error = _validate_startup_events(
        scenario["name"],
        startup_lines,
        assertion_sets=scenario["startup_assertions"],
    )
    if startup_error:
        result["verdict"] = "FAIL"
        result["reason"] = startup_error
        return result

    distinct_sessions = {line.get("sessionId") for line in startup_lines}
    expected_distinct_sessions = scenario.get("expects_distinct_sessions", 1)
    if len(distinct_sessions) != expected_distinct_sessions:
        result["verdict"] = "FAIL"
        result["reason"] = (
            f"distinct startup session IDs: expected {expected_distinct_sessions}, got {len(distinct_sessions)}"
        )
        return result

    if scenario["name"] == "heartbeat":
        heartbeat_lines = [line for line in lines if line.get("event") == "heartbeat"]
        if not heartbeat_lines:
            result["verdict"] = "FAIL"
            result["reason"] = "no heartbeat event observed in audit file"
            return result
        startup_session_ids = {line.get("sessionId") for line in startup_lines}
        heartbeat_session_ids = {line.get("sessionId") for line in heartbeat_lines}
        if startup_session_ids != heartbeat_session_ids:
            result["verdict"] = "FAIL"
            result["reason"] = (
                "heartbeat session IDs did not match startup session IDs: "
                f"startup={sorted(startup_session_ids)}, heartbeat={sorted(heartbeat_session_ids)}"
            )
            return result

    result["verdict"] = "PASS"
    return result


def _generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{timestamp}-{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-url",
        default=DEFAULT_STAGING_URL,
        help=(
            "Staging telemetry endpoint URL. If omitted, reads "
            "NEMO_GUARDRAILS_SMOKE_STAGING_URL from the operator's environment."
        ),
    )
    parser.add_argument(
        "--library-config",
        default=str(DEFAULT_LIBRARY_CONFIG),
        help=(
            f"Path to the RailsConfig used by library, cli, and opt-out scenarios. Default: {DEFAULT_LIBRARY_CONFIG}."
        ),
    )
    parser.add_argument(
        "--rich-config",
        default=str(DEFAULT_RICH_CONFIG),
        help=(
            f"Path to the richer RailsConfig used by the library_rich_config scenario. Default: {DEFAULT_RICH_CONFIG}."
        ),
    )
    parser.add_argument(
        "--feature-alias-config",
        default=str(DEFAULT_FEATURE_ALIAS_CONFIG),
        help=(
            "Path to the RailsConfig used by the library_feature_aliases scenario. "
            f"Default: {DEFAULT_FEATURE_ALIAS_CONFIG}."
        ),
    )
    parser.add_argument(
        "--v2-config",
        default=str(DEFAULT_V2_CONFIG),
        help=(
            "Path to the Colang 2.x RailsConfig used by the library_v2_custom_flows scenario. "
            f"Default: {DEFAULT_V2_CONFIG}."
        ),
    )
    parser.add_argument(
        "--iorails-config",
        default=str(DEFAULT_IORAILS_CONFIG),
        help=(
            "Path to a RailsConfig whose flows are all IORails-compatible. "
            "Used by library_iorails (subprocess sets NEMO_GUARDRAILS_IORAILS_ENGINE=true). "
            f"Default: {DEFAULT_IORAILS_CONFIG}."
        ),
    )
    parser.add_argument(
        "--server-config-root",
        default=str(DEFAULT_SERVER_CONFIG_ROOT),
        help=(
            "Path to the directory containing per-config subdirectories the server "
            f"will list (parent of cfg1, cfg2, cfg3, rich). Default: {DEFAULT_SERVER_CONFIG_ROOT}."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Directory to write per-scenario audit files and manifest.json into. Defaults to a tempdir.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Run only the named scenario(s). Repeatable. Default: all scenarios.",
    )
    args = parser.parse_args()

    _sanity_check_driver_env()
    if not args.staging_url:
        sys.stderr.write(
            "missing staging endpoint: pass --staging-url or set "
            "NEMO_GUARDRAILS_SMOKE_STAGING_URL in your local environment.\n"
            "Do not commit internal staging telemetry URLs to the repository.\n"
        )
        return 2

    run_id = _generate_run_id()
    run_dir = Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix=f"smoke-{run_id}-"))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    print(f"smoke run: {run_id}")
    print(f"  staging:            {args.staging_url}")
    print(f"  run dir:            {run_dir}")
    print(f"  library config:     {args.library_config}")
    print(f"  rich config:        {args.rich_config}")
    print(f"  feature aliases:    {args.feature_alias_config}")
    print(f"  v2 config:          {args.v2_config}")
    print(f"  iorails config:     {args.iorails_config}")
    print(f"  server config root: {args.server_config_root}")
    print()

    validator = _make_validator()

    all_scenarios = _build_scenarios(
        library_config=args.library_config,
        rich_config=args.rich_config,
        feature_alias_config=args.feature_alias_config,
        v2_config=args.v2_config,
        iorails_config=args.iorails_config,
        server_config_root=args.server_config_root,
    )
    if args.scenario:
        selected_names = set(args.scenario)
        scenarios = [scenario for scenario in all_scenarios if scenario["name"] in selected_names]
        unknown = selected_names - {scenario["name"] for scenario in all_scenarios}
        if unknown:
            sys.stderr.write(f"unknown scenario(s): {sorted(unknown)}\n")
            return 2
    else:
        scenarios = all_scenarios

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        print(f"[{scenario['name']}] running ({scenario['kind']})...", flush=True)
        result = _run_scenario(
            scenario,
            run_id=run_id,
            staging_url=args.staging_url,
            run_dir=run_dir,
            validator=validator,
        )
        verdict = result["verdict"]
        reason = result.get("reason", "")
        suffix = f" ({reason})" if reason else ""
        print(f"[{result['name']}] {verdict}{suffix}")
        results.append(result)

    all_startup_session_ids = sorted(
        {session_id for result in results for session_id in result.get("startup_session_ids", [])}
    )
    kibana_filter = _format_kibana_filter(all_startup_session_ids)
    manifest = {
        "run_id": run_id,
        "staging_url": args.staging_url,
        "run_dir": str(run_dir),
        "startup_session_ids": all_startup_session_ids,
        "kibana_filter": kibana_filter,
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print("=== summary ===")
    for result in results:
        print(f"  {result['name']:24s} {result['verdict']}")
    print()
    print(f"manifest: {manifest_path}")
    print(f"kibana filter: {kibana_filter}")
    print(
        "offline verify: poetry run python scripts/kibana_verify_export.py "
        f"--manifest {manifest_path} --export kibana.json"
    )

    return 0 if all(result["verdict"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
