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

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from nemoguardrails import RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.rail_outcome import RailOutcome
from nemoguardrails.context import get_conversation_id
from nemoguardrails.guardrails.tool_schema import Tool, ToolResult, tool_call_validation
from nemoguardrails.http import (
    ClosableHTTPClient,
    HTTPClient,
    HTTPClientError,
    HTTPResponseDecodeError,
    RetryPolicy,
    create_http_client,
    http_call,
)
from nemoguardrails.http.composition import ensure_retrying_http_client
from nemoguardrails.library.hiddenlayer._token import (
    REQUEST_TIMEOUT_SECONDS,
    _access_token,
    _describe,
)
from nemoguardrails.library.hiddenlayer._verdict import (
    BLOCK_REASON,
    FAIL_OPEN_REASON,
    Surface,
    _Evaluation,
    _outcome,
)
from nemoguardrails.library.hiddenlayer.errs import (
    HiddenLayerActionError,
    HiddenLayerCheckError,
    HiddenLayerTransportError,
    HiddenLayerUnknownActionError,
    HiddenLayerVerdictError,
)
from nemoguardrails.library.hiddenlayer.rail_config import (
    DEFAULT_API_URL,
    DEFAULT_REQUESTER_ID,
    HiddenLayerRailConfig,
    _check_api_url,
)
from nemoguardrails.library.hiddenlayer.request import (
    _SURFACE_PATH,
    _headers,
    _input_check,
    _output_check,
    _roundtrip_id,
    _tool_call_check,
    _tool_result_check,
)
from nemoguardrails.types import ToolCall

log = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

NOTHING_TO_CHECK_REASON = "No text to check; HiddenLayer was not called"


@dataclass(frozen=True)
class _Settings:
    """A HiddenLayerRailConfig with environment fallbacks and defaults applied."""

    api_url: str
    project_id: Optional[str]
    requester_id: str
    fail_open: bool
    max_retries: int
    max_retry_after_seconds: float
    retry_backoff_seconds: float
    send_session_id: bool


def _rail_config(config: Optional[RailsConfig]) -> HiddenLayerRailConfig:
    rails_config = getattr(config.rails, "config", None) if config is not None else None
    rail_config = getattr(rails_config, "hiddenlayer", None)
    return rail_config if isinstance(rail_config, HiddenLayerRailConfig) else HiddenLayerRailConfig()


def _field_or_env(value: Optional[str], env_name: str) -> Optional[str]:
    """Return the config value, else the environment variable; an empty string counts as unset."""
    return value or os.environ.get(env_name) or None


def _flag_or_env(value: Optional[bool], env_name: str, default: bool) -> bool:
    """Return the config value, else the environment variable, else *default*.

    The environment variable counts as unset when empty or not a recognized boolean.
    """
    if value is not None:
        return value
    env_value = (os.environ.get(env_name) or "").strip().lower()
    if env_value in ("1", "true", "yes", "on"):
        return True
    if env_value in ("0", "false", "no", "off"):
        return False
    return default


def _settings(rail_config: HiddenLayerRailConfig) -> _Settings:
    """Resolve each setting from the config field, then its environment variable, then the default.

    Raises:
        ValueError: If the resolved API URL is not https (or http on a loopback host). The config
            field is already checked at load; this covers HIDDENLAYER_BASE_URL.
    """
    api_url = (_field_or_env(rail_config.api_url, "HIDDENLAYER_BASE_URL") or DEFAULT_API_URL).rstrip("/")
    return _Settings(
        api_url=_check_api_url(api_url),
        project_id=_field_or_env(rail_config.project_id, "HIDDENLAYER_PROJECT_ID"),
        requester_id=_field_or_env(rail_config.requester_id, "HIDDENLAYER_REQUESTER_ID") or DEFAULT_REQUESTER_ID,
        fail_open=rail_config.fail_open,
        max_retries=rail_config.max_retries,
        max_retry_after_seconds=rail_config.max_retry_after_seconds,
        retry_backoff_seconds=rail_config.retry_backoff_seconds,
        send_session_id=_flag_or_env(rail_config.send_session_id, "HIDDENLAYER_SEND_SESSION_ID", True),
    )


def _main_model(config: Optional[RailsConfig]) -> Tuple[Optional[str], Optional[str]]:
    """Return the ``(model, engine)`` of the main model, or ``(None, None)`` without one.

    Both are optional on the routes, and HiddenLayer records an unknown model rather than a
    placeholder, so neither is invented here. ``engine`` is sent as the guarded model's vendor.
    """
    for model in config.models if config is not None else ():
        if model.type == "main":
            return model.model, model.engine
    return None, None


def _credentials() -> Tuple[str, str]:
    client_id = os.environ.get("HIDDENLAYER_CLIENT_ID")
    client_secret = os.environ.get("HIDDENLAYER_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "HiddenLayer credentials not found. Please set HIDDENLAYER_CLIENT_ID and HIDDENLAYER_CLIENT_SECRET."
        )
    return client_id, client_secret


def _retry_policy(settings: _Settings) -> RetryPolicy:
    # POST is resent because an evaluation is a stateless scan: HiddenLayer records the
    # interaction but a repeat of the same body changes nothing server-side, and the round-trip
    # ID is resolved once per check so a resend reports as the same evaluation.
    #
    # Transport errors are not retried. The setting covers timeouts and connection errors
    # together, and a rail runs inline, so retrying a timeout would cost another full
    # REQUEST_TIMEOUT_SECONDS before the caller hears anything. A connection error resolves to
    # the fail setting instead.
    max_delay = settings.max_retry_after_seconds
    return RetryPolicy(
        max_attempts=settings.max_retries + 1,
        retryable_methods=frozenset({"POST"}),
        retryable_status_codes=_RETRYABLE_STATUS_CODES,
        initial_delay=min(settings.retry_backoff_seconds, max_delay),
        max_delay=max_delay,
        max_retry_after=max_delay,
        retry_transport_errors=False,
        honor_retry_override_header=False,
        clamp_retry_after=True,
    )


async def _request_evaluation(
    client: Optional[HTTPClient],
    factory: Callable[[], ClosableHTTPClient],
    surface: Surface,
    settings: _Settings,
    provider: Optional[str],
    body: Dict[str, Any],
    client_id: str,
    client_secret: str,
) -> _Evaluation:
    """Fetch a token, post the evaluation to its route, refresh the token once on a 401, and read the 200."""
    url = f"{settings.api_url}{_SURFACE_PATH[surface]}"
    # Resolved once: a second attempt is the same evaluation of the same model call, not a new one.
    roundtrip_id = _roundtrip_id()
    session_id = get_conversation_id() if settings.send_session_id else None

    async def token(rejected: Optional[str] = None) -> str:
        return await _access_token(client, settings.api_url, client_id, client_secret, factory, rejected=rejected)

    async def evaluate(bearer: str):
        return await http_call(
            client,
            "POST",
            url,
            headers=_headers(
                bearer,
                roundtrip_id,
                settings.requester_id,
                provider,
                session_id=session_id,
                project_id=settings.project_id,
            ),
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
            raise_for_status=False,
            factory=factory,
        )

    bearer = await token()
    response = await evaluate(bearer)
    if response.status_code == 401:
        response = await evaluate(await token(rejected=bearer))
    if response.status_code != 200:
        raise HiddenLayerTransportError(
            f"HiddenLayer evaluation failed: {_describe(response)}", status_code=response.status_code
        )
    try:
        return _Evaluation(headers=response.headers, body=response.json())
    except HTTPResponseDecodeError:
        raise HiddenLayerTransportError(
            f"HiddenLayer evaluation response is not valid JSON: {_describe(response)}",
            status_code=response.status_code,
        ) from None


def _failure_text(error: Exception) -> str:
    # Only the rail's own messages are known to be free of request and response content.
    if isinstance(error, (HiddenLayerTransportError, HiddenLayerActionError, HiddenLayerVerdictError)):
        return str(error)
    return type(error).__name__


async def _evaluate(
    surface: Surface,
    provider: Optional[str],
    body: Dict[str, Any],
    checked_text: Optional[str],
    settings: _Settings,
    http_client: Optional[HTTPClient],
) -> RailOutcome:
    """Evaluate *body* with HiddenLayer and map the verdict for *surface*.

    Missing credentials raise ``ValueError`` regardless of ``fail_open``: a config defect never
    fails open. A transport failure, which includes any response that carries no action to apply,
    allows the request with a ``fail_open`` marker when ``fail_open`` is set, and raises
    :class:`HiddenLayerCheckError` otherwise. An evaluation HiddenLayer could not complete is not a transport
    failure: it comes back as a 200 carrying the verdict its deployment's fail setting chose, and
    that verdict is applied. So is an action the rail cannot carry out, which blocks.
    """
    client_id, client_secret = _credentials()
    policy = _retry_policy(settings)

    def factory() -> ClosableHTTPClient:
        return ensure_retrying_http_client(create_http_client(timeout=REQUEST_TIMEOUT_SECONDS), policy)

    # Wrapping an injected client adds retry without taking ownership of it; http_call closes
    # only what factory builds.
    client = ensure_retrying_http_client(http_client, policy) if http_client is not None else None
    try:
        evaluation = await _request_evaluation(
            client, factory, surface, settings, provider, body, client_id, client_secret
        )
        return _outcome(surface, evaluation, checked_text)
    except HiddenLayerVerdictError as error:
        log.warning("HiddenLayer %s check returned an unusable REDACT (%s); blocking.", surface, error)
        return RailOutcome.block(reason=BLOCK_REASON, metadata={"action": "REDACT"})
    except HiddenLayerUnknownActionError as error:
        log.warning("HiddenLayer %s check returned an action this rail cannot apply (%s); blocking.", surface, error)
        return RailOutcome.block(reason=BLOCK_REASON)
    except (HTTPClientError, HiddenLayerTransportError, HiddenLayerActionError) as error:
        status = getattr(error, "status_code", None)
        if settings.fail_open:
            # HiddenLayer never learns that NeMo allowed this request, so the log is its only record.
            log.warning(
                "HiddenLayer %s check could not be completed (%s, status=%s); fail_open is enabled, allowing content.",
                surface,
                type(error).__name__,
                status,
            )
            return RailOutcome.allow(reason=FAIL_OPEN_REASON, metadata={"fail_open": True})
        log.error(
            "HiddenLayer %s check could not be completed (%s, status=%s); fail_open is off, failing the request.",
            surface,
            type(error).__name__,
            status,
        )
        raise HiddenLayerCheckError(f"HiddenLayer {surface} check failed: {_failure_text(error)}") from None


@action(name="hiddenlayer_evaluate", is_system_action=True)
async def hiddenlayer_evaluate(
    surface: str,
    text: str,
    config: Optional[RailsConfig] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs: Any,
) -> RailOutcome:
    """Evaluates a user or bot message with HiddenLayer AI Runtime Security.

    Args:
        surface: Fixed per flow, "input" or "output".
        text: The user message (input) or bot message (output) to check.
        config: The active RailsConfig; used to resolve ``rails.config.hiddenlayer`` and the main model.
        http_client: Optional caller-owned HTTP client. The action does not close an injected client.

    Returns:
        BLOCK on a HiddenLayer BLOCK, TRANSFORM with HiddenLayer's redacted text on a REDACT,
        otherwise ALLOW. An empty message is allowed without a call, because a check carrying no
        text is rejected and there is nothing in it to evaluate. Outcomes allowed because
        HiddenLayer could not be reached include ``fail_open=True`` in their metadata.
    """
    if surface not in ("input", "output"):
        raise ValueError("surface must be 'input' or 'output'")
    if not text:
        return RailOutcome.allow(reason=NOTHING_TO_CHECK_REASON)
    settings = _settings(_rail_config(config))
    model, provider = _main_model(config)
    body = _input_check(model, text) if surface == "input" else _output_check(model, text)
    return await _evaluate(surface, provider, body, text, settings, http_client)


@action(name="hiddenlayer_evaluate_tool_call", is_system_action=True)
@tool_call_validation
async def hiddenlayer_evaluate_tool_call(
    tool_call: ToolCall,
    tool_definition: Optional[Tool],
    config: Optional[RailsConfig] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs: Any,
) -> RailOutcome:
    """Evaluates a tool call with HiddenLayer AI Runtime Security.

    Args:
        tool_call: The tool call to check, sent whole (name and arguments).
        tool_definition: The declared tool. Sent so policy rules can read what the model was
            offered, and used by @tool_call_validation to check the call's arguments against its
            schema before this function runs.
        config: The active RailsConfig; used to resolve ``rails.config.hiddenlayer`` and the main model.
        http_client: Optional caller-owned HTTP client. The action does not close an injected client.

    Returns:
        BLOCK on a HiddenLayer BLOCK or REDACT, otherwise ALLOW. HiddenLayer promotes a REDACT
        of a body carrying tool calls to a BLOCK, so a redaction is not expected here. Outcomes
        allowed because HiddenLayer could not be reached include ``fail_open=True`` in their metadata.
    """
    settings = _settings(_rail_config(config))
    model, provider = _main_model(config)
    body = _tool_call_check(model, tool_call, tool_definition)
    return await _evaluate("tool_call", provider, body, None, settings, http_client)


@action(name="hiddenlayer_evaluate_tool_result", is_system_action=True)
async def hiddenlayer_evaluate_tool_result(
    tool_call: ToolCall,
    tool_result: ToolResult,
    tool_definition: Optional[Tool] = None,
    config: Optional[RailsConfig] = None,
    http_client: Optional[HTTPClient] = None,
    **kwargs: Any,
) -> RailOutcome:
    """Evaluates a tool result, together with the call it answers, with HiddenLayer AI Runtime Security.

    Args:
        tool_call: The prior call this result answers.
        tool_result: The tool result to check.
        tool_definition: The declared tool. The engine builds a tool-result check without a
            toolset, so this is None in practice and nothing is sent.
        config: The active RailsConfig; used to resolve ``rails.config.hiddenlayer`` and the main model.
        http_client: Optional caller-owned HTTP client. The action does not close an injected client.

    Returns:
        BLOCK on a HiddenLayer BLOCK or REDACT, otherwise ALLOW. HiddenLayer promotes a REDACT
        of a body carrying tool results to a BLOCK, so a redaction is not expected here. Outcomes
        allowed because HiddenLayer could not be reached include ``fail_open=True`` in their metadata.
    """
    settings = _settings(_rail_config(config))
    model, provider = _main_model(config)
    body = _tool_result_check(model, tool_call, tool_result, tool_definition)
    return await _evaluate("tool_result", provider, body, None, settings, http_client)
