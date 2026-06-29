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
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Dict, List

import pytest
import pytest_asyncio
import yaml
from vcr.util import read_body

from tests.recorded.cassette import cassette_with_parsed_bodies, cassette_with_rehydrated_bodies, normalize_body
from tests.recorded.sanitization import (
    ALLOWED_HEADERS,
    FILTERED_HEADER_PREFIXES,
    FILTERED_HEADERS,
    FILTERED_QUERY_PARAMETERS,
    JSON_SECRET_KEYS,
    NULLABLE_VOLATILE_RESPONSE_JSON_FIELDS,
    SECRET_PATTERNS,
    VOLATILE_RESPONSE_HEADERS,
    VOLATILE_RESPONSE_JSON_FIELDS,
    VOLATILE_RESPONSE_METADATA_FIELDS,
)
from tests.recorded.utils import (
    DUMMY_NVIDIA_API_KEY,
    DUMMY_OPENAI_API_KEY,
    set_api_key_for_record_mode,
)

DUMMY_SERVICE_API_KEY = "recorded-replay"
_NON_JSON_BODY = object()


class _ReadableCassetteDumper(yaml.SafeDumper):
    pass


def _represent_readable_string(dumper: yaml.SafeDumper, data: str) -> yaml.nodes.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_ReadableCassetteDumper.add_representer(str, _represent_readable_string)


class ReadableYamlSerializer:
    @staticmethod
    def deserialize(cassette_string: str) -> Any:
        """Restore readable cassette bodies to the raw shape VCR expects."""
        return cassette_with_rehydrated_bodies(yaml.safe_load(cassette_string))

    @staticmethod
    def serialize(cassette_dict: dict[str, Any]) -> str:
        """Write cassettes with parsed JSON bodies and stable YAML formatting."""
        return yaml.dump(
            cassette_with_parsed_bodies(cassette_dict),
            Dumper=_ReadableCassetteDumper,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )


def _replace_case_insensitive(headers: Dict[str, Any], header_names: set[str], value: Any = None) -> None:
    for name in list(headers):
        if name.lower() in header_names:
            if value is None:
                del headers[name]
            else:
                headers[name] = value


def _filter_headers_by_prefix(headers: Dict[str, Any]) -> None:
    for name in list(headers):
        lowered = name.lower()
        if lowered in ALLOWED_HEADERS:
            continue
        if any(lowered.startswith(prefix) for prefix in FILTERED_HEADER_PREFIXES):
            del headers[name]


def _scrub_text(value: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def _normalize_response_metadata(key: str, value: Any) -> Any:
    if key in VOLATILE_RESPONSE_JSON_FIELDS:
        return VOLATILE_RESPONSE_JSON_FIELDS[key]
    if key in NULLABLE_VOLATILE_RESPONSE_JSON_FIELDS and value is not None:
        return NULLABLE_VOLATILE_RESPONSE_JSON_FIELDS[key]
    return value


def _scrub_json(value: Any, *, normalize_response_metadata: bool = False) -> Any:
    if isinstance(value, dict):
        scrubbed = {}
        for key, nested in value.items():
            if normalize_response_metadata and key in VOLATILE_RESPONSE_METADATA_FIELDS:
                scrubbed[key] = _normalize_response_metadata(key, nested)
            elif key.lower() in JSON_SECRET_KEYS:
                scrubbed[key] = "[REDACTED]"
            else:
                scrubbed[key] = _scrub_json(
                    nested,
                    normalize_response_metadata=False,
                )
        return scrubbed
    if isinstance(value, list):
        return [
            _scrub_json(
                item,
                normalize_response_metadata=normalize_response_metadata,
            )
            for item in value
        ]
    if isinstance(value, str):
        return _scrub_text(value)
    return value


def _decode_json_body(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, (dict, list)):
        return body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    return None


def _decode_match_body_json(body: Any) -> Any:
    try:
        return _decode_json_body(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return _NON_JSON_BODY


def _normalize_raw_match_body(body: Any) -> Any:
    if isinstance(body, bytearray):
        return bytes(body)
    return body


def recorded_body_matcher(request_1: Any, request_2: Any) -> None:
    """Compare recorded and replay requests after applying the same scrubbing rules.

    This keeps replay strict about semantic request changes while ignoring
    redacted secrets and normalization that are intentionally applied at record
    time.
    """
    body_1 = read_body(request_1)
    body_2 = read_body(request_2)
    json_body_1 = _decode_match_body_json(body_1)
    json_body_2 = _decode_match_body_json(body_2)

    if json_body_1 is not _NON_JSON_BODY and json_body_2 is not _NON_JSON_BODY:
        matched_1 = normalize_body(_scrub_request_json(json_body_1))
        matched_2 = normalize_body(_scrub_request_json(json_body_2))
        if matched_1 != matched_2:
            raise AssertionError(
                "Recorded request JSON body does not match replay request after scrubbing and normalization:\n"
                f"recorded={matched_1!r}\n"
                f"replay={matched_2!r}"
            )
        return

    matched_1 = _normalize_raw_match_body(body_1)
    matched_2 = _normalize_raw_match_body(body_2)
    if matched_1 != matched_2:
        raise AssertionError(
            f"Recorded raw request body does not match replay request:\nrecorded={matched_1!r}\nreplay={matched_2!r}"
        )


def _encode_body_like(original_body: Any, data: Any) -> Any:
    body = json.dumps(data, indent=2)
    if isinstance(original_body, bytes):
        return body.encode("utf-8")
    if isinstance(original_body, str):
        return body
    return data


def _body_to_text(body: Any) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8")
    return body if isinstance(body, str) else ""


def _encode_text_like(original_body: Any, text: str) -> Any:
    return text.encode("utf-8") if isinstance(original_body, bytes) else text


def _scrub_raw_body(body: Any) -> Any:
    try:
        text = _body_to_text(body)
    except UnicodeDecodeError:
        return body
    if not text:
        return body
    scrubbed = _scrub_text(text)
    if scrubbed == text:
        return body
    return _encode_text_like(body, scrubbed)


def _header_values(headers: dict[str, Any], name: str) -> list[str]:
    for key, value in headers.items():
        if key.lower() == name:
            return value if isinstance(value, list) else [value]
    return []


def _scrub_request_json(data: Any) -> Any:
    return _scrub_json(data)


def _scrub_response_json(data: Any) -> Any:
    scrubbed = _scrub_json(
        data,
        normalize_response_metadata=True,
    )
    if isinstance(scrubbed, dict) and {"jailbreak", "score"} <= set(scrubbed):
        scrubbed["score"] = 0.0
    return scrubbed


def _scrub_sse_body(body: Any) -> Any:
    text = _body_to_text(body)
    if not text:
        return body

    events = []
    for event in text.split("\n\n"):
        if not event:
            continue
        lines = []
        for line in event.splitlines():
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                try:
                    data = json.loads(line.removeprefix("data: "))
                except json.JSONDecodeError:
                    lines.append(line)
                    continue
                data = _scrub_response_json(data)
                lines.append("data: " + json.dumps(data, separators=(",", ":")))
            else:
                lines.append(line)
        events.append("\n".join(lines))

    return _encode_text_like(body, "\n\n".join(events) + "\n\n")


def before_record_request(request: Any) -> Any:
    """Redact request headers and bodies before VCR writes a cassette."""
    _replace_case_insensitive(request.headers, FILTERED_HEADERS)
    _filter_headers_by_prefix(request.headers)

    try:
        data = _decode_json_body(request.body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        request.body = _scrub_raw_body(request.body)
        return request

    if data is not None:
        request.body = _encode_body_like(request.body, _scrub_request_json(data))
    return request


def before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Redact volatile response headers and bodies before VCR writes a cassette."""
    headers = response.get("headers", {})
    _replace_case_insensitive(headers, FILTERED_HEADERS | VOLATILE_RESPONSE_HEADERS)
    _filter_headers_by_prefix(headers)

    body_container = response.get("body")
    if not isinstance(body_container, dict):
        return response

    body = body_container.get("string")
    if body is None:
        return response

    content_types = [value.lower() for value in _header_values(headers, "content-type")]
    if any("text/event-stream" in value for value in content_types):
        body_container["string"] = _scrub_sse_body(body)
        return response

    try:
        data = _decode_json_body(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        body_container["string"] = _scrub_raw_body(body)
        return response

    scrubbed = _scrub_response_json(data)
    body_container["string"] = _encode_body_like(body, scrubbed)
    return response


def pytest_recording_configure(config: pytest.Config, vcr: Any) -> None:
    vcr.register_serializer("yaml", ReadableYamlSerializer)
    vcr.register_matcher("recorded_body", recorded_body_matcher)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = request.node.path
    return str(module.parent / "cassettes" / module.stem)


@pytest.fixture
def recorded_cassette_path(vcr_cassette_dir: str, default_cassette_name: str) -> Path:
    return Path(vcr_cassette_dir) / f"{default_cassette_name}.yaml"


def build_vcr_config() -> Dict[str, Any]:
    """Build the shared VCR config used by all recorded tests."""
    return {
        "decode_compressed_response": True,
        "filter_headers": [(name, None) for name in FILTERED_HEADERS],
        "filter_query_parameters": [(name, None) for name in FILTERED_QUERY_PARAMETERS],
        "before_record_request": before_record_request,
        "before_record_response": before_record_response,
        "match_on": ["method", "scheme", "host", "port", "path", "query", "recorded_body"],
    }


_VCR_CONFIG = build_vcr_config()


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return _VCR_CONFIG


@pytest_asyncio.fixture(autouse=True)
async def close_owned_http_clients(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    from nemoguardrails.llm.clients import base

    tracked: List[Any] = []
    original_init = base.BaseClient.__init__

    def tracking_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "_owns_client", False):
            tracked.append(self)

    monkeypatch.setattr(base.BaseClient, "__init__", tracking_init)

    yield

    leaked = [client for client in tracked if client._owns_client and not client._client.is_closed]
    for client in leaked:
        await client._client.aclose()


_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
    "NO_PROXY",
    "no_proxy",
)


@pytest.fixture(autouse=True)
def strip_proxy_env_during_replay(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> None:
    """Make replay independent of the ambient proxy configuration.

    Under ``--block-network`` a proxy is useless, and a SOCKS proxy is fatal:
    httpx raises ``ImportError`` when ``socksio`` is not installed, turning a
    cassette hit into an error that depends only on the developer or CI shell.
    Strip proxy variables during replay so a hit is deterministic everywhere.
    Recording keeps the ambient proxy so real provider calls can still egress.
    """
    if record_mode == "none":
        for name in _PROXY_ENV_VARS:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> str:
    return set_api_key_for_record_mode(monkeypatch, "OPENAI_API_KEY", DUMMY_OPENAI_API_KEY, record_mode)


@pytest.fixture
def nvidia_api_key(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> str:
    return set_api_key_for_record_mode(monkeypatch, "NVIDIA_API_KEY", DUMMY_NVIDIA_API_KEY, record_mode)


@pytest.fixture
def service_api_key(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> Callable[[str], str]:
    def set_service_api_key(env_name: str) -> str:
        return set_api_key_for_record_mode(monkeypatch, env_name, DUMMY_SERVICE_API_KEY, record_mode)

    return set_service_api_key


_PROVIDER_KEY_FIXTURES = {"openai": "openai_api_key", "nim": "nvidia_api_key"}


def _provider_key_fixture_name(provider: str) -> str:
    fixture_name = _PROVIDER_KEY_FIXTURES.get(provider)
    if fixture_name is not None:
        return fixture_name

    supported = ", ".join(sorted(_PROVIDER_KEY_FIXTURES))
    raise ValueError(f"Unknown recorded provider {provider!r}; expected one of: {supported}")


def provider_key(request: pytest.FixtureRequest, provider: str) -> None:
    """Activate the API-key fixture for one LLM provider (``openai`` or ``nim``)."""
    request.getfixturevalue(_provider_key_fixture_name(provider))
