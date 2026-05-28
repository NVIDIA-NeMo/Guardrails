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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from vcr.util import read_body

from tests.recorded.cassette import cassette_with_parsed_bodies, cassette_with_rehydrated_bodies
from tests.recorded.sanitization import (
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
        return cassette_with_rehydrated_bodies(yaml.safe_load(cassette_string))

    @staticmethod
    def serialize(cassette_dict: dict[str, Any]) -> str:
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
                normalize_response_metadata=False,
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
    body_1 = read_body(request_1)
    body_2 = read_body(request_2)
    json_body_1 = _decode_match_body_json(body_1)
    json_body_2 = _decode_match_body_json(body_2)

    if json_body_1 is not _NON_JSON_BODY and json_body_2 is not _NON_JSON_BODY:
        scrubbed_1 = _scrub_request_json(json_body_1)
        scrubbed_2 = _scrub_request_json(json_body_2)
        if scrubbed_1 != scrubbed_2:
            raise AssertionError(f"{scrubbed_1!r} != {scrubbed_2!r}")
        return

    if _normalize_raw_match_body(body_1) != _normalize_raw_match_body(body_2):
        raise AssertionError


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
                data = _scrub_json(data, normalize_response_metadata=True)
                lines.append("data: " + json.dumps(data, separators=(",", ":")))
            else:
                lines.append(line)
        events.append("\n".join(lines))

    return _encode_text_like(body, "\n\n".join(events) + "\n\n")


def before_record_request(request: Any) -> Any:
    _replace_case_insensitive(request.headers, FILTERED_HEADERS)

    try:
        data = _decode_json_body(request.body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return request

    if isinstance(data, dict):
        request.body = _encode_body_like(request.body, _scrub_request_json(data))
    return request


def before_record_response(response: Dict[str, Any]) -> Dict[str, Any]:
    headers = response.get("headers", {})
    _replace_case_insensitive(headers, FILTERED_HEADERS | VOLATILE_RESPONSE_HEADERS)

    body = response.get("body", {}).get("string")
    if body is None:
        return response

    content_types = [value.lower() for value in _header_values(headers, "content-type")]
    if any("text/event-stream" in value for value in content_types):
        response["body"]["string"] = _scrub_sse_body(body)
        return response

    try:
        data = _decode_json_body(body)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return response

    scrubbed = _scrub_response_json(data)
    response["body"]["string"] = _encode_body_like(body, scrubbed)
    return response


def pytest_recording_configure(config: pytest.Config, vcr: Any) -> None:
    vcr.register_serializer("yaml", ReadableYamlSerializer)
    vcr.register_matcher("recorded_body", recorded_body_matcher)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = request.node.fspath
    return str(Path(module.dirname) / "cassettes" / module.purebasename)


@pytest.fixture
def recorded_cassette_path(vcr_cassette_dir: str, default_cassette_name: str) -> Path:
    return Path(vcr_cassette_dir) / f"{default_cassette_name}.yaml"


def build_vcr_config() -> Dict[str, Any]:
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


def recording_credentials(
    request: pytest.FixtureRequest,
    required_llm_providers: tuple[str, ...],
    required_env_vars: tuple[str, ...] = (),
) -> None:
    """Activate API-key fixtures for every required provider and arbitrary service env var.

    In replay mode the fixtures install dummy values; in record mode they require
    the real environment variables and skip the test if any are missing.
    """
    for provider in required_llm_providers:
        request.getfixturevalue(_provider_key_fixture_name(provider))
    service_api_key = request.getfixturevalue("service_api_key")
    for env_name in required_env_vars:
        service_api_key(env_name)
