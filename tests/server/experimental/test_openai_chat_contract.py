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

"""Validate Chat contract exports and pinned provider metadata offline."""

import json
import re
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import Field

from nemoguardrails.server.experimental.provider.projection_policy import (
    CONTRACT_VERSION,
    PolicyModel,
    constrained,
    export_payload_schema,
    guarded,
)
from nemoguardrails.server.experimental.providers.openai import source as openai_pin
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_binding import (
    PAYLOAD_CONTRACT as REQUEST_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.request_projection import (
    ChatCompletionsGuardedRequestProjection,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_binding import (
    PAYLOAD_CONTRACT as RESPONSE_CONTRACT,
)
from nemoguardrails.server.experimental.providers.openai.chat_completions.response_projection import (
    ChatCompletionsGuardedResponseProjection,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
SOURCE_PATH = REPOSITORY_ROOT / "nemoguardrails/server/experimental/contracts/openai/source.yaml"


@pytest.fixture(scope="module")
def openai_source() -> dict[str, str]:
    """Load source provenance, not a separately authored operation policy."""
    source = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(source, dict)
    return source


def test_openai_source_declares_revision_and_digest(openai_source: dict[str, str]) -> None:
    """Pin metadata has the expected shape; this does not verify upstream bytes."""
    assert set(openai_source) == {
        "document_url",
        "download_url",
        "revision",
        "document_version",
        "sha256",
    }
    assert re.fullmatch(r"[0-9a-f]{40}", openai_source["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", openai_source["sha256"])
    assert openai_source["document_version"].strip()


@pytest.mark.parametrize(
    ("key", "host", "prefix"),
    [
        ("document_url", "github.com", "/openai/openai-openapi/blob"),
        ("download_url", "raw.githubusercontent.com", "/openai/openai-openapi"),
    ],
)
def test_openai_source_urls_use_the_declared_revision(
    openai_source: dict[str, str], key: str, host: str, prefix: str
) -> None:
    """Both locations identify the same pinned file rather than a moving branch."""
    url = urlparse(openai_source[key])

    assert url.scheme == "https"
    assert url.netloc == host
    assert url.path == f"{prefix}/{openai_source['revision']}/openapi.yaml"
    assert not url.query
    assert not url.fragment


def test_python_pin_matches_source_metadata(openai_source: dict[str, str]) -> None:
    """The importable pin repeats source.yaml exactly, so the two cannot drift."""
    assert {
        "document_url": openai_pin.PROVIDER_DOCUMENT_URL,
        "download_url": openai_pin.PROVIDER_DOWNLOAD_URL,
        "revision": openai_pin.PROVIDER_REVISION,
        "document_version": openai_pin.PROVIDER_DOCUMENT_VERSION,
        "sha256": openai_pin.PROVIDER_DOCUMENT_SHA256,
    } == openai_source


GUARD_CONTRACT_SCHEMA = (
    Path(__file__).parents[3] / "nemoguardrails/server/experimental/contracts/guard-contract.schema.json"
)


def _contract_errors(request: type[PolicyModel], response: type[PolicyModel]) -> list[str]:
    """Wrap two payload exports in a minimal operation contract and validate it."""
    contract = {
        "version": CONTRACT_VERSION,
        "operationId": "createChatCompletion",
        "profile": "single_text.v1",
        "request": export_payload_schema(request, projection_id=REQUEST_CONTRACT.projection_id),
        "response": export_payload_schema(response, projection_id=RESPONSE_CONTRACT.projection_id),
        "integration": {
            "endpoint": {
                "unsupported_request_code": "unsupported_request",
                "unsupported_response_code": "unsupported_response",
            }
        },
    }
    validator = Draft202012Validator(json.loads(GUARD_CONTRACT_SCHEMA.read_text(encoding="utf-8")))
    return [error.message for error in validator.iter_errors(contract)]


def test_chat_exports_conform_to_the_guard_contract_schema():
    assert _contract_errors(ChatCompletionsGuardedRequestProjection, ChatCompletionsGuardedResponseProjection) == []


def test_contract_schema_check_catches_unexportable_constraints():
    class Response(PolicyModel):
        text: Annotated[str, guarded("assistant")]
        count: Annotated[int, Field(ge=0), constrained()]

    assert _contract_errors(ChatCompletionsGuardedRequestProjection, Response)
