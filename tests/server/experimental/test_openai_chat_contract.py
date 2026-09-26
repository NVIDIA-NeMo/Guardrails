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
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml
from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).parents[3]
CONTRACTS_ROOT = REPOSITORY_ROOT / "nemoguardrails/server/experimental/contracts"
OPENAI_CONTRACTS_ROOT = CONTRACTS_ROOT / "openai"
SOURCE_PATH = OPENAI_CONTRACTS_ROOT / "source.yaml"
CHAT_CONTRACT_PATH = OPENAI_CONTRACTS_ROOT / "chat-completions.guard.yaml"
SCHEMA_PATH = CONTRACTS_ROOT / "guard-contract.schema.json"
EXTENSION = "x-nemo-guardrails"


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def chat_contract() -> dict[str, Any]:
    """Load the OpenAI Chat Completions guard contract."""
    return _load_yaml(CHAT_CONTRACT_PATH)


def test_openai_source_is_immutable_and_content_addressed() -> None:
    """The OpenAI description is pinned to an immutable revision and digest."""
    source = _load_yaml(SOURCE_PATH)

    assert set(source) == {
        "document_url",
        "download_url",
        "revision",
        "document_version",
        "sha256",
    }
    assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
    assert re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
    assert source["revision"] in source["document_url"]
    assert source["revision"] in source["download_url"]
    assert urlparse(source["document_url"]).scheme == "https"
    assert urlparse(source["download_url"]).scheme == "https"


def test_openai_chat_contract_matches_authoring_schema(
    chat_contract: dict[str, Any],
) -> None:
    """The OpenAI Chat contract conforms to the guard-contract authoring schema."""
    schema = json.loads(SCHEMA_PATH.read_text())

    Draft202012Validator(schema).validate(chat_contract)


def test_openai_chat_contract_is_one_cohesive_operation(
    chat_contract: dict[str, Any],
) -> None:
    """One contract owns the request, buffered response, stream, and binding."""
    assert set(chat_contract) == {
        "version",
        "operationId",
        "profile",
        "request",
        "response",
        "stream",
        "integration",
    }
    assert chat_contract["operationId"] == "createChatCompletion"
    assert chat_contract["profile"] == "single_text.v1"
    assert "overlay" not in chat_contract
    assert "actions" not in chat_contract


def test_openai_chat_contract_declares_replaceable_text_subjects(
    chat_contract: dict[str, Any],
) -> None:
    """Request and response policy identifies replaceable user and assistant text."""
    request_content = chat_contract["request"]["properties"]["messages"]["items"]["properties"]["content"]
    response_content = chat_contract["response"]["properties"]["choices"]["items"]["properties"]["message"][
        "properties"
    ]["content"]

    assert request_content[EXTENSION]["subject"] == {
        "kind": "text",
        "role": "user",
        "replaceable": True,
    }
    assert response_content[EXTENSION]["subject"] == {
        "kind": "text",
        "role": "assistant",
        "replaceable": True,
        "replacement_blocked_by": "annotations",
        "replacement_reason": "provider_integrity.annotated_text",
    }


def test_openai_chat_contract_declares_stream_transport_and_hooks(
    chat_contract: dict[str, Any],
) -> None:
    """Stream policy binds SSE framing and the handwritten lifecycle hook."""
    transport = chat_contract["stream"][EXTENSION]["transport"]
    endpoint = chat_contract["integration"]["endpoint"]

    assert transport == {
        "require_sse_event": False,
        "non_data_shape": "[DONE]",
        "sentinels": {"[DONE]": "[DONE]"},
    }
    assert endpoint["route_path"] == "/v1/chat/completions"
    assert endpoint["stream_hooks"] == {
        "module": "nemoguardrails.server.experimental.providers.openai.chat_completions.stream_hooks",
        "name": "ChatCompletionsStreamHooks",
    }
